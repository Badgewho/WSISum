import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import h5py

from PIL import Image

from pathlib import Path

from timm.models import create_model

import utils
import modeling_pretrain
from datasets import DataAugmentationForMAE

from torchvision.transforms import ToPILImage
from einops import rearrange


def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('--img_dir', type=str, default='/data/Summary/eval_dataset/MOEab500', help='input image directory')
    parser.add_argument('--save_dir', type=str, default='/data/Summary/summaryset/CAMELYON/500-0.875-TITAN', help='directory to save summaries')
    parser.add_argument('--model_path', type=str, default='/data/Summary/MAE-pytorch/result/ckp750-0.917-multi/checkpoint-239.pth', help='checkpoint path of model')
    parser.add_argument('--input_size', default=224, type=int, help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.875, type=float, help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='pretrain_mae_base_patch16_224', type=str, metavar='MODEL', help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT', help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )
    return model


def process_file(args, file_path, model, device):
    with h5py.File(file_path, 'r') as f:
        img1 = f['features'][:].copy()

    transforms = DataAugmentationForMAE(args)
    img1, bool_masked_pos = transforms(img1)
    WSIfeature = img1[:, -1, :]
    WSIfeature = WSIfeature.to(device)
    img = img1[:, :-1, :]

    bool_masked_pos = torch.from_numpy(bool_masked_pos)
    summary = img[:, ~bool_masked_pos[1:].bool(), :]
    summary = summary.squeeze(0)

    max_iterations = 1000
    min_iterations = 200
    threshold = 0.98

    best_summary = None
    max_cosine_similarity = -1.0

    for i in range(max_iterations):
        with torch.no_grad():
            bool_masked_pos_batch = bool_masked_pos[None, :].to(device, non_blocking=True).flatten(1).to(torch.bool)
            img_batch = img.to(device, non_blocking=True)
            outputs, cls = model(img_batch, bool_masked_pos_batch)

            dot_product = torch.sum(cls * WSIfeature, dim=1)
            norm_cls = torch.norm(cls, p=2, dim=1)
            norm_WSIfeature = torch.norm(WSIfeature, p=2, dim=1)
            cosine_similarity = dot_product / (norm_cls * norm_WSIfeature)

            # 更新最优
            if cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = cosine_similarity
                # summary[0] = cls
                best_summary = summary

            # 只有当迭代次数 >= min_iterations 时，才允许提前退出
            if i >= (min_iterations - 1) and max_cosine_similarity.item() >= threshold:
                print(f"Early stopping at iteration {i+1} with similarity: {max_cosine_similarity.item():.4f}")
                break

    return best_summary, max_cosine_similarity



def main(args):
    print(args)

    device = torch.device(args.device)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = 64
    print("Patch size = %s" % str(patch_size))
    args.window_size = 64
    args.patch_size = patch_size

    model.to(device)
    checkpoint = torch.load(args.model_path, map_location='cuda')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Iterate over all datasets
    dataset_dirs = Path(args.img_dir).glob('*')
    for dataset_dir in dataset_dirs:
        if dataset_dir.is_dir():
            save_path = Path(args.save_dir) / dataset_dir.name
            save_path.mkdir(parents=True, exist_ok=True)
            dataset_dir = os.path.join(dataset_dir, 'cluster500')

            # Process each .h5 file in the dataset directory
            h5_files = Path(dataset_dir).glob('*.h5')
            for h5_file in h5_files:
                print(f"Processing {h5_file}...")

                # Check if the summary file already exists
                save_file_path = save_path / f"{h5_file.stem}.h5"
                if save_file_path.exists():
                    print(f"Skipping {h5_file} because the summary already exists.")
                    continue

                best_summary, max_cosine_similarity = process_file(args, h5_file, model, device)
                # Save the best summary to the save directory as a new .h5 file
                with h5py.File(save_file_path, 'w') as hf:
                    hf.create_dataset('features', data=best_summary.cpu().numpy())

                print(f"Saved best summary for {h5_file.name} to {save_file_path} with cosine similarity: {max_cosine_similarity.item():.4f}")

if __name__ == '__main__':
    opts = get_args()
    main(opts)
