# Histomorphology-driven multi-instance learning for breast cancer WSI classification




![](./figs/p12.png)
![](./figs/zhutu.png)
## NEWS


## Abstract

> Each gigapixel whole slide image (WSI) contains tens of thousands of patches, many of which carry redundant information, resulting in substantial computational and storage overhead.This motivates the need for automatic \textbf{WSI Sum}marization (\textbf{WSISum}), which aims to extract a compact subset of patches that can effectively approximate the original WSI. In this paper, we propose a {WSISum} framework that integrates low-level reconstruction with clustering-based sparse sampling and high-level reconstruction with knowledge distillation from multiple WSI-level foundation model, aiming to generate WSI summarization that preserve both local and global semantic information. Experimental results show that WSISum achieves satisfactory performance in a variety of downstream tasks, including molecular subtyping, cancer subtyping, and metastasis subtyping, while significantly reducing computational cost.

## NOTES

**2025-06-24**: We released the full version of WSISum, including models and train scripts.

## Installation
* Environment: CUDA 11.8 / Python 3.10
* Create a virtual environment
```shell
> conda create -n wsisum python=3.10 -y
> conda activate wsisum
```

* Install HMD
```shell
> git clone https://github.com/Badgewho/WSISum.git
> cd WSISum
> pip install requirements.txt



## How to Train
### Prepare your data
1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/) and [BRACS](https://www.bracs.icar.cnr.it/) and [CAMELYON17](https://camelyon17.grand-challenge.org/Data/)

2. Use the WSI processing tool provided by [CONCHv1.5](https://github.com/mahmoodlab/CONCH) to extract pretrained feature for each 256 $\times$ 256 patch (20x), which we then save as `.h5` files for each WSI. So, we get one `h5_files` folder storing `.h5` files for all WSIs of one study.

3. Use the slide embedding provided by [Prov-gigapath](https://github.com/prov-gigapath/prov-gigapath), [CHIEF](https://github.com/hms-dbmi/CHIEF), [PRISM](https://huggingface.co/paige-ai/Prism),  [Virchow](https://huggingface.co/paige-ai/Virchow)

The final structure of datasets should be as following:
```bash
DATA_ROOT_DIR/
    └──h5_files/
        └──dataset1/
            ├── slide_1.h5
            ├── slide_2.h5
            └── ...

```


run the following code for creating train dataset

```shell
python ./creatdataset/creat_dataset.py
python ./creatdataset/creat_traindataset_multi.py
```
run the following code for creating evaluate dataset

```shell
python ./creatdataset/creat_evaldatset_multi.py
```
run the following code for training WSISum 

```shell
python ./WSISum/run_mae_pretraining.py
```

run the following code for inference 

```shell
python ./WSISum/run_wsisummary.py
```

run the following code for benchmark 

```shell
python ./benchmark/creat_dataset_random.py
python ./benchmark/creat_dataset_cluster.py
python ./benchmark/creat_dataset_entropy.py
python ./benchmark/creat_dataset_location.py
```

## Visualization of WSISum with different benchmarks

  <div style="flex: 1; margin: 5px;">
    <img src="./figs/heatmap.png" alt="Heatmap" style="width: 100%;">
  </div>

## Interpretability

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; margin: 5px;">
    <img src="./figs/lizi2.png" alt="Image 1" style="width: 100%;">
  </div>
  <div style="flex: 1; margin: 5px;">
    <img src="./figs/kejieshi.png" alt="Image 2" style="width: 100%;">
  </div>

</div>

## Acknowledgements
This work was supported by National Natural Science Foundation of China (62402473, 62271465), and Suzhou Basic Research Program (SYG202338).


## License & Citation 
<!-- If you find our work useful in your research, please consider citing our paper at:

```text

``` -->
