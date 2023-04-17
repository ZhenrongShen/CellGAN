# CellGAN-for-Cervical-Cell-Synthesis
Official Pytorch Implementation for "CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification"

### Environment
- Python 3.10.10
- Pytorch 2.0.0+cu117
- opencv-python, scikit-image, tqdm

### Quick Start

1. We provide a pre-trained model `checkpoints/model.pth` for synthesizing cytopathological images of various cervical squamous cell types, including NILM, ASC-US, LSIL, ASC-H, and HSIL.

2. Use the following command for synthesizing a single image of a specified cell type.

```python
python single_test.py --gpu [GPU] --model_path [model_path] --save_dir [set a dir to save your images] --cell_type [set your desired cell type] 
```

## Reproducing Experiments
### Data Preparation
- Split your training data into different directories according to cell types. 

- Write the paths of images to `img_list.txt` as in the following example.

```
NILM/case1.png
NILM/case2.png
ASC_US/case3.png
LSIL/case4.png
......
```

- Edit `configs/default_config.yaml `, change `DATAROOT: DATA/cell_patches` to your data root. 

### Training
```python
python train.py --config [config_name.yaml]
```

### Testing
```python
python test.py --config [config_name.yaml]
```

**Authors:**   
> Zhenrong Shen[1], Maosong Cao[2], Sheng Wang[1], Lichi Zhang[1], Qian Wang[2]
> 
**Institution:**
> [1] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
> [2] School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
