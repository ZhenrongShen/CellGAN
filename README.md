# CellGAN-for-Cervical-Cell-Synthesis
Official Pytorch Implementation for "CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification". 

### Method
![Overview of CellGAN](/figures/overview.png "Overview of CellGAN")

CellGAN synthesizes 256×256 cytopathological images of different cervical squamous cell types (`NILM, ASC-US, LSIL, ASC-H, and HSIL`). It serves as an data augmentation tool for automatic cervical abnormality screening.

### Qualitative Results
![Visualization Results](/figures/results.png "Visualization Results")

### Environment
- Python 3.10.10
- Pytorch 2.0.0+cu117
- opencv-python, scikit-image, tqdm

### Quick Start

1. We provide a pre-trained CellGAN generator `checkpoints/model.pth` for synthesizing cytopathological images.

2. Use the following command to synthesize a certain number of images for a desired cervical cell type.

```python
python cellgan_inference.py --config [config_name] --model [model_path] --output_dir [directory to save generated images] --cell_type [desired cell type] --data_num [number of generated images]
```

## Usage
### Data Preparation
- In `DATAROOT`, split your images into different subdirectories according to the cell types and prepare a `img_list.txt`. 

- The directory structure of `DATAROOT` should be prepared as in the following example: 

```
DATAROOT
├─ NILM
|  ├─ NILM_image_0001.png
|  └─ ......
├─ ASC_US
|  ├─ ASC_US_image_0001.png
|  └─ ......
├─ LSIL
|  ├─ LSIL_image_0001.png
|  └─ ......
├─ ASC_H
|  ├─ ASC_H_image_0001.png
|  └─ ......
├─ HSIL
|  ├─ HSIL_image_0001.png
|  └─ ......
└─ img_list.txt
```

- The TXT file `img_list.txt` should contain one image path `{category_name}/{image_name}.png` per line as in the following example.

```
NILM/NILM_image_0001.png
NILM/NILM_image_0002.png
......
ASC_US/ASC_US_image_0001.png
......
```

- Set the argument `DATAROOT` in `configs/default_config.yaml` to your training data root. 

### Training
Refer to `configs/default_config.yaml` for customizing your own configuration file `configs/{config_name}.yaml`. All the arguments are self-explanatory by their names and comments. Use the following command:

```
python train.py --config [config_name]
```

### Testing
Edit the testing arguments in `configs/{config_name}.yaml` and use the following command:

```
python test.py --config [config_name]
```

## Literature Information
**Authors:**   
> Zhenrong Shen[1], Maosong Cao[2], Sheng Wang[1], Lichi Zhang[1], Qian Wang[2]*
> 
**Institution:**
> [1] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
> 
> [2] School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
