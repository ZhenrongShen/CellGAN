# CellGAN-for-Cervical-Cell-Synthesis
Official Pytorch Implementation for "CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification". CellGAN can synthesize 256×256 cytopathological images of different cervical squamous cell types including NILM, ASC-US, LSIL, ASC-H, and HSIL cells. It serves as an data augmentation tool for automatic cervical abnormality screening.

### Environment
- Python 3.10.10
- Pytorch 2.0.0+cu117
- opencv-python, scikit-image, tqdm

### Quick Start

1. We provide a pre-trained model `checkpoints/model.pth` for synthesizing cytopathological images.

2. Use the following command for synthesizing a certain number of image of a desired cervical cell type.

```python
python cellgan_inference.py --config [config_name] --model [model_path] --output_dir [directory to save generated images] --cell_type [desired cell type] --data_num [number of generated images]
```

## Reproducing Experiments
### Data Preparation
- Prepare your data in `DATAROOT` as: 

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

- The TXT file `img_list.txt` should contain one image path '{category_name}/{image_name}' per line as in the following example.

```
NILM/NILM_image_0001.png
NILM/NILM_image_0002.png
......
ASC_US/ASC_US_image_0001.png
......
```

- Set the argument `DATAROOT` in `configs/default_config.yaml` to your training data root. 

### Training
```
python train.py --config [config_name]
```

### Testing
```
python test.py c
```

**Authors:**   
> Zhenrong Shen[1], Maosong Cao[2], Sheng Wang[1], Lichi Zhang[1], Qian Wang[2]*
> 
**Institution:**
> [1] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
> 
> [2] School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
