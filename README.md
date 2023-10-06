# CellGAN-for-Cervical-Cell-Synthesis
Official Pytorch Implementation for "CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification" (Early Accepted in MICCAI 2023 https://link.springer.com/chapter/10.1007/978-3-031-43987-2_47)

### Method
![Overview of CellGAN](/figures/overview.png "Overview of CellGAN")

CellGAN synthesizes 256×256 cytopathological images of different cervical squamous cell types (`NILM, ASC-US, LSIL, ASC-H, and HSIL`). It can serve as a data augmentation tool for patch-level cell classification in automatic cervical abnormality screening.

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
### Training
- Refer to `configs/default_config.yaml` for customizing your own configuration file `configs/{config_name}.yaml`. All the arguments are self-explanatory by their names and comments.

- Set the argument `DATAROOT` in `configs/{config_name}.yaml` to your training data root. 

- In `DATAROOT`, split your images into subdirectories according to the cell types and prepare an `img_list.txt`. 

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

- After finishing data preparation, use the following command:

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
> Zhenrong Shen[1], Maosong Cao[2], Sheng Wang[1,3], Lichi Zhang[1], Qian Wang[2]*
> 
**Institution:**
> [1] School of Biomedical Engineering, Shanghai Jiao Tong University, Shanghai, China
> 
> [2] School of Biomedical Engineering, ShanghaiTech University, Shanghai, China
>
> [3] Shanghai United Imaging Intelligence Co., Ltd., Shanghai, China
> 
**Manuscript Link:**
> https://arxiv.org/abs/2307.06182 (preprint on arXiv)
>
> https://link.springer.com/chapter/10.1007/978-3-031-43987-2_47 (MICCAI 2023, conference version)
>
**Citation:**
```
@inproceedings{shen2023cellgan,
  title={CellGAN: Conditional Cervical Cell Synthesis for Augmenting Cytopathological Image Classification},
  author={Shen, Zhenrong and Cao, Maosong and Wang, Sheng and Zhang, Lichi and Wang, Qian},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={487--496},
  year={2023},
  organization={Springer}
}
```
