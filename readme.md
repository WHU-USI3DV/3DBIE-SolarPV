<h2> 
<a href="https://github.com/WHU-USI3DV/3DBIE-SolarPV/" target="_blank">City-scale solar PV potential estimation on 3D buildings using multi-source RS data: A case study in Wuhan, China</a>
</h2>

This is the PyTorch implementation of the following publication:

> **City-scale solar PV potential estimation on 3D buildings using multi-source RS data: A case study in Wuhan, China**<br/>
> [Zhe Chen](https://github.com/ChenZhe-Code), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm), [Rui Zhu](https://felix-rz.github.io/), [Zhen Dong](https://dongzhenwhu.github.io/index.html)<br/>
> [**Paper**](https://doi.org/10.1016/j.jag.2022.103107)  *Applied Energy Under Review*<br/>

The part on building extraction using unsupervised domain adapation utilized our another work:
> **Joint alignment of the distribution in input and feature space for cross-domain aerial image semantic segmentation**<br/>
> [Zhe Chen](https://github.com/ChenZhe-Code), [Bisheng Yang](https://3s.whu.edu.cn/info/1025/1415.htm), [Ailong Ma](http://jszy.whu.edu.cn/maailong/zh_CN/index.htm), Mingjun Peng, Haiting Li, Tao Chen, [Chi Chen](https://3s.whu.edu.cn/info/1025/1364.htm), [Zhen Dong](https://dongzhenwhu.github.io/index.html)<br/>
> [**Paper**](https://doi.org/10.1016/j.jag.2022.103107)  *JAG 2022*<br/>

## 🔭 Introduction
<p align="center">
<strong>City-scale solar PV potential estimation on 3D buildings using multi-source RS data:<br/> A case study in Wuhan, China</strong>
</p>
<div align=center>
<img src="teaser.png" alt="Network" style="zoom:40%" align='middle'>
</div>

<p align="justify">
<strong>Abstract:</strong>
Assessing the solar photovoltaic (PV) potential on buildings is essential for environmental protection and sustainable development. However, currently, the high costs of data acquisition and labor required to obtain 3D building models limit the scalability of such estimations extending to a large scale. To overcome the limitations, this study proposes a method of using freely available multi-source Remote Sensing (RS) data to estimate the solar PV potential on buildings at the city scale without any labeling. Firstly, Unsupervised Domain Adaptation (UDA) is introduced to transfer the building extraction knowledge learned by Deep Semantic Segmentation Networks (DSSN) from public datasets to available satellite images in a label-free manner. In addition, the coarse-grained land cover product is utilized to provide prior knowledge for reducing negative transfer. Secondly, the building heights are derived from the global open Digital Surface Model (DSM) using morphological operations. The building information obtained from the above two aspects supports the subsequent estimation. In the case study of Wuhan, China, the solar PV potential on all buildings throughout the city is estimated without any data acquisition cost or human labeling cost through the proposed method. In 2021, the estimated solar irradiation received by buildings in Wuhan is 289737.58 GWh. Taking into account the current technical conditions, the corresponding solar PV potential is 43460.64 GWh, which can meet the electricity demands of residents.
</p>

## 🆕 News
- 2023-12-15: Code is aviliable! 🎉


## 🚅 Cross-domain building footprint extraction
### 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 11.3
- Python 3.8.0
- Pytorch 1.12.1
- GeForce RTX 4090.

### 🔧 Installation
You can create an environment directy using the provided ```environment.yaml```
```
conda env create -f environment.yaml
conda activate fgdal
```

### 💾 Dataset 
Our method has been experimented in both the benchmark and practical applications.
>- **ISPRS 2D cross-domain semantic segmentation benchmark**  
&ensp;&ensp;&ensp;&ensp;Provided by [Te Shi](https://github.com/te-shi/MUCSS?tab=readme-ov-file), the ISRPS image dataset for cross-domain semantic segmentation can be downloaded via [Google Drive](https://drive.google.com/file/d/1amV--tjtjBMUscUVBqXxXws_vBCo-QdV/view) or [BaiduDisk](https://pan.baidu.com/share/init?surl=Ob12TozQ2Xjcm3rcv7LuRA) (Acess Code: vaam).
>- **Pratical applications: SpaceNet-Shanghai to GES-Wuhan**  
&ensp;&ensp;&ensp;&ensp; The SpaceNet-Shanghai dataset and GES image dataset ( Wuchang District, Wuhan ) can be downloaded via [Google Drive](https://drive.google.com/file/d/1amV--tjtjBMUscUVBqXxXws_vBCo-QdV/view) or [BaiduDist](https://pan.baidu.com/share/init?surl=Ob12TozQ2Xjcm3rcv7LuRA) (Acess Code: ****)

Once the datasets are downloaded and decompressed, change the folder path of the dataset according to the actual path in file *```UDA-Seg/core/datasets/dataset_path_catalog.py```* (line 33-36) for training and testing purposes.

### 🔦 Train
We provide the training script for source domain training and domain adaptation training. 
```
bash train_with_sd.sh
```
Specially, supervised training on labeled source domain data is needed to initialize the network parameters firstly.
```
# Set the num of GPUs, for example, 2 GPUs
export NGPUS=2
# train on source data
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_src.py -cfg configs/configs/rs_deeplabv2_r101_src.yaml OUTPUT_DIR results/src_r101_try/
```


Note that our framework does not use **self distill**. However, you can slightly modify the code (network) in *```train_self_distill.py```* to conduct self distill and further improve the performance.

```
bash train_with_sd.sh
```

### ✏️ Test

## 🚅 Pretrained model
FreeReg does not need any training but utilizes pretrained models of existing projects:

- Diffusion Feature related, download them and place them in ```tools/controlnet/models``` directory: 
  - Stable Diffusion v1.5 [[ckpt]](https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt);
  <!-- - ControlNet(SDv1.5) conditioning on depth images [[ckpt]](https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11f1p_sd15_depth.pth); -->
  - ControlNet(SDv1.5) conditioning on depth images [[ckpt-ft]](https://drive.google.com/file/d/1YSYXHZtg4Mvdh_twOK_FIc8kao3sA3z2/view?usp=drive_link); **[23/11/23 Update: We fine-tune ControlNet with real RGB-D data for better performances.]**

- Depth Estimation related, download them and place them in ```tools/zoe/models``` directory:
  - Zoe-depth-n [[ckpt]](https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt) for indoor datasets such as 3dmatch/scannet;
  - Zoe-depth-nk [[ckpt]](https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_NK.pt) for outdoor datasets such as kitti;

- Geometric Feature related, download them and place them in ```tools/fcgf/models``` directory:
  - FCGF-indoor [[ckpt]](https://drive.google.com/file/d/1cLFlKC_novdwFbxk6dtLlqOUlmynV7jc/view?usp=sharing);
  - FCGF-outdoor [[ckpt]](https://drive.google.com/file/d/1D6mKqzGqg9seeU3s2QJ7MEgHJPKxNo8F/view?usp=sharing)；
  - Or download FCGF models from [[Baidu-Disk]](https://pan.baidu.com/s/16-6osDbN8EaWRgT1dvUiQg)(code:35h9).


## 💾 Dataset 
The datasets are accessible in [[Baidu-Disk]](https://pan.baidu.com/s/16-6osDbN8EaWRgT1dvUiQg)(code:35h9) and Google Cloud:

- [[3DMatch]](https://drive.google.com/file/d/1tSTlYFou6UEKR_UJa0Qm0Dy6foW4ubIs/view?usp=sharing);
- [[ScanNet]](https://drive.google.com/file/d/1wSoPzuAIZ3DFU1Gk2wcREXQG3jd9PW7s/view?usp=sharing);
- [[Kitti-DC]](https://drive.google.com/file/d/1c1TcUV2fMmXKK_vyZstVLD9J4-pCVCRu/view?usp=sharing).

Please place the data to ```./data```.

## ✏️ Test
To eval FreeReg on three benchmarks, you can use the following commands:
```
python run.py --dataset 3dmatch --type dg
python run.py --dataset scannet --type dg
python run.py --dataset kitti --type dg
```
you can replace ```--type dg``` that uses fused features and Kabsch solver with ```--type d``` for only using diffusion features and pnp solver or ```--type g``` for only using geometric features and Kabsch solver.

## 💡 Citation
If you find this repo helpful, please give us a 😍 star 😍.
Please consider citing FreeReg if this program benefits your project
```
@article{wang2023freereg,
  title={FreeReg: Image-to-Point Cloud Registration Leveraging Pretrained Diffusion Models and Monocular Depth Estimators},
  author={Haiping Wang and Yuan Liu and Bing Wang and Yujing Sun and Zhen Dong and Wenping Wang and Bisheng Yang},
  journal={arXiv preprint arXiv:2310.03420},
  year={2023}
}
```

## 🔗 Related Projects
We sincerely thank the excellent projects:
- [ControlNet](https://github.com/lllyasviel/ControlNet) for Diffusion Feature extraction;
- [Zoe-Depth](https://github.com/isl-org/ZoeDepth) for metric depth estimation on rgb images;
- [FCGF](https://github.com/chrischoy/FCGF) for Geometric Feature extraction;
- [IDC-DC](https://github.com/kujason/ip_basic) for depth completion.
