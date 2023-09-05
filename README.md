# SegMiF

![visitors](https://visitor-badge.glitch.me/badge?page_id=JinyuanLiu-CV.SegMiF)
Jinyuan Liu, Zhu Liu, Guanyao Wu, Long Ma, Risheng Liu, Wei Zhong, Zhongxuan Luo,  Xin Fan*,**“Multi-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation”**, International Conference on Computer Vision **(ICCV)**, 2023. **(Oral)**

- [*[ArXiv]*](https://arxiv.org/abs/2308.02097)

---

<h2> <p align="center"> FMB Dataset </p> </h2>  

### Preview

The preview of our FMB dataset is as follows.

---

![preview](assets/overview.pdf)
 
---

### Download
- [*[Google Drive]*](https://drive.google.com/drive/folders/1T_jVi80tjgyHTQDpn-TjfySyW4CK1LlF?usp=sharing)

<h2> <p align="center"> SegMiF Fusion </p> </h2>  

## Set Up on Your Own Machine

When you want to dive deeper or apply it on a larger scale, you can configure our SegMiF on your computer by following the steps below.

#### Virtual Environment

We strongly recommend that you use Conda as a package manager.

```shell
# create virtual environment
conda create -n SegMiF python=3.10
conda activate SegMiF
# select pytorch version yourself
# install SegMiF requirements
pip install -r requirements.txt
```

#### Data Preparation

Related data, checkpoint, and our results on MFNet Dataset can be downloaded in 
- [*[Google Drive]*](https://drive.google.com/drive/folders/1MFTVd32-VNcpiFfNsu9Rw73YZJATPHA6?usp=sharing)

## Citation

If this work has been helpful to you, please feel free to cite our paper!

```
@inproceedings{liu2023segmif,
  title={ulti-interactive Feature Learning and a Full-time Multi-modality Benchmark for Image Fusion and Segmentation},
  author={Liu, Jinyuan and Liu, Zhu and Wu, Guanyao and Ma, Long and Liu, Risheng and Zhong, Wei and Luo, Zhongxuan and Fan, Xin},
  booktitle={International Conference on Computer Vision},
  year={2023}
}
```

