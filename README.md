# LaVIDE: A Language-Vision Discriminator for Detecting Changes in Satellite Image with Map References

Authors: Shuguo Jiang, Fang Xu, Sen Jia, and Gui-Song Xia.

<!--[[paper](https://arxiv.org/abs/2212.03588)] [[github](https://github.com/ZiqinZhou66/ZegCLIP)] [[docker image](https://hub.docker.com/repository/docker/ziqinzhou/zegclip/general)] [[pretrained models](https://github.com/ZiqinZhou66/ZegCLIP/tree/main#pretrained-models)] [[visualization](https://github.com/ZiqinZhou66/ZegCLIP/blob/main/figs/vis.png)]  [[visualization of class queries](https://github.com/ZiqinZhou66/ZegCLIP/blob/main/figs/vis-query.png)]-->

---

> **Abstract:** *Change detection is a widely adopted technique in remote sensing for comparing multi-temporal data to capture surface dynamics, playing a critical role in accelerating map updating and supporting the monitoring of natural resources. Current studies in this field typically rely on the comparison of bi-temporal satellite images, which is significantly hindered when only a single image is available. In contrast, land cover maps are often readily available---through geographic information systems, cartographic archives, or crowdsourced annotations---offering a promising yet underexplored resource for single-image change detection. Unlike images that carry low-level visual details of ground objects, maps convey high-level categorical information.  This discrepancy in abstraction levels complicates the semantic alignment and comparison of the two data types. In this paper, we propose a Language-VIsion Discriminator for dEtecting changes in satellite images with map references, namely LaVIDE, which leverages language to bridge the information gap between maps and images.  LaVIDE formulates change detection as the problem of ``Does the pixel belong to [class]?'', aligning maps and images within the feature space of the language-vision model to associate high-level map categories with low-level image details. Specifically, we propose a restricted prompt learning approach, which incorporates image characteristics to automatically generate object prompts that are consistent with the semantic content of remote sensing images. Extensive evaluation on four benchmark datasets demonstrates that LaVIDE can effectively identify geospatial changes when only up-to-date satellite images are available, outperforming state-of-the-art change detection algorithms, e.g.,  with gains of about 18.4\% on the DynamicEarthNet dataset and 5.2\% on the HRSCD dataset.* 
>
> <p align="center">
> <img width="1000" src="figs/overall_of_LaVIDE.png">
> </p>

## Visualization of Change Detection Results
### On DynamicEarthNet
> <p align="center">
> <img width="1000" src="figs/dynamic_earth_net.png">
> </p>

### On HRSCD
> <p align="center">
> <img width="1000" src="figs/hrscd.png">
> </p>

### On BANDON
> <p align="center">
> <img width="1000" src="figs/bandon.png">
> </p>

### On SECOND
> <p align="center">
> <img width="1000" src="figs/second.png">
> </p>

<!--## Environment:

Option 1:

- Install pytorch

 `conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch`

- Install the mmsegmentation library and some required packages.

 `pip install mmcv-full==1.4.4 mmsegmentation==0.24.0`
 `pip install scipy timm==0.3.2`

Option 2:

- Directly apply the same Image we provieded in Dockerhub:

 `docker push ziqinzhou/zegclip:latest`

## Downloading and preprocessing Dataset:
According to MMseg: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md

## Preparing Pretrained CLIP model:
Download the pretrained model here: Path/to/ViT-B-16.pt
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt

## Pretrained models:

|     Dataset     |   Setting    |  pAcc | mIoU(S) | mIoU(U) | hIoU |                           Model Zoo                           |
| :-------------: | :---------:  | :---: | :-----: | :-----: | :--: |  :----------------------------------------------------------: |
| PASCAL VOC 2012 |  Inductive   |  94.6 |   91.9  |   77.8  | 84.3 | [[Google Drive](https://drive.google.com/file/d/1MqIFntWtTQk3HPiZp8hysI7CQIDTXteS/view?usp=share_link)] |
| PASCAL VOC 2012 | Transductive |  96.2 |   92.3  |   89.9  | 91.1 | [[Google Drive](https://drive.google.com/file/d/1PsXOd_A9Pww3ftclTDVWKcAvNIOAn6E-/view?usp=share_link)] |
| PASCAL VOC 2012 |    Fully     |  96.3 |   92.4  |   90.9  | 91.6 | [[Google Drive](https://drive.google.com/file/d/1oBj2FRV17TAKKrSexg4SUKa4I0egEwaj/view?usp=sharing)] |
| COCO Stuff 164K |  Inductive   |  62.0 |   40.2  |   41.1  | 40.8 | [[Google Drive](https://drive.google.com/file/d/12M6T97o9wyxbJKrR7zLfFMDsGVTiq4WY/view?usp=share_link)]|
| COCO Stuff 164K | Transductive |  69.2 |   40.7  |   59.9  | 48.5 | [[Google Drive](https://drive.google.com/file/d/1S8ia0-0oAUELxQXWqz0OlKoEv68gUuMp/view?usp=sharing)]|
| COCO Stuff 164K |    Fully     |  69.9 |   40.7  |   63.2  | 49.6 | [[Google Drive](https://drive.google.com/file/d/1DvUpYZa0rtPUBOjsYWwt-TEs6YdvUG0C/view?usp=share_link)] |

Note that here we report the averaged results of several training models and provide one of them.

## Efficiency results:
|     Dataset     |  #Params(M) |  Flops(G)  |     FPS    |
| :-------------: | :---------: | :--------: | :--------: |
| PASCAL VOC 2012 |    13.8     |    110.4   |     9.0    |
| COCO Stuff 164K |    14.6     |    123.9   |     6.7    |

Note that all experience are conducted on a single 1080Ti GPU and #Params(M) represents the number of learnable parameters.

## Training (Inductive):

 ```shell
 bash dist_train.sh configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100
 bash dist_train.sh configs/voc12/vpt_seg_zero_vit-b_512x512_20k_12_10.py Path/to/voc12/zero_12_10
 ```

## Training (Transductive):
 ```shell
 bash dist_train.sh ./configs/coco/vpt_seg_zero_vit-b_512x512_40k_12_100_multi_st.py Path/to/coco/zero_12_100_st --load-from=Path/to/coco/zero_12_100/iter_40000.pth
 bash dist_train.sh ./configs/voc12/vpt_seg_zero_vit-b_512x512_10k_12_10_st.py Path/to/voc12/zero_12_10_st --load-from=Path/to/voc12/zero_12_10/iter_10000.pth
 ```

## Training (Fully supervised):
 ```shell
 bash dist_train.sh configs/coco/vpt_seg_fully_vit-b_512x512_80k_12_100_multi.py Path/to/coco/fully_12_100
 bash dist_train.sh configs/voc12/vpt_seg_fully_vit-b_512x512_20k_12_10.py Path/to/voc12/fully_12_10
 ```

## Inference:
 `python test.py ./path/to/config ./path/to/model.pth --eval=mIoU`

For example: 
```shell
CUDA_VISIBLE_DEVICES="0" python test.py configs/coco/vpt_seg_zero_vit-b_512x512_80k_12_100_multi.py Path/to/coco/zero_12_100/latest.pth --eval=mIoU
```

## Cross Dataset Inference:
```shell
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-voc.py Path/to/coco/vpt_seg_zero_80k_12_100_multi/iter_80000.pth --eval=mIoU
CUDA_VISIBLE_DEVICES="0" python test.py ./configs/cross_dataset/coco-to-context.py Path/to/coco/vpt_seg_zero_80k_12_100_multi/iter_80000.pth --eval=mIoU
```

## Related Assets \& Acknowledgement

Our work is closely related to the following assets that inspire our implementation. We gratefully thank the authors. 

 - CLIP:  https://github.com/openai/CLIP
 - Maskformer: https://bowenc0221.github.io/maskformer
 - Zegformer: https://github.com/dingjiansw101/ZegFormer
 - zsseg: https://github.com/MendelXu/zsseg.baseline
 - MaskCLIP: https://github.com/chongzhou96/MaskCLIP
 - SegViT: https://github.com/zbwxp/SegVit
 - DenseCLIP: https://github.com/raoyongming/DenseCLIP/blob/master/segmentation/denseclip
 - Visual Prompt Tuning: https://github.com/KMnP/vpt
 
## Citation:
If you find this project useful, please consider citing:
```
@article{zhou2022zegclip,
  title={ZegCLIP: Towards adapting CLIP for zero-shot semantic segmentation},
  author={Zhou, Ziqin and Lei, Yinjie and Zhang, Bowen and Liu, Lingqiao and Liu, Yifan},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```-->
