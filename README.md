# source-free-domain-adaptation
This is an source-free domain adaptation repository based on PyTorch. It was developed by [Wenxin Su](https://hazelsu.github.io/). If you encounter any issues or have questions, please don't hesitate to contact Wenxin at suwenxin43@gmail.com , baiyunxiang11@gmail.com or guokai063@gmail.com. 
<!-- It is also the official repository for the following works:
- [**ICLR(Oral)'25**][Proxy Denoising for Source-Free Domain Adaptation (ProDe)](https://arxiv.org/abs/2406.01658)
- [**Underveiw**][Source-Free Domain Adaptation with Task-Specific Multimodal Knowledge Distillation (TSD)] 
- [**ARXIV'24**][Unified Source-Free Domain Adaptation (CausalDA)](https://arxiv.org/abs/2403.07601)
- [**CVPR'24**][Source-Free Domain Adaptation with Frozen Multimodal Foundation Model (DIFO)](https://arxiv.org/abs/2311.16510v3)
- [**IJCV'23**][Source-Free Domain Adaptation via Target Prediction Distribution Searching (TPDS)](https://link.springer.com/article/10.1007/s11263-023-01892-w)
- [**NN'22**][Semantic consistency learning on manifold for source data-free unsupervised domain adaptation (SCLM)](https://www.sciencedirect.com/science/article/pii/S0893608022001897)
- [**IROS'21**][Model Adaptation through Hypothesis Transfer with Gradual Knowledge Distillation (GKD)](https://ieeexplore.ieee.org/abstract/document/9636206)
-->

This repository is also supports the following methods:
  - Source, [SHOT](http://proceedings.mlr.press/v119/liang20a/liang20a.pdf),
  [NRC](https://proceedings.neurips.cc/paper_files/paper/2021/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf), [COWA](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf), [AdaContrast](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf), [PLUE](https://openaccess.thecvf.com/content/CVPR2023/papers/Litrico_Guiding_Pseudo-Labels_With_Uncertainty_Estimation_for_Source-Free_Unsupervised_Domain_Adaptation_CVPR_2023_paper.pdf)
</details>

We encourage contributions! Pull requests to add methods are very welcome and appreciated.

<!-- 
## Our Publications
- [**ICLR(Oral)'25**][Proxy Denoising for Source-Free Domain Adaptation (ProDe)](https://arxiv.org/abs/2406.01658)  and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/ProDe.py).

- [**ARXIV'24**][Unified Source-Free Domain Adaptation (CausalDA)](https://arxiv.org/abs/2403.07601), and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/CausalDA.py)

- [**CVPR'24**][Source-Free Domain Adaptation with Frozen Multimodal Foundation Model](https://arxiv.org/abs/2311.16510v3), [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/difo.py), and [*Chinese version*](https://zhuanlan.zhihu.com/p/687080854)
- [**Underveiw**][Source-Free Domain Adaptation with Task-Specific Multimodal Knowledge Distillation], [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/tsd.py)
- [**IJCV'23**][Source-Free Domain Adaptation via Target Prediction Distribution Searching](https://link.springer.com/article/10.1007/s11263-023-01892-w) and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/tpds.py)

- [**TMM'23**][Progressive Source-Aware Transformer for Generalized Source-Free Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/10269002) and [*Code*](https://github.com/tntek/PSAT-GDA)

- [**CAAI TRIT'22**][Model adaptation via credible local context representation](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/cit2.12228) and [*Code*](https://github.com/tntek/CLCR)

- [**NN'22**][Semantic consistency learning on manifold for source data-free unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0893608022001897) and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/sclm.py)

- [**IROS'21**][Model Adaptation through Hypothesis Transfer with Gradual Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9636206) and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/gkd.py)
-->

## Preliminary

To use the repository, we provide a conda environment.
```bash
conda update conda
conda env create -f environment.yml
conda activate sfa
```
- **Datasets**
  - `office-31` [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view?resourcekey=0-gNMHVtZfRAyO_t2_WrOunA)
  - `office-home` [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view)
  - `VISDA-C` [VISDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)
  - `domainnet126` [DomainNet (cleaned)](http://ai.bu.edu/M3SDA/)
  - `imagenet_a` [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
  - `imagenet_r` [ImageNet-R](https://github.com/hendrycks/imagenet-r)
  - `imagenet_v2` [ImageNet-V2](https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main)
  - `imagenet_k` [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

You need to download the above dataset,modify the path of images in each '.txt' under the folder './data/'.In addition, class name files for each dataset also under the folder './data/'.The prepared directory would look like:
```bash
├── data
    ├── office-home
        ├── amazon_list.txt
        ├── classname.txt
        ├── dslr_list.txt
        ├── webcam_list.txt
    ├── office-home
        ├── Art_list.txt
        ├── classname.txt
        ├── Clipart_list.txt
        ├── Product_list.txt
        ├── RealWorld_list.txt
    ...  ...
```
For the ImageNet variations, modify the `${DATA_DIR}` in the `conf.py` to your data directory where stores the ImageNet variations datasets.

## Training
We provide config files for experiments. 
### Source
- For office-31, office-home and VISDA-C, there is an example to training a source model :
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/source.yaml" SETTING.S 0
```
- For domainnet126, we follow [AdaContrast](https://github.com/DianCh/AdaContrast) to train the source model.

- For adapting to ImageNet variations, all pre-trained models available in [Torchvision](https://pytorch.org/vision/0.14/models.html) or [timm](https://github.com/huggingface/pytorch-image-models/tree/v0.6.13) can be used.

- We also provide the pre-trained source models which can be downloaded from [here](https://drive.google.com/drive/folders/17n6goPXw_-ERgTK8R8nm4M_8PJPTEK1j?usp=sharing).

### Target
After obtaining the source models, modify the `${CKPT_DIR}` in the `conf.py` to your source model directory. For office-31, office-home and VISDA-C, simply run the following Python file with the corresponding config file to execute source-free domain adaptation.
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_of_oh_vs.py --cfg "cfgs/office-home/difo.yaml" SETTING.S 0 SETTING.T 1
```
For domainnet126 and ImageNet variations.
```bash
CUDA_VISIBLE_DEVICES=0 python image_target_in_126.py --cfg "cfgs/domainnet126/difo.yaml" SETTING.S 0 SETTING.T 1
```

## Acknowledgements
+ SHOT [official](https://github.com/tim-learn/SHOT)
+ NRC [official](https://github.com/Albert0147/NRC_SFDA)
+ COWA [official](https://github.com/Jhyun17/CoWA-JMDS)
+ AdaContrast [official](https://github.com/DianCh/AdaContrast)
+ PLUE [official](https://github.com/MattiaLitrico/Guiding-Pseudo-labels-with-Uncertainty-Estimation-for-Source-free-Unsupervised-Domain-Adaptation)
+ CoOp [official](https://github.com/KaiyangZhou/CoOp)
+ RMT [official](https://github.com/mariodoebler/test-time-adaptation)
