# source-free-domain-adaptation
This is a source-free domain adaptation repository based on PyTorch. It is also the official repository for the following works:
- [Source-Free Domain Adaptation with Frozen Multimodal Foundation Model](https://arxiv.org/pdf/2311.16510.pdf) (CVPR2024)
- [Source-Free Domain Adaptation via Target Prediction Distribution Searching](https://link.springer.com/article/10.1007/s11263-023-01892-w) (IJCV2023)
- [Model Adaptation through Hypothesis Transfer with Gradual Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9636206)(IROS2021)

</details>

We encourage contributions! Pull requests to add methods are very welcome and appreciated.

## Our Publications
- [**CVPR'24**] [Source-Free Domain Adaptation with Frozen Multimodal Foundation Model](https://arxiv.org/pdf/2402.19122.pdf), and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/difo.py)

- [**IJCV'23**][Source-Free Domain Adaptation via Target Prediction Distribution Searching](https://link.springer.com/article/10.1007/s11263-023-01892-w) and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/tpds.py)

- [**TMM'23**][Progressive Source-Aware Transformer for Generalized Source-Free Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/10269002) and [*Code*](https://github.com/tntek/PSAT-GDA)

- [**CAAI TRIT'22**][Model adaptation via credible local context representation](https://ietresearch.onlinelibrary.wiley.com/doi/pdf/10.1049/cit2.12228) and [*Code*](https://github.com/tntek/CLCR)

- [**NN'22**][Semantic consistency learning on manifold for source data-free unsupervised domain adaptation](https://www.sciencedirect.com/science/article/pii/S0893608022001897) and [*Code*](https://github.com/tntek/SCLM)

- [**IROS'21**][Model Adaptation through Hypothesis Transfer with Gradual Knowledge Distillation](https://ieeexplore.ieee.org/abstract/document/9636206) and [*Code*](https://github.com/tntek/source-free-domain-adaptation/blob/main/src/methods/oh/gkd.py)

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
    ├── VISDA-C
        ├── classname.txt
        ├── train_list.txt
        ├── validation_list.txt
    ...  ...
```
For the ImageNet variations, modify the `DATA_DIR` in the `conf.py` to your data directory where stores the ImageNet variations datasets.

- **Methods**
  - The repository currently supports the following methods: source, [SHOT](http://proceedings.mlr.press/v119/liang20a/liang20a.pdf),
  [NRC](https://proceedings.neurips.cc/paper_files/paper/2021/file/f5deaeeae1538fb6c45901d524ee2f98-Paper.pdf), [GKD](https://ieeexplore.ieee.org/abstract/document/9636206), [COWA](https://proceedings.mlr.press/v162/lee22c/lee22c.pdf), [TPDS](https://link.springer.com/article/10.1007/s11263-023-01892-w), [DIFO](https://arxiv.org/abs/2311.16510)
