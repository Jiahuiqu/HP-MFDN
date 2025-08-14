# HP-MFDN
Multiscale Common-Private Feature Adversarial Decoupling Network for Hyperspectral Pansharpening
# Abstract
Hyperspectral pansharpening involves the fusion of a high-resolution panchromatic (PAN) image with a lower-resolution hyperspectral (LRHS) image, yielding a remarkable high-resolution hyperspectral (HRHS) image. Most of the existing pansharpening methods design specific feature extraction modules to dig deep spatial–spectral features. However, these methods ignore the commonalities and characteristics between PAN image and HS image, which may cause common information redundancy and private information loss. Here, we propose a multiscale common-private feature decoupling network based on adversarial learning (called HP-MFDN) for HS pansharpening, which refines and integrates the common-private features extracted from PAN and LRHS images of different scales losslessly, enhancing the pansharpening performance by fully utilizing the complementary information of PAN and HS images. Specifically, in each scale of PAN and HS images, the co-learning common-private feature decoupling module (CL-CPFDM) consisting of adversarial learning network and the specific decoupling losses is presented to decouple PAN and HS features into mutually orthogonal and independent common-private features. In addition, we specially design an information lossless refinement-based fusion module (ILRFM) for private information integration based on invertible neural network (INN), ensuring an effective spatial–spectral information flow for HRHS image reconstruction. Experimental results demonstrate that the proposed HP-MFADN outperforms other widely accepted state-of-the-art methods in both objective metrics and visual appearance. The code link is: https://github.com/Jiahuiqu/HP-MFDN.
# Training
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1234 Demo.py > /media/Private_Common/train.log 2>&1 &    
# Cite
If you find this code helpful, please kindly cite:

```
@article{HOU2025113031,
title = {Multiscale common-private feature adversarial decoupling network for hyperspectral pansharpening},
journal = {Knowledge-Based Systems},
volume = {310},
pages = {113031},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.113031},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125000796},
author = {Shaoxiong Hou and Song Xiao and Jiahui Qu and Wenqian Dong}
}

```
