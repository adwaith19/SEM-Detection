# ***SEM-AI-Image-Detection***



This repository contains the code and supporting files used to generate, evaluate, and detect AI-generated scanning electron microscopy (SEM) images using FastGAN, StyleGAN2-ADA, and a fine-tuned CLIP Vision Transformer (ViT-L/14).



The study evaluates synthetic SEM images generated from two SEM domains:



Nanoparticle SEM images

Biological SEM images



The repository contains the workflows for raw SEM image preprocessing (sample), CLIP-based image classification, quantitative image comparison, cross-domain evaluation, and EigenCAM visualization.



### **Overview**



The workflow used in this study consists of four main stages:



Synthetic SEM image generation using FastGAN and StyleGAN2-ADA.

Quantitative evaluation of generated images using MSE, LPIPS, and FID.

Classification of authentic and synthetic SEM images using a fine-tuned CLIP ViT-L/14 model.

Visual interpretation of the classifier using EigenCAM.



The classifier was trained separately for the nanoparticle and biological SEM datasets.



### **Dataset Sources**



The SEM images used in this study were obtained from previously published and publicly available datasets.



1. ##### Nanoparticle SEM dataset



Boiko, D. A., Pentsak, E. O., Cherepanova, V. A. \& Ananikov, V. P. Electron microscopy dataset for the recognition of nanoscale ordering effects and location of nanoparticles. Scientific Data 7, 101 (2020).



The original dataset should be obtained from the repository provided by the authors:



[https://doi.org/10.1038/s41597-020-0439-1](https://doi.org/10.1038/s41597-020-0439-1)



##### 2\. Biological SEM dataset



Aversa, R., Modarres, M. H., Cozzini, S., Ciancio, R. \& Chiusole, A. The first annotated set of scanning electron microscopy images for nanoscience. Scientific Data 5, 180172 (2018).



[https://doi.org/10.1038/sdata.2018.172](https://doi.org/10.1038/sdata.2018.172)



Important: The original datasets are not redistributed in this repository unless permitted by their respective licenses. Users should obtain the original datasets from the sources above.



### **Data Preparation**



Images were converted to RGB format and resized to 256 × 256 pixels for dataset preparation and GAN training.



For classifier training, the datasets were divided into:



70% training : 30% testing



The test set was kept separate from model training and was used only for final evaluation.



For each SEM domain, the dataset was organized into separate FastGAN and StyleGAN2-ADA branches containing authentic and generated images:

dataset/

├── train/

│   ├── FastGAN/

│   │   ├── 0\_real/

│   │   └── 1\_fake/

│   └── StyleGAN\_ADA/

│       ├── 0\_real/

│       └── 1\_fake/

│

└── test/

&#x20;   ├── FastGAN/

&#x20;   │   ├── 0\_real/

&#x20;   │   └── 1\_fake/

&#x20;   └── StyleGAN\_ADA/

&#x20;       ├── 0\_real/

&#x20;       └── 1\_fake/

The pre-processed dataset used for classifier training and testing can be accessed at <>



### **FastGAN Image Generation**



The PyTorch implementation of FastGAN was used to generate synthetic SEM images: [https://github.com/bingchenlll/FastGAN-pytorch](https://github.com/bingchenlll/FastGAN-pytorch)



Training configuration used in this study:



Base generator/discriminator channels: 64

Latent vector dimension: 256

Batch size: 16

Training iterations: 100,000

Random horizontal flipping: enabled

Training: from scratch

Image resolution: 256 × 256



All other parameters were retained from the corresponding implementation unless otherwise specified.



Reference:



Liu, B., Zhu, Y., Song, K. \& Elgammal, A. Towards faster and stabilized GAN training for high-fidelity few-shot image synthesis. ICLR (2021).



### **StyleGAN2-ADA Image Generation**



The PyTorch implementation of StyleGAN2-ADA was used to generate synthetic SEM images: [https://github.com/nvlabs/stylegan2-ada-pytorch](https://github.com/nvlabs/stylegan2-ada-pytorch)



Training configuration used in this study:



Batch size: 16

Training ticks: 1,000

Image mirroring: enabled

Adaptive discriminator augmentation: enabled according to the original implementation

Training: from scratch

Image resolution: 256 × 256



All other parameters were retained from the original implementation unless otherwise specified.



Reference:



Karras, T. et al. Training generative adversarial networks with limited data. NeurIPS 33 (2020).



### **CLIP-Based SEM Image Classifier**

Model:



The classifier uses the CLIP ViT-L/14 image encoder with the official OpenAI pretrained weights through OpenCLIP.



Reference:



Radford, A. et al. Learning transferable visual models from natural language supervision. ICML 139, 8748–8763 (2021).



OpenCLIP:



Ilharco, G. et al. OpenCLIP. Zenodo (2021).

**Cross-Domain Evaluation**
---



Once the model pt. file for one dataset is ready, predict\_folder.py can be used to perform the cross validation on the test set images (real or fake) of the other dataset.



EigenCAM Visualization



EigenCAM was used to visualize spatial regions associated with strong activation patterns in the learned feature representations of the CLIP image encoder.



The final transformer residual block was used as the target layer.



Because CLIP uses a Vision Transformer, the 257-token output contains:



1 CLS token

256 image-patch tokens



The 256 image tokens were reshaped into a 16 × 16 spatial grid for visualization.



The resulting activation maps were overlaid on the corresponding RGB input images.



Important: EigenCAM visualizations are used here as qualitative representations of activation patterns. They do not directly provide class-specific attribution for the real/fake prediction.



Reference:



Muhammad, M. B. \& Yeasin, M. Eigen-CAM: class activation map using principal components (2020).



CAM implementation:



Gildenblat, J. \& contributors. PyTorch library for CAM methods (2021).



[https://github.com/shyhyawJou/EigenCAM-Pytorch](https://github.com/shyhyawJou/EigenCAM-Pytorch)



### **Citation**



If you use this repository or the associated datasets/code, please cite the corresponding manuscript:



<link to manuscript>



Once the manuscript is published, this section should be updated with the final journal citation and DOI.



Please also cite the original SEM datasets and software frameworks used in the study.



### **License**



Code developed for this study is released under the MIT License. Third-party software and datasets remain subject to their respective licenses and terms of use.



