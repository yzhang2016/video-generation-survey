# Video Generation Survey
A reading list of video generation

### [Related Text-to-image Generation and Super-resolution](https://github.com/yzhang2016/video-generation-survey/blob/main/Text-to-Image.MD)

## :point_right: Databases

* **HowTo100M**

  [ICCV 2019] Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips \[[PDF](https://arxiv.org/pdf/1906.03327.pdf), [Project](https://www.di.ens.fr/willow/research/howto100m/) \]
  
* **Web10M**

  [ICCV 2021]Frozen in Time: A Joint Video and Image Encoder for End-to-End Retrieval \[[PDF](https://arxiv.org/pdf/2104.00650.pdf), [Project](https://github.com/m-bain/webvid) \]
  
* **UCF-101**

  [arxiv 2012] Ucf101: A dataset of 101 human actions classes from videos in the wild \[[PDF](https://arxiv.org/pdf/1212.0402.pdf), [Project](https://www.crcv.ucf.edu/data/UCF101.php) \]
  
* **Sky Time-lapse** 

  [CVPR 2018] Learning to generate time-lapse videos using multi-stage dynamic generative adversarial networks \[[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xiong_Learning_to_Generate_CVPR_2018_paper.pdf), [Project](https://github.com/weixiong-ur/mdgan) \]
  
* **TaiChi** 

  [NIPS 2019] First order motion model for image animation \[ [PDF](https://papers.nips.cc/paper/2019/file/31c0b36aef265d9221af80872ceb62f9-Paper.pdf), [Project](https://github.com/AliaksandrSiarohin/first-order-model) \]

* **Celebv-text**
[arxiv ]CelebV-Text: A Large-Scale Facial Text-Video Dataset [[PDF](), [Page](https://celebv-text.github.io/)]



## :point_right: GAN/VAE-based methods 
[NIPS 2016] **---VGAN---** Generating Videos with Scene Dynamics \[[PDF](https://proceedings.neurips.cc/paper/2016/file/04025959b191f8f9de3f924f0940515f-Paper.pdf), [code](https://github.com/cvondrick/videogan) \]

[ICCV 2017] **---TGAN---** Temporal Generative Adversarial Nets with Singular Value Clipping \[[PDF](https://arxiv.org/pdf/1611.06624.pdf), [code](https://github.com/pfnet-research/tgan) \]

[CVPR 2018] **---MoCoGAN---** MoCoGAN: Decomposing Motion and Content for Video Generation \[[PDF](https://arxiv.org/pdf/1707.04993.pdf), [code](https://github.com/sergeytulyakov/mocogan) \]

[NIPS 2018] **---SVG---** Stochastic Video Generation with a Learned Prior \[[PDF](https://proceedings.mlr.press/v80/denton18a/denton18a.pdf), [code](https://github.com/edenton/svg) \]

[ECCV 2018] Probabilistic Video Generation using
Holistic Attribute Control \[[PDF](https://openaccess.thecvf.com/content_ECCV_2018/papers/Jiawei_He_Probabilistic_Video_Generation_ECCV_2018_paper.pdf), code\]

[CVPR 2019; CVL ETH] **---SWGAN---** Sliced Wasserstein Generative Models \[[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wu_Sliced_Wasserstein_Generative_Models_CVPR_2019_paper.pdf), [code](https://github.com/skolouri/swae) \]


[NIPS 2019; NVLabs] **---vid2vid---** Few-shot Video-to-Video Synthesis \[[PDF](https://nvlabs.github.io/few-shot-vid2vid/main.pdf), [code](https://github.com/NVlabs/few-shot-vid2vid) \]

[arxiv 2020; Deepmind] **---DVD-GAN---** ADVERSARIAL VIDEO GENERATION ON COMPLEX DATASETS \[[PDF](https://arxiv.org/pdf/1907.06571.pdf), [code](https://github.com/Harrypotterrrr/DVD-GAN) \]

[IJCV 2020] **---TGANv2---** Train Sparsely, Generate Densely: Memory-efficient Unsupervised Training of High-resolution Temporal GAN \[[PDF](https://arxiv.org/pdf/1811.09245.pdf), [code](https://github.com/pfnet-research/tgan2) \]

[PMLR 2021] **---TGANv2-ODE---** Latent Neural Differential Equations for Video Generation \[[PDF](https://arxiv.org/pdf/2011.03864.pdf), [code](https://github.com/Zasder3/Latent-Neural-Differential-Equations-for-Video-Generation) \]

[ICLR 2021 ] **---DVG---** Diverse Video Generation using a Gaussian Process Trigger \[[PDF](https://openreview.net/pdf?id=Qm7R_SdqTpT), [code](https://github.com/shgaurav1/DVG) \]

[Arxiv 2021; MRSA] **---GODIVA---** GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions \[[PDF]([https://arxiv.org/pdf/2205.15868.pdf](https://arxiv.org/pdf/2104.14806.pdf)), [code](https://github.com/sihyun-yu/digan) \]

[CVPR 2022 ] **---StyleGAN-V--** StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2 \[[PDF](https://arxiv.org/pdf/2112.14683.pdf), [code](https://github.com/universome/stylegan-v) \]


## :point_right: Implicit Neural Representations
[ICLR 2022] Generating videos with dynamics-aware implicit generative adversarial networks \[[PDF](https://openreview.net/pdf?id=Czsdv-S4-w9), [code]() \]

## :point_right: Transformer-based 
[arxiv 2021] **---VideoGPT--** VideoGPT: Video Generation using VQ-VAE and Transformers \[[PDF](https://arxiv.org/pdf/2104.10157.pdf), [code](https://github.com/wilson1yan/VideoGPT) \]

[ECCV 2022; Microsoft] **---NÜWA--** NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion \[[PDF](https://arxiv.org/pdf/2111.12417.pdf), code \]

[NIPS 2022; Microsoft] **---NÜWA-Infinity--** NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis \[[PDF](https://arxiv.org/pdf/2207.09814.pdf), code \]

[Arxiv 2020; Tsinghua] **---CogVideo--** CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers \[[PDF](https://arxiv.org/pdf/2205.15868.pdf), [code](https://github.com/THUDM/CogVideo) \]

*[ECCV 2022] **---TATS--** Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer \[[PDF](https://arxiv.org/pdf/2204.03638.pdf), [code](https://github.com/SongweiGe/TATS)\]


*[NeurIPs 2022] **---MCVD---** MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation [[PDF](https://arxiv.org/abs/2205.09853), [code](https://github.com/voletiv/mcvd-pytorch)]

*[arxiv 2022; Google] **---PHENAKI--** PHENAKI: VARIABLE LENGTH VIDEO GENERATION FROM OPEN DOMAIN TEXTUAL DESCRIPTIONS \[[PDF](https://arxiv.org/pdf/2210.02399.pdf), code \]


[arxiv 2022.12]MAGVIT: Masked Generative Video Transformer[[PDF](https://arxiv.org/pdf/2212.05199.pdf)]

## :point_right: Diffusion-based methods 
*[NIPS 2022; Google] **---VDM--**  Video Diffusion Models \[[PDF](https://arxiv.org/pdf/2204.03458.pdf), [code](https://github.com/lucidrains/video-diffusion-pytorch) \]

*[arxiv 2022; Meta] **---MAKE-A-VIDEO--** MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA \[[PDF](https://arxiv.org/pdf/2209.14792.pdf), code \]

*[arxiv 2022; Google] **---IMAGEN VIDEO--** IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS \[[PDF](https://arxiv.org/pdf/2210.02303.pdf), code \]

*[arxiv 2022; ByteDace] ***MAGIC VIDEO***:Efficient Video Generation With Latent Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.11018.pdf), code\]

*[arxiv 2022; Tencent] Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths  \[[PDF](https://arxiv.org/pdf/2211.13221.pdf), code\]

[AAAI 2022; JHU ] VIDM: Video Implicit Diffusion Model \[[PDF](https://kfmei.page/vidm/Video_implicit_diffusion_models.pdf)\]

[arxiv 2023.01; Meta] Text-To-4D Dynamic Scene Generation [[PDF](https://arxiv.org/pdf/2301.11280.pdf), [Page](https://make-a-video3d.github.io/)]

[arxiv 2023.03]Video Probabilistic Diffusion Models in Projected Latent Space [[PDF](https://arxiv.org/abs/2302.07685), [Page](https://sihyun.me/PVDM/)]

[arxiv 2023.03]Controllable Video Generation by Learning the Underlying Dynamical System with Neural ODE [[PDF](https://arxiv.org/abs/2303.05323)]


## :point_right: Video Completion (animation, interpolation, prediction)
[arxiv 2022; Meta] Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation \[[PDF](https://arxiv.org/pdf/2211.12824.pdf), code]

## CV Related 
[arxiv 2022.12; ByteDace]PV3D: A 3D GENERATIVE MODEL FOR PORTRAIT VIDEO GENERATION [[PDF](https://arxiv.org/pdf/2212.06384.pdf)]

[arxiv 2022.12]MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation[[PDF](https://arxiv.org/pdf/2212.09478.pdf)]

[arxiv 2022.12]Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation [[PDF](https://arxiv.org/pdf/2212.11565.pdf), [Page](https://tuneavideo.github.io/)]

[arxiv 2023.01]Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation [[PDF](https://arxiv.org/pdf/2301.03396.pdf), [Page](https://mstypulkowski.github.io/diffusedheads)]

[arxiv 2023.01]DiffTalk: Crafting Diffusion Models for Generalized Talking Head Synthesis [[PDF](https://arxiv.org/pdf/2301.03786.pdf), [Page](https://sstzal.github.io/DiffTalk/)]

[arxiv 2023.02 Google]Scaling Vision Transformers to 22 Billion Parameters [[PDF](https://arxiv.org/abs/2302.05442)]

## NLP related
[arxiv 2022.10]DIFFUSEQ: SEQUENCE TO SEQUENCE TEXT GENERATION WITH DIFFUSION MODELS [[PDF](https://arxiv.org/pdf/2210.08933.pdf)]

[arxiv 2023.02]The Flan Collection: Designing Data and Methods for Effective Instruction Tuning [[PDF](https://arxiv.org/pdf/2301.13688.pdf)]


## Speech 
[arxiv 2023.01]Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers[[PDF](https://arxiv.org/abs/2301.02111), [Page](https://valle-demo.github.io/)]

