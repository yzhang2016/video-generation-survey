# Video Generation Survey
A reading list of video generation

### [Related Text-to-image Generation and Super-resolution](https://github.com/yzhang2016/video-generation-survey/blob/main/Text-to-Image.MD)

## :point_right: Models to play with

### Open source

* **VideoCrafter/Floor33** [[Page](http://floor33.tech/)], [[Discord](https://discord.gg/rrayYqZ4tf)], [[Code & Models](https://github.com/AILab-CVC/VideoCrafter)]

* **ModelScope** [[Page](https://modelscope.cn/models/damo/text-to-video-synthesis/summary), [i2v](https://modelscope.cn/models/damo/Image-to-Video/summary)], [[Code & Models](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)]

* **Hotshot-XL** [[Page](https://hotshot.co/)], [[Code & Models](https://github.com/hotshotco/Hotshot-XL)]

* **AnimeDiff** [[Page](https://animatediff.github.io/), [Code & Models](https://github.com/guoyww/AnimateDiff)]

* **Zeroscope V2 XL** [[Page](https://huggingface.co/cerspense/zeroscope_v2_XL)]

### Non-open source

* **Gen-1/Gen-2** [[Page](https://research.runwayml.com/gen2)]

* **Pika Lab** [[Page](https://www.pika.art/)], [[Discord](http://discord.gg/pika)]

* **Moonvalley** [[Page](https://moonvalley.ai/)], [[Discord](https://discord.gg/vk3aaH7r)]

* **Leonard Ai** [[Page](https://leonardo.ai/)]

* **Morph Studio** [[Page](https://www.morphstudio.xyz/)], [[Discord](https://discord.gg/hjd9JvXTU5)]
  
* **Lensgo** [[Page](https://lensgo.ai/), [Discord]()]

* **Genmo** [[Page](https://www.genmo.ai/)]

* **PlaiDay** [[Discord](https://discord.gg/6f6Q9pWb)]

* **Nerverends** [[Page](https://neverends.life/create)]

* **HiDream.ai/Pixeling** [[Page](https://hidream.ai/#/Pixeling)]

* **Assistant++** [[Page](https://assistive.chat/video)]

## :point_right: Databases

* **HowTo100M**

  [ICCV 2019] Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips \[[PDF](https://arxiv.org/pdf/1906.03327.pdf), [Project](https://www.di.ens.fr/willow/research/howto100m/) \]

* **HD-VILA-100M**
  
  [CVPR 2022]Advancing High-Resolution Video-Language Representation with Large-Scale Video Transcriptions [[PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Xue_Advancing_High-Resolution_Video-Language_Representation_With_Large-Scale_Video_Transcriptions_CVPR_2022_paper.pdf), [Page](https://github.com/microsoft/XPretrain/blob/main/hd-vila-100m/README.md)]
  
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

* **Youku-mPLUG**
  
  [arxiv 2023.06]Youku-mPLUG: A 10 Million Large-scale Chinese Video-Language Dataset for Pre-training and Benchmarks [[PDF](https://arxiv.org/abs/2306.04362)]

* **InternVid**
  
  [arxiv 2023.07]InternVid: A Large-scale Video-Text Dataset for Multimodal Understanding and Generation [[PDF](https://arxiv.org/abs/2307.06942)]

* **DNA-Rendering**
  
  [arxiv 2023.07] DNA-Rendering: A Diverse Neural Actor Repository for High-Fidelity Human-centric Rendering [[PDF](https://arxiv.org/abs/2307.10173)]

* **Vimeo25M** (not open-source)
  
  [arxiv 2023.09] LAVIE: HIGH-QUALITY VIDEO GENERATION WITH CASCADED LATENT DIFFUSION MODELS [[PDF](https://arxiv.org/pdf/2309.15103.pdf)]

* **HD-VG-130M**

  [arxiv 2023.06]VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation [[PDF](https://arxiv.org/abs/2305.10874), [Page](https://github.com/daooshee/HD-VG-130M)]





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

*[CVPR 2022 ] **---StyleGAN-V--** StyleGAN-V: A Continuous Video Generator with the Price, Image Quality and Perks of StyleGAN2 \[[PDF](https://arxiv.org/pdf/2112.14683.pdf), [code](https://github.com/universome/stylegan-v) \]

*[NeurIPs 2022] **---MCVD---** MCVD: Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation [[PDF](https://arxiv.org/abs/2205.09853), [code](https://github.com/voletiv/mcvd-pytorch)]

## :point_right: Implicit Neural Representations
[ICLR 2022] Generating videos with dynamics-aware implicit generative adversarial networks \[[PDF](https://openreview.net/pdf?id=Czsdv-S4-w9), [code]() \]

## :point_right: Transformer-based 
[arxiv 2021] **---VideoGPT--** VideoGPT: Video Generation using VQ-VAE and Transformers \[[PDF](https://arxiv.org/pdf/2104.10157.pdf), [code](https://github.com/wilson1yan/VideoGPT) \]

[ECCV 2022; Microsoft] **---NÜWA--** NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion \[[PDF](https://arxiv.org/pdf/2111.12417.pdf), code \]

[NIPS 2022; Microsoft] **---NÜWA-Infinity--** NUWA-Infinity: Autoregressive over Autoregressive Generation for Infinite Visual Synthesis \[[PDF](https://arxiv.org/pdf/2207.09814.pdf), code \]

[Arxiv 2020; Tsinghua] **---CogVideo--** CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers \[[PDF](https://arxiv.org/pdf/2205.15868.pdf), [code](https://github.com/THUDM/CogVideo) \]

*[ECCV 2022] **---TATS--** Long Video Generation with Time-Agnostic VQGAN and Time-Sensitive Transformer \[[PDF](https://arxiv.org/pdf/2204.03638.pdf), [code](https://github.com/SongweiGe/TATS)\]


*[arxiv 2022; Google] **---PHENAKI--** PHENAKI: VARIABLE LENGTH VIDEO GENERATION FROM OPEN DOMAIN TEXTUAL DESCRIPTIONS \[[PDF](https://arxiv.org/pdf/2210.02399.pdf), code \]

[arxiv 2022.12]MAGVIT: Masked Generative Video Transformer[[PDF](https://arxiv.org/pdf/2212.05199.pdf)]

[arxiv 2023.11]Optimal Noise pursuit for Augmenting Text-to-Video Generation [[PDF](https://arxiv.org/abs/2311.00949)]



## :point_right: Diffusion-based methods 
*[NIPS 2022; Google] **---VDM--**  Video Diffusion Models \[[PDF](https://arxiv.org/pdf/2204.03458.pdf), [code](https://github.com/lucidrains/video-diffusion-pytorch) \]

*[arxiv 2022; Meta] **---MAKE-A-VIDEO--** MAKE-A-VIDEO: TEXT-TO-VIDEO GENERATION WITHOUT TEXT-VIDEO DATA \[[PDF](https://arxiv.org/pdf/2209.14792.pdf), code \]

*[arxiv 2022; Google] **---IMAGEN VIDEO--** IMAGEN VIDEO: HIGH DEFINITION VIDEO GENERATION WITH DIFFUSION MODELS \[[PDF](https://arxiv.org/pdf/2210.02303.pdf), code \]

*[arxiv 2022; ByteDace] ***MAGIC VIDEO***:Efficient Video Generation With Latent Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.11018.pdf), code\]

*[arxiv 2022; Tencent] ***LVDM*** Latent Video Diffusion Models for High-Fidelity Video Generation with Arbitrary Lengths  \[[PDF](https://arxiv.org/pdf/2211.13221.pdf), code\]

[AAAI 2022; JHU ] VIDM: Video Implicit Diffusion Model \[[PDF](https://kfmei.page/vidm/Video_implicit_diffusion_models.pdf)\]

[arxiv 2023.01; Meta] Text-To-4D Dynamic Scene Generation [[PDF](https://arxiv.org/pdf/2301.11280.pdf), [Page](https://make-a-video3d.github.io/)]

[arxiv 2023.03]Video Probabilistic Diffusion Models in Projected Latent Space [[PDF](https://arxiv.org/abs/2302.07685), [Page](https://sihyun.me/PVDM/)]

[arxiv 2023.03]Controllable Video Generation by Learning the Underlying Dynamical System with Neural ODE [[PDF](https://arxiv.org/abs/2303.05323)]

[arxiv 2023.03]Decomposed Diffusion Models for High-Quality Video Generation [[PDF](https://arxiv.org/pdf/2303.08320.pdf)]

[arxiv 2023.03]NUWA-XL: Diffusion over Diffusion for eXtremely Long Video Generation [[PDF](https://arxiv.org/abs/2303.12346)]

*[arxiv 2023.04]Latent-Shift: Latent Diffusion with Temporal Shift for Efficient Text-to-Video Generation [[PDF](https://arxiv.org/abs/2304.08477)]

*[arxiv 2023.04]Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models [[PDF](https://arxiv.org/abs/2304.08818), [Page](https://research.nvidia.com/labs/toronto-ai/VideoLDM/)]

[arxiv 2023.04]LaMD: Latent Motion Diffusion for Video Generation [[PDF](https://arxiv.org/abs/2304.11603)]

*[arxiv 2023.05]Preserve Your Own Correlation: A Noise Prior for Video Diffusion Models[[PDF](https://arxiv.org/pdf/2305.10474.pdf), [Page](https://research.nvidia.com/labs/dir/pyoco/)]

[arxiv 2023.05]VideoFactory: Swap Attention in Spatiotemporal Diffusions for Text-to-Video Generation [[PDF](https://arxiv.org/pdf/2305.10874.pdf)]

[arxiv 2023.08]ModelScope Text-to-Video Technical Report [[PDF](https://arxiv.org/pdf/2308.06571.pdf)]

[arxiv 2023.08]Dual-Stream Diffusion Net for Text-to-Video Generation [[PDF](https://huggingface.co/papers/2308.08316)]

[arxiv 2023.08]SimDA: Simple Diffusion Adapter for Efficient Video Generation [[PDF](https://arxiv.org/abs/2308.09710), [Page](https://chenhsing.github.io/SimDA/)]

[arxiv 2023.08]Dysen-VDM: Empowering Dynamics-aware Text-to-Video Diffusion with Large Language Models [[PDF](https://arxiv.org/pdf/2308.13812.pdf), [Page](https://haofei.vip/Dysen-VDM/)]

[arxiv 2023.09]Reuse and Diffuse: Iterative Denoising for Text-to-Video Generation[[PDF](https://arxiv.org/pdf/2309.03549.pdf),[Page](https://anonymous0x233.github.io/ReuseAndDiffuse/)]

[arxiv 2023.09]LAVIE: HIGH-QUALITY VIDEO GENERATION WITH CASCADED LATENT DIFFUSION MODELS [[PDF](https://arxiv.org/pdf/2309.15103.pdf), [Page](https://vchitect.github.io/LaVie-project/)]

[arxiv 2023.09]VideoDirectorGPT: Consistent Multi-scene Video Generation via LLM-Guided Planning [[PDF](https://arxiv.org/abs/2309.15091), [Page](https://videodirectorgpt.github.io/)]

[arxiv 2023.10]Show-1: Marrying Pixel and Latent Diffusion Models for Text-to-Video Generation [[PDF](https://arxiv.org/abs/2309.15818), [Page](https://showlab.github.io/Show-1)]

[arxiv 2023.10]LLM-grounded Video Diffusion Models [[PDF](https://arxiv.org/abs/2309.17444),[Page](https://llm-grounded-video-diffusion.github.io/)]

[arxiv 2023.10]VideoCrafter1: Open Diffusion Models for High-Quality Video Generation [[PDF](https://arxiv.org/abs/2310.19512),[Page](https://github.com/AILab-CVC/VideoCrafter)]

[arxiv 2023.11]Make Pixels Dance: High-Dynamic Video Generation [[PDF](https://arxiv.org/abs/2311.10982), [Page](https://makepixelsdance.github.io/)]

[arxiv 2023.11]Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets[[PDF](https://static1.squarespace.com/static/6213c340453c3f502425776e/t/655ce779b9d47d342a93c890/1700587395994/stable_video_diffusion.pdf), [Page](https://t.co/P2lmq343cf)]

[arxiv 2023.11]Kandinsky Video [[PDF](https://arxiv.org/abs/2311.13073),[Page](https://ai-forever.github.io/kandinsky-video/)]

[arxiv 2023.11] Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning [[PDF](https://arxiv.org/abs/2311.10709),[Page](https://emu-video.metademolab.com/)]

[arxiv 2023.12]GenDeF: Learning Generative Deformation Field for Video Generation [[PDF](https://arxiv.org/abs/2312.04561),[Page](https://aim-uofa.github.io/GenDeF/)]

[arxiv 2023.12]GenTron: Delving Deep into Diffusion Transformers for Image and Video Generation [[PDF](https://arxiv.org/abs/2312.04557),[Page](https://www.shoufachen.com/gentron_website/)]

[arxiv 2023.12]Hierarchical Spatio-temporal Decoupling for Text-to-Video Generation [[PDF](https://arxiv.org/abs/2312.04483), [Page](https://higen-t2v.github.io/)]

[arxiv 2023.12]AnimateZero:Video Diffusion Models are Zero-Shot Image Animators [[PDF](https://arxiv.org/abs/2312.03793),[Page](https://vvictoryuki.github.io/animatezero.github.io/)]

[arxiv 2023.12]Photorealistic Video Generation with Diffusion Models [[PDF](https://arxiv.org/abs/2312.06662),[Page](https://walt-video-diffusion.github.io/)]

[arxiv 2023.12]A Recipe for Scaling up Text-to-Video Generation with Text-free Videos [[PDF](A Recipe for Scaling up Text-to-Video Generation with Text-free Videos),[Page](https://tf-t2v.github.io/)]

[arxiv 2023.12]MagicVideo-V2: Multi-Stage High-Aesthetic Video Generation [[PDF](https://arxiv.org/pdf/2401.04468.pdf), [Page](https://magicvideov2.github.io/)]

[arxiv 2024.1]Latte: Latent Diffusion Transformer for Video Generation [[PDF](https://arxiv.org/abs/2401.03048),[Page](https://maxin-cn.github.io/latte_project)]



## LLMs-based 
[arxiv 2023.12]VideoPoet: A Large Language Model for Zero-Shot Video Generation [[PDF](https://arxiv.org/abs/2312.14125),[Page](http://sites.research.google/videopoet/)]



## improve Video Diffusion models 
[arxiv 2023.10]ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models [[PDF](https://arxiv.org/abs/2310.07702), [Page](https://yingqinghe.github.io/scalecrafter/)]

[arxiv 2023.10]FreeU: Free Lunch in Diffusion U-Net [[PDF](https://arxiv.org/pdf/2309.11497.pdf), [Page](https://chenyangsi.top/FreeU/)]

[arxiv 2023.12]FreeInit: Bridging Initialization Gap in Video Diffusion Models [[PDF](https://arxiv.org/abs/2312.07537),[Page](https://tianxingwu.github.io/pages/FreeInit/)]




## multi-prompt  & long video generation
[arxiv 2023.12]MTVG : Multi-text Video Generation with Text-to-Video Models [[PDF](https://arxiv.org/abs/2312.04086)]


## long video generation 
[arxiv 2023.]Gen-L-Video: Long Video Generation via Temporal Co-Denoising [[PDF](https://arxiv.org/abs/2305.18264), [Page](https://g-u-n.github.io/projects/gen-long-video/index.html)]

[arxiv 2023.10]FreeNoise: Tuning-Free Longer Video Diffusion Via Noise Rescheduling [[PDF](https://arxiv.org/abs/2310.15169),[Page](http://haonanqiu.com/projects/FreeNoise.html)]

[arxiv 2023.12]VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models[[PDF](https://arxiv.org/abs/2311.18837),[Page](https://chenhsing.github.io/VIDiff)]

[arxiv 2023.12]AVID: Any-Length Video Inpainting with Diffusion Model [[PDF](https://arxiv.org/abs/2312.03816),[Page](https://zhang-zx.github.io/AVID/)]

[arxiv 2023.12]RealCraft: Attention Control as A Solution for Zero-shot Long Video Editing [[PDF](https://arxiv.org/abs/2312.12635)]


## Higher Resolution 
[arxiv 2023.10] ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models [[PDF](https://arxiv.org/abs/2310.07702), [Page](https://yingqinghe.github.io/scalecrafter/)]



## infinity scene /360
[arxiv 2023.12]Going from Anywhere to Everywhere[[PDF](https://arxiv.org/abs/2312.03884),[Page](https://kovenyu.com/wonderjourney/)]

[arxiv 2024.1]360DVD: Controllable Panorama Video Generation with 360-Degree Video Diffusion Model [[PDF](https://arxiv.org/abs/2401.06578)]



## :point_right: Controllable Video Generation 

*[arxiv 2023.04]Motion-Conditioned Diffusion Model for Controllable Video Synthesis [[PDF](https://arxiv.org/abs/2304.14404), [Page](https://tsaishien-chen.github.io/MCDiff/)]

[arxiv 2023.06]Video Diffusion Models with Local-Global Context Guidance [[PDF](https://arxiv.org/abs/2306.02562)]

[arxiv 2023.06]VideoComposer: Compositional Video Synthesis with Motion Controllability [[PDF](https://arxiv.org/abs/2306.02018)]

[arxiv 2023.07]Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation [[PDF](https://arxiv.org/abs/2307.06940), [Page](https://videocrafter.github.io/Animate-A-Story)]

[arxiv 2023.10]MotionDirector: Motion Customization of Text-to-Video Diffusion Models [[PDF](https://arxiv.org/abs/2310.08465),[Page](https://showlab.github.io/MotionDirector/)]

[arxiv 2023.11]MagicDance: Realistic Human Dance Video Generation with Motions & Facial Expressions Transfer [[PDF](https://arxiv.org/abs/2311.12052), [Page](https://boese0601.github.io/magicdance/)]

[arxiv 2023.11]Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer[[PDF](https://arxiv.org/abs/2311.17009),[Page](https://diffusion-motion-transfer.github.io/)]

[arxiv 2023.11]SparseCtrl: Adding Sparse Controls to Text-to-Video Diffusion Models[[PDF](https://arxiv.org/abs/2311.16933), [Page](https://guoyww.github.io/projects/SparseCtrl)]

[arxiv 2023.11]MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model [[PDF](https://arxiv.org/abs/2311.16498), [Page](https://showlab.github.io/magicanimate)]

[arxiv 2023.12]DreaMoving: A Human Dance Video Generation Framework based on Diffusion Models [[PDF](https://arxiv.org/abs/2312.05107), [Page](https://dreamoving.github.io/dreamoving)]

[arxiv 2023.12]MotionCtrl: A Unified and Flexible Motion Controller for Video Generation [[PDF](https://arxiv.org/abs/2312.03641),[Page](https://wzhouxiff.github.io/projects/MotionCtrl/)]

[arxiv 2023.12]Fine-grained Controllable Video Generation via Object Appearance and Context [[PDF](https://arxiv.org/abs/2312.02919),[Page](https://hhsinping.github.io/factor)]

[arxiv 2023.12]Drag-A-Video: Non-rigid Video Editing with Point-based Interaction [[PDF](https://arxiv.org/abs/2312.02936),[Page](https://drag-a-video.github.io/)]

[arxiv 2023.12]Peekaboo: Interactive Video Generation via Masked-Diffusion [[PDF](https://arxiv.org/abs/2312.07509),[Page](https://jinga-lala.github.io/projects/Peekaboo/)]

[arxiv 2023.12]InstructVideo: Instructing Video Diffusion Models with Human Feedback [[PDF](https://arxiv.org/abs/2312.12490),[Page](https://instructvideo.github.io/)]



## Video in/outpainting 
[MM 2023.09]Hierarchical Masked 3D Diffusion Model for Video Outpainting [[PDF](https://arxiv.org/abs/2309.02119)]

[arxiv 2023.11]Flow-Guided Diffusion for Video Inpainting [[PDF](https://arxiv.org/abs/2311.15368)]



## Video SR 
[arxiv 2023.11]Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models [[PDF](https://arxiv.org/abs/2311.15908)]

[arxiv 2023.12]Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution [[PDF](https://arxiv.org/abs/2312.06640),[Page](https://shangchenzhou.com/projects/upscale-a-video/)]

[arxiv 2023.12]Video Dynamics Prior: An Internal Learning Approach for Robust Video Enhancements [[PDF](https://arxiv.org/abs/2312.07835),[Page](http://www.cs.umd.edu/~gauravsh/vdp.html)]



## downstream apps
[arxiv 2023.11]Breathing Life Into Sketches Using Text-to-Video Priors [[PDF](https://arxiv.org/abs/2311.13608),[Page](https://livesketch.github.io/)]

[arxiv 2023.11]Flow-Guided Diffusion for Video Inpainting [[PDF](https://arxiv.org/abs/2311.15368)]


## Video Concept 
[arxiv 2023.07]Animate-A-Story: Storytelling with Retrieval-Augmented Video Generation [[PDF](https://arxiv.org/abs/2307.06940), [Page](https://videocrafter.github.io/Animate-A-Story)]

[arxiv 2023.11]VideoDreamer: Customized Multi-Subject Text-to-Video Generation with Disen-Mix Finetuning[[PDF](https://arxiv.org/pdf/%3CARXIV%20PAPER%20ID%3E.pdf),[Page](https://videodreamer23.github.io/)]

[arxiv 2023.12]VideoAssembler: Identity-Consistent Video Generation with Reference Entities using Diffusion Model [[PDF](https://arxiv.org/abs/2311.17338),[Page](https://gulucaptain.github.io/videoassembler/)]

[arxiv 2023.12]VideoBooth: Diffusion-based Video Generation with Image Prompts [[PDF](https://arxiv.org/abs/2312.00777),[Page](https://vchitect.github.io/VideoBooth-project/)]

[arxiv 2023.12]DreamVideo: Composing Your Dream Videos with Customized Subject and Motion [[PDF](https://arxiv.org/abs/2312.04433),[Page](https://dreamvideo-t2v.github.io/)]

[arxiv 2023.12]PIA: Your Personalized Image Animator via Plug-and-Play Modules in Text-to-Image Models [[PDF](https://pi-animator.github.io/)]



## Image-to-video Generation 
[arxiv 2023.09]VideoGen: A Reference-Guided Latent Diffusion Approach for High Definition Text-to-Video Generation [[PDF](https://arxiv.org/abs/2309.00398)]

[arxiv 2023.09]Generative Image Dynamics [[PDF](https://arxiv.org/abs/2309.07906),[Page](http://generative-dynamics.github.io/)]

[arxiv 2023.10]DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors [[PDF](https://arxiv.org/abs/2310.12190), [Page](https://github.com/AILab-CVC/VideoCrafter)]

[arxiv 2023.11]SEINE: Short-to-Long Video Diffusion Model for Generative Transition and Prediction [[PDF](https://arxiv.org/abs/2310.20700),[Page](https://vchitect.github.io/SEINE-project/)]

[arxiv 2023.11]I2VGen-XL: High-Quality Image-to-Video Synthesis via Cascaded Diffusion Models
[[PDF](https://arxiv.org/abs/2311.04145),[Page](https://i2vgen-xl.github.io/page04.html)]

[arxiv 2023.11]Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning [[PDF](https://arxiv.org/abs/2311.10709),[Page](https://emu-video.metademolab.com/)]

[arxiv 2023.11]MoVideo: Motion-Aware Video Generation with Diffusion Models[[PDF](https://github.com/JingyunLiang/MoVideo/releases/download/v0.0/MoVideo.pdf),[Page](https://jingyunliang.github.io/MoVideo/)]

[arxiv 2023.11]Make Pixels Dance: High-Dynamic Video Generation[[PDF](),[Page](https://makepixelsdance.github.io/)]

[arxiv 2023.11]Decouple Content and Motion for Conditional Image-to-Video Generation [[PDF](https://arxiv.org/abs/2311.14294)]

[arxiv 2023.12]ART•V: Auto-Regressive Text-to-Video Generation with Diffusion Models [[PDF](https://arxiv.org/abs/2311.18834), [Page](https://warranweng.github.io/art.v)]

[arxiv 2023.12]MicroCinema: A Divide-and-Conquer Approach for Text-to-Video Generation [[PDF](https://arxiv.org/abs/2311.18829), [Page](https://wangyanhui666.github.io/MicroCinema.github.io/)]

[arxiv 2023.12]DreamVideo: High-Fidelity Image-to-Video Generation with Image Retention and Text Guidance [[PDF](https://arxiv.org/abs/2312.03018),[Page](https://anonymous0769.github.io/DreamVideo/)]

[arxiv 2023.12]LivePhoto: Real Image Animation with Text-guided Motion Control [[PDF](https://arxiv.org/abs/2312.02928), [Page](https://xavierchen34.github.io/LivePhoto-Page/)]

[arxiv 2023.12]I2V-Adapter: A General Image-to-Video Adapter for Video Diffusion Models [[PDF](https://arxiv.org/abs/2312.16693)]



## 4D generation 
[arxiv 2023.11]Animate124: Animating One Image to 4D Dynamic Scene [[PDF](https://arxiv.org/abs/2311.14603),[Page](https://animate124.github.io/)]

[arxiv 2023.12]4D-fy: Text-to-4D Generation Using Hybrid Score Distillation Sampling[[PDF](https://arxiv.org/abs/2311.17984), [Page](https://sherwinbahmani.github.io/4dfy)]

[arxiv 2023.12]4DGen: Grounded 4D Content Generation with Spatial-temporal Consistency [[PDF](https://arxiv.org/abs/2312.17225),[Page](https://vita-group.github.io/4DGen/)]

[arxiv 2023.12]DreamGaussian4D: Generative 4D Gaussian Splatting [[PDF](https://arxiv.org/abs/2312.17142), [Page](https://jiawei-ren.github.io/projects/dreamgaussian4d)]



## Audio-to-video Generation
[arxiv 2023.09]Diverse and Aligned Audio-to-Video Generation via Text-to-Video Model Adaptation [[PDF](https://arxiv.org/abs/2309.16429)]



## Video editing with video models 
[arxiv 2023.12]VIDiff: Translating Videos via Multi-Modal Instructions with Diffusion Models[[PDF](https://arxiv.org/abs/2311.18837),[Page](https://chenhsing.github.io/VIDiff)]

[arxiv 2023.12]Neutral Editing Framework for Diffusion-based Video Editing [[PDF](https://arxiv.org/abs/2312.06708),[Page](https://neuedit.github.io/)]

[arxiv 2024.01]FlowVid: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis[[PDF](https://arxiv.org/abs/2312.17681),[Page](https://jeff-liangf.github.io/projects/flowvid/)]




## Image Model for video generation and editing 
*[arxiv 2022.12]Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation [[PDF](https://arxiv.org/abs/2212.11565), [Page](https://tuneavideo.github.io/)]

[arxiv 2023.03]Video-P2P: Video Editing with Cross-attention Control [[PDF](https://arxiv.org/abs/2303.04761), [Page](https://video-p2p.github.io/)]

[arxiv 2023.03]Edit-A-Video: Single Video Editing with Object-Aware Consistency [[PDF](https://arxiv.org/abs/2303.07945), [Page](https://edit-a-video.github.io/)]

[arxiv 2023.03]FateZero: Fusing Attentions for Zero-shot Text-based Video Editing [[PDF](https://arxiv.org/abs/2303.09535), [Page](https://github.com/ChenyangQiQi/FateZero)]

[arxiv 2023.03]Pix2Video: Video Editing using Image Diffusion [[PDF](https://arxiv.org/abs/2303.12688)]

->[arxiv 2023.03]Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators [[PDF](https://arxiv.org/abs/2303.13439), [code](https://github.com/Picsart-AI-Research/Text2Video-Zero)]

[arxiv 2023.03]Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models[[PDF](https://arxiv.org/abs/2303.17599),[code](https://github.com/baaivision/vid2vid-zero)]

[arxiv 2023.04]Follow Your Pose: Pose-Guided Text-to-Video Generation using Pose-Free Videos[[PDF](https://arxiv.org/abs/2304.01186)]

[arxiv 2023.05]ControlVideo: Training-free Controllable Text-to-Video Generation [[PDF](https://arxiv.org/abs/2305.13077), [Page](https://github.com/YBYBZhang/ControlVideo)]

[arxiv 2023.05]Control-A-Video: Controllable Text-to-Video Generation with Diffusion Models[[PDF](https://arxiv.org/abs/2305.13840), [Page](https://controlavideo.github.io/)]

[arxiv-2023.05]Large Language Models are Frame-level Directors for Zero-shot Text-to-Video Generation [[PDF](https://arxiv.org/abs/2305.14330), [Page](https://github.com/KU-CVLAB/DirecT2V)]

[arxiv 2023.05]Video ControlNet: Towards Temporally Consistent Synthetic-to-Real Video Translation Using Conditional Image Diffusion Models [[PDF](https://arxiv.org/abs/2305.19193)]

[arxiv 2023.05]SAVE: Spectral-Shift-Aware Adaptation of Image Diffusion Models for Text-guided Video Editing [[PDF](https://arxiv.org/abs/2305.18670)]

[arxiv 2023.05]InstructVid2Vid: Controllable Video Editing with Natural Language Instructions [[PDF](https://arxiv.org/abs/2305.12328)]

[arxiv 2023.05] ControlVideo: Adding Conditional Control for One Shot Text-to-Video Editing [[PDF](https://arxiv.org/pdf/2305.17098.pdf), [Page](https://ml.cs.tsinghua.edu.cn/controlvideo/)]

[arxiv 2023.05]Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising [[PDF](https://arxiv.org/abs/2305.18264),[Page](https://g-u-n.github.io/projects/gen-long-video/index.html)]

[arxiv 2023.06]Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance [[PDF](https://arxiv.org/abs/2306.00943), [Page](https://doubiiu.github.io/projects/Make-Your-Video/)]

[arxiv 2023.06]VidEdit: Zero-Shot and Spatially Aware Text-Driven Video Editing [[PDF](https://arxiv.org/abs/2306.08707),[Page](https://videdit.github.io/)]

*[arxiv 2023.06]Rerender A Video: Zero-Shot Text-Guided Video-to-Video Translation [[PDF](https://arxiv.org/abs/2306.07954), [Page](https://anonymous-31415926.github.io/)]

*[arxiv 2023.07]AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning [[PDF](https://arxiv.org/abs/2307.04725),  [Page](https://animatediff.github.io/)]

*[arxiv 2023.07]TokenFlow: Consistent Diffusion Features for Consistent Video Editing [[PDF](https://arxiv.org/pdf/2307.10373.pdf),[Page](https://diffusion-tokenflow.github.io/)]

[arxiv 2023.07]VideoControlNet: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet [[PDF](https://arxiv.org/pdf/2307.14073.pdf), [Page](https://vcg-aigc.github.io/)]

[arxiv 2023.08]CoDeF: Content Deformation Fields for Temporally Consistent Video Processing [[PDF](https://arxiv.org/pdf/2308.07926.pdf), [Page](https://qiuyu96.github.io/CoDeF/)]

[arxiv 2023.08]DragNUWA: Fine-grained Control in Video Generation by Integrating Text, Image, and Trajectory [[PDF](https://arxiv.org/abs/2308.08089), [Page](https://www.microsoft.com/en-us/research/project/dragnuwa/)]

[arxiv 2023.08]StableVideo: Text-driven Consistency-aware Diffusion Video Editing [[PDF](https://arxiv.org/abs/2308.09592), [Page](https://github.com/rese1f/StableVideo)]

[arxiv 2023.08]Edit Temporal-Consistent Videos with Image Diffusion Model [[PDF](https://arxiv.org/abs/2308.09091)]

[arxiv 2023.08]EVE: Efficient zero-shot text-based Video Editing with Depth Map Guidance and Temporal Consistency Constraints [[PDF](https://arxiv.org/pdf/2308.10648.pdf)]

[arxiv 2023.08]MagicEdit: High-Fidelity and Temporally Coherent Video Editing [[PDF](https://arxiv.org/pdf/2308.14749), [Page](https://magic-edit.github.io/)]

[arxiv 2023.09]MagicProp: Diffusion-based Video Editing via Motionaware Appearance Propagation[[PDF](https://arxiv.org/pdf/2309.00908.pdf)]

[arxiv 2023.09]Free-Bloom: Zero-Shot Text-to-Video Generator with LLM Director and LDM Animator[[PDF](https://arxiv.org/abs/2309.14494), [Page](https://github.com/SooLab/Free-Bloom)]

[arxiv 2023.09]CCEdit: Creative and Controllable Video Editing via Diffusion Models [[PDF](https://arxiv.org/abs/2309.16496)]

[arxiv 2023.10]Ground-A-Video: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models [[PDF](https://arxiv.org/abs/2310.01107),[Page](https://ground-a-video.github.io/)]

[arxiv 2023.10]FLATTEN: optical FLow-guided ATTENtion for consistent text-to-video editing [[PDF](https://arxiv.org/abs/2310.05922),[Page](https://flatten-video-editing.github.io/)]

[arxiv 2023.10]ConditionVideo: Training-Free Condition-Guided Text-to-Video Generation [[PDF](https://arxiv.org/abs/2310.07697),[Page](https://pengbo807.github.io/conditionvideo-website/)]

[arxiv 2023.10, nerf] DynVideo-E: Harnessing Dynamic NeRF for Large-Scale Motion- and View-Change Human-Centric Video Editing [[PDF](https://arxiv.org/abs/2310.10624), [Page](https://showlab.github.io/DynVideo-E/)]

[arxiv 2023.10]LAMP: Learn A Motion Pattern for Few-Shot-Based Video Generation [[PDF](https://arxiv.org/abs/2310.10769),[Page](https://rq-wu.github.io/projects/LAMP/index.html)]

[arxiv 2023.11]LATENTWARP: CONSISTENT DIFFUSION LATENTS FOR ZERO-SHOT VIDEO-TO-VIDEO TRANSLATION [[PDF](https://arxiv.org/pdf/2311.00353.pdf)]

[arxiv 2023.11]Cut-and-Paste: Subject-Driven Video Editing with Attention Control[[PDF](https://arxiv.org/abs/2311.11697)]

[arxiv 2023.11]MotionZero:Exploiting Motion Priors for Zero-shot Text-to-Video Generation [[PDF](https://arxiv.org/abs/2311.16635)]

[arxiv 2023.12]Motion-Conditioned Image Animation for Video Editing [[PDF](https://arxiv.org/pdf/2311.18827.pdf), [Page](https://facebookresearch.github.io/MoCA/)]

[arxiv 2023.12]RAVE: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models [[PDF](https://arxiv.org/abs/2312.04524),[Page](https://rave-video.github.io/)]

[arxiv 2023.12]DiffusionAtlas: High-Fidelity Consistent Diffusion Video Editing [[PDF](https://arxiv.org/abs/2312.03772)]

[arxiv 2023.12]MagicStick: Controllable Video Editing via Control Handle Transformations [[PDF](https://arxiv.org/abs/2312.03047),[Page](https://github.com/mayuelala/MagicStick)]

[arxiv 2023.12]SAVE: Protagonist Diversification with Structure Agnostic Video Editing [[PDF](https://arxiv.org/abs/2312.02503),[Page](https://ldynx.github.io/SAVE/)]

[arxiv 2023.12]VidToMe: Video Token Merging for Zero-Shot Video Editing [[PDF](https://arxiv.org/abs/2312.10656),[Page](https://vidtome-diffusion.github.io/)]

[arxiv 2023.12]Fairy: Fast Parallelized Instruction-Guided Video-to-Video Synthesis [[PDF](https://arxiv.org/abs/2312.13834),[Page](https://fairy-video2video.github.io/)]

[arxiv 2024.1]Object-Centric Diffusion for Efficient Video Editing [[PDF](https://arxiv.org/abs/2401.05735)]

[arxiv 2024.1]VASE: Object-Centric Shape and Appearance Manipulation of Real Videos [[PDF](https://arxiv.org/abs/2401.02473),[Page](https://helia95.github.io/vase-website/)]



## :point_right: Video Completion (animation, interpolation, prediction)
[arxiv 2022; Meta] Tell Me What Happened: Unifying Text-guided Video Completion via Multimodal Masked Video Generation \[[PDF](https://arxiv.org/pdf/2211.12824.pdf), code]

[arxiv 2023.03]LDMVFI: Video Frame Interpolation with Latent Diffusion Models[[PDF](https://arxiv.org/abs/2303.09508)]

*[arxiv 2023.03]Seer: Language Instructed Video Prediction with Latent Diffusion Models [[PDF](https://arxiv.org/abs/2303.14897)]


## motion transfer 
[arxiv 2023.05]LEO: Generative Latent Image Animator for Human Video Synthesis [[PDF](https://arxiv.org/abs/2305.03989),[Page](https://wyhsirius.github.io/LEO-project/)]

*[arxiv 2023.03]Conditional Image-to-Video Generation with Latent Flow Diffusion Models [[PDF](https://arxiv.org/abs/2303.13744)]

[arxiv 2023.07]DisCo: Disentangled Control for Referring Human Dance Generation in Real World
[[PDF](https://arxiv.org/abs/2307.00040), [Page](https://disco-dance.github.io/)]

[arxiv 2023.12]MotionEditor: Editing Video Motion via Content-Aware Diffusion [[PDF](https://arxiv.org/abs/2311.18830),[Page](https://francis-rings.github.io/MotionEditor/)]

[arxiv 2023.12]Customizing Motion in Text-to-Video Diffusion Models [[PDF](https://arxiv.org/abs/2312.04966),[Page](https://joaanna.github.io/customizing_motion/)]

[arxiv 2023.12]MotionCrafter: One-Shot Motion Customization of Diffusion Models [[PDF](https://arxiv.org/abs/2312.05288)]




## style transfer 
[arxiv 2023.06]Probabilistic Adaptation of Text-to-Video Models [[PDF](https://arxiv.org/abs/2306.01872)]

[arxiv 2023.11]Highly Detailed and Temporal Consistent Video Stylization via Synchronized Multi-Frame Diffusion[[PDF](https://arxiv.org/abs/2311.14343)]

[arxiv 2023.12]StyleCrafter: Enhancing Stylized Text-to-Video Generation with Style Adapter[[PDF](https://arxiv.org/abs/2312.00330),[Page](https://gongyeliu.github.io/StyleCrafter.github.io/)]

[arxiv 2023.12]DragVideo: Interactive Drag-style Video Editing [[PDF](https://arxiv.org/abs/2312.02216)]




## Evaluation 
[arxiv 2023.10]EvalCrafter: Benchmarking and Evaluating Large Video Generation Models [[PDF](https://arxiv.org/abs/2310.11440),[Page](https://evalcrafter.github.io/)]

[arxiv 2023.11]FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation [[PDF](https://arxiv.org/abs/2311.01813)]

[arxiv 2023.11]Online Video Quality Enhancement with Spatial-Temporal Look-up Tables [[PDF](https://arxiv.org/abs/2311.13616)]


## Survey
[arxiv 2023.03]A Survey on Video Diffusion Models [[PDF](https://arxiv.org/abs/2310.10647)]



## Evaluation 
[ICCV 2023]Exploring Video Quality Assessment on User Generated Contents from Aesthetic and Technical Perspectives [[PDF](https://arxiv.org/abs/2211.04894),[Page](https://github.com/VQAssessment/DOVER)]

[arxiv 2023.10]EvalCrafter: Benchmarking and Evaluating Large Video Generation Models[[PDF](https://arxiv.org/abs/2310.11440), [Page](https://arxiv.org/abs/2310.11440)]

[arxiv 2023.11]HIDRO-VQA: High Dynamic Range Oracle for Video Quality Assessment [[PDF](https://arxiv.org/abs/2311.11059)]

[arxiv 2023.12]VBench: Comprehensive Benchmark Suite for Video Generative Models [[PDF](https://arxiv.org/abs/2311.17982), [Page](https://vchitect.github.io/VBench-project/)]


## Speed 
[arxiv 2023.12]F3-Pruning: A Training-Free and Generalized Pruning Strategy towards Faster and Finer Text-to-Video Synthesis [[PDF](https://arxiv.org/abs/2312.03459)]

[arxiv 2023.12]VideoLCM: Video Latent Consistency Model [[PDF](https://arxiv.org/abs/2312.09109)]

[arxiv 2024.01]FlashVideo: A Framework for Swift Inference in Text-to-Video Generation [[PDF](https://arxiv.org/abs/2401.00869)]



## Others 
[arxiv 2023.05]AADiff: Audio-Aligned Video Synthesis with Text-to-Image Diffusion [[PDF](https://arxiv.org/abs/2305.04001)]

[arxiv 2023.05]Multi-object Video Generation from Single Frame Layouts [[PDF](https://arxiv.org/abs/2305.03983)]

[arxiv 2023.06]Learn the Force We Can: Multi-Object Video Generation from Pixel-Level Interactions [[PDF](https://arxiv.org/abs/2306.03988)]

[arxiv 2023.08]DiffSynth: Latent In-Iteration Deflickering for Realistic Video Synthesis [[PDF](https://arxiv.org/abs/2308.03463)]



## CV Related 
[arxiv 2022.12; ByteDace]PV3D: A 3D GENERATIVE MODEL FOR PORTRAIT VIDEO GENERATION [[PDF](https://arxiv.org/pdf/2212.06384.pdf)]

[arxiv 2022.12]MM-Diffusion: Learning Multi-Modal Diffusion Models for Joint Audio and Video Generation[[PDF](https://arxiv.org/pdf/2212.09478.pdf)]

[arxiv 2022.12]Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation [[PDF](https://arxiv.org/pdf/2212.11565.pdf), [Page](https://tuneavideo.github.io/)]

[arxiv 2023.01]Diffused Heads: Diffusion Models Beat GANs on Talking-Face Generation [[PDF](https://arxiv.org/pdf/2301.03396.pdf), [Page](https://mstypulkowski.github.io/diffusedheads)]

[arxiv 2023.01]DiffTalk: Crafting Diffusion Models for Generalized Talking Head Synthesis [[PDF](https://arxiv.org/pdf/2301.03786.pdf), [Page](https://sstzal.github.io/DiffTalk/)]

[arxiv 2023.02 Google]Scaling Vision Transformers to 22 Billion Parameters [[PDF](https://arxiv.org/abs/2302.05442)]

[arxiv 2023.05]VDT: An Empirical Study on Video Diffusion with Transformers [[PDF](https://arxiv.org/abs/2305.13311), [code](https://github.com/RERV/VDT)]


## NLP related
[arxiv 2022.10]DIFFUSEQ: SEQUENCE TO SEQUENCE TEXT GENERATION WITH DIFFUSION MODELS [[PDF](https://arxiv.org/pdf/2210.08933.pdf)]

[arxiv 2023.02]The Flan Collection: Designing Data and Methods for Effective Instruction Tuning [[PDF](https://arxiv.org/pdf/2301.13688.pdf)]


## Speech 
[arxiv 2023.01]Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers[[PDF](https://arxiv.org/abs/2301.02111), [Page](https://valle-demo.github.io/)]

