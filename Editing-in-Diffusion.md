# Image Editing In Diffusion [Open in Notion](https://vinthony.notion.site/Diffusion-Edit-84312204c0c446e4b9bcc50a1dce84e1)


## Inversion 

**[Arxiv.2208; NVIDIA]**  ***An Image is Worth One Word:*** Personalizing Text-to-Image Generation using Textual Inversion [[Paper & Code]](https://github.com/rinongal/textual_inversion)  
Learnable text embedding; Loss with multiple diffusion step noise.

**[ICLR2022; Stanford & CMU]** ***SDEdit:*** Guided Image Synthesis and Editing with Stochastic Differential Equations [[Project]](https://sde-image-editing.github.io/) [[Paper]](https://arxiv.org/pdf/2108.01073.pdf)  
Add noise then denoise. Not robust.

**[arxiv 22.08; meta]** ***Prompt-to-Prompt*** Image Editing with Cross Attention Control [[Paper]](https://arxiv.org/abs/2208.01626)  
Switch Cross-Attention to maintain spatial structure information.

**[arxiv 22.08; Scale AI]** ***Direct Inversion***: Optimization-Free Text-Driven Real Image Editing with Diffusion Models [[Paper]](https://arxiv.org/pdf/2211.07825)  
Use encoder to predict noise / latent of Stable Diffusion.

**[arxiv 22.11; UC Berkeley]** ***InstructPix2Pix***: Learning to Follow Image Editing Instructions [[Project]](https://www.timothybrooks.com/instruct-pix2pix)  [[Paper]](https://arxiv.org/pdf/2211.09800.pdf)  
GPT-3 & Stable Diffusion to construct a dataset~(450,000), fine-tune the Stable-Diffusion with generated data; Two-condition using extrapolated score estimation.  
Strong & flexible sentence editing.  

**[NIPS 22; google]** ***DreamBooth***: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation [[Project]](https://dreambooth.github.io/) [[Paper]](https://arxiv.org/abs/2208.12242) [[Code (Unofficial)]](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)  
Random characters as inversion target; Language Drift: same-class object are forgotten to generate; Space regularization loss for solving such problems; Fine-tune SR model; 



**[ICLR 23]** ***DiffEdit***: Diffusion-based semantic image editing with mask guidance [[Paper]](https://openreview.net/forum?id=3lge0p5o-M-)  
Predict a mask for local editing.

**[ICLR 23]** ***DiffIT***: Diffusion-based Image Translation Using Disentangled Style and Content Repesentation [[Paper]](https://openreview.net/pdf?id=Nayau9fwXU)  
<details>
<summary>Details</summary>
Nice preliminary: former methods directly incoporate the gradient of a pre-trained classifer (classify the editimated x_0) to modify the x during the training or inference;  
Structure Maintain: 
Self-similarity loss: the cosine distance between the DINO-VIT tokens of one images; calulate the distance between x and x_src; Use additional Contrastive Loss to enbale same position have closer distance;  
Style Maintain: (Text)  CLIP-direction loss; Ensemble CLIP embedding; (Image-guided) CLS Token L2 loss  
Speed-up: Enable the current t CLS different from the previous t CLS token  
Resampling strategy: sample k step to find whose gradient is easily affected by the loss.  
Weakness: Only words editing
</details>
  
**[ICLR 23]** Dual Diffusion Implicit Bridges for Image-to-image Translation [[Paper]](https://openreview.net/pdf?id=5HLoTvVGDe)  
Two diffusion; Math hard to understand

**[ICLR 23, Google]** Classifier-free Diffusion Guidance [[Paper]](https://arxiv.org/pdf/2207.12598.pdf)

**[arxiv 2022]** ***EDICT***: Exact Diffusion Inversion via Coupled Transformations \[[PDF](https://arxiv.org/abs/2211.12446)\]  
Affine Coupling Layer in Flow to make the inversion invertable; Mixing weight to stable the process.

**[arxiv 22.11]** ***Paint by Example***: Exemplar-based Image Editing with Diffusion Models [[PDF]](https://arxiv.org/abs/2211.13227)  
Exampler image as the condition; BottleNeck: CLS token only pass the semantic information; Augmentation of reference image and mask.

[arxiv 2022.10; ByteDance]MagicMix: Semantic Mixing with Diffusion Models [[PDF]](https://arxiv.org/abs/2210.16056)  


[arxiv 2022.12; UT] Multiresolution Textual Inversion [[PDF]](https://arxiv.org/abs/2210.16056)  

[arxiv 2022.12; Microsoft]X-Paste: Revisit Copy-Paste at Scale with CLIP and StableDiffusion\[[PDF](https://arxiv.org/pdf/2212.03863.pdf)\]

[arxi 2022.12]SINE: SINgle Image Editing with Text-to-Image Diffusion Models \[[PDF](https://arxiv.org/pdf/2212.04489.pdf)\]

[arxiv 2022.12]Multi-Concept Customization of Text-to-Image Diffusion \[[PDF](https://arxiv.org/abs/2212.04488)\]



## Story-telling

**[arxiv 22.09]** ***Story Dall-E***: Adapting pretrained text-to-image transformers for story continuation [[PDF]](https://arxiv.org/pdf/2209.06192.pdf)  
Based on Dall-E, use cross-attention to incorporate source image into the generation

**[arxiv 22.11; Ailibaba]** Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.10950.pdf), code\]  
<details>
<summary>Details</summary>
Challange: incorporating history captions and scenes for current image generation  
Method: Use history latent as information (Auto-Regressive)  
  x_0 is provided; CLIP for current caption encoding; BLIP for the previous caption and generated image encoding;  
  Adaptive Learning: for new characters: like Dreambooth, using new words to represent it and finetune the G to remember the character.
</details>

**[arxiv 2022]** ***Make-A-Story***: Visual Memory Conditioned Consistent Story Generation  \[[PDF](https://arxiv.org/pdf/2211.13319.pdf), code\]  


## SVG
[arxiv 2022.11; UCB] VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models \[[PDF](https://arxiv.org/abs/2211.11319)\]



## Translation
[arxiv 2022; Google]Sketch-Guided Text-to-Image Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.13752.pdf), code\]  

[arxiv 2022.11; Microsoft]ReCo: Region-Controlled Text-to-Image Generation \[[PDF](https://arxiv.org/pdf/2211.15518.pdf), code\]  

[arxiv 2022.11; Meta]SpaText: Spatio-Textual Representation for Controllable Image Generation  \[[PDF](https://arxiv.org/pdf/2211.14305.pdf), code\]  

**[arxiv 2022.11; Seoul National University]** ***DATID-3D:*** Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model. \[[PROJECT](https://datid-3d.github.io/)]  
**Incredible Stylization Reults.** Diffusion model to generate different pose data; CLIP to filter bad semantic case; Pose estimator to filter bad pose case; Regularization; DreamBooth for Instance adaptation.

[arxiv 2022.12]High-Fidelity Guided Image Synthesis with Latent Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.17084.pdf)\]  

[arxiv 2022.12]Fine-grained Image Editing by Pixel-wise Guidance Using Diffusion Models \[[PDF](https://arxiv.org/pdf/2212.02024.pdf)\]

[arxiv 2022.12]Towards Practical Plug-and-Play Diffusion Models [[PDF](https://arxiv.org/pdf/2212.05973.pdf)]


## Inpainting 
[arxiv 2022; MSRA]Paint by Example: Exemplar-based Image Editing with Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.13227.pdf), [code](https://github.com/Fantasy-Studio/Paint-by-Example)\]  


## Super-Resolution 
[arxiv 2022.12]ADIR: Adaptive Diffusion for Image Reconstruction  \[[PDF](https://shadyabh.github.io/ADIR/ADIR_files/ADIR.pdf)

## Style transfer 
[arxiv 22.11; kuaishou] ***DiffStyler***: Controllable Dual Diffusion for Text-Driven Image Stylization \[[PDF](https://arxiv.org/pdf/2211.10682.pdf), code\]  

[ICLR 23] TEXT-GUIDED DIFFUSION IMAGE STYLE TRANSFER WITH CONTRASTIVE LOSS [[Paper]](https://openreview.net/pdf?id=iJ_E0ZCy8fi)  

[arxiv 22.11; kuaishou&CAS] Inversion-Based Creativity Transfer with Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.13203.pdf), [Code](https://github.com/zyxElsa/creativity-transfer)\]

[arxiv 2022.12]Diff-Font: Diffusion Model for Robust One-Shot Font Generation [[PDF](https://arxiv.org/pdf/2212.05895.pdf)]


## Face swapping 
[arxiv 2022.12]HS-Diffusion: Learning a Semantic-Guided Diffusion Model for Head Swapping[[PDF](https://arxiv.org/pdf/2212.06458.pdf)]


## Related 
[arxiv 2022.11]Investigating Prompt Engineering in Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.15462.pdf)\] 

[arxiv 2022.11]Versatile Diffusion: Text, Images and Variations All in One Diffusion Model \[[PDF](https://arxiv.org/pdf/2211.08332.pdf)\] 

[arxiv 2022.11; ByteDance]Shifted Diffusion for Text-to-image Generation  \[[PDF](https://arxiv.org/pdf/2211.15388.pdf)\] 

[arxiv 2022.11; ]3DDesigner: Towards Photorealistic 3D Object Generation and Editing with Text-guided Diffusion Models  \[[PDF](https://arxiv.org/pdf/2211.14108.pdf)\] 

**[ECCV 2022; Best Paper]** ***Partial Distance:*** On the Versatile Uses of Partial Distance Correlation in Deep Learning. \[[PDF](https://arxiv.org/abs/2207.09684)\]  
Can measure the co-variance distance of any two distribution. Can be used for disentangling the common attributes!

[arxiv 2022.12]SinDDM: A Single Image Denoising Diffusion Model \[[PDF](https://arxiv.org/pdf/2211.16582.pdf)\] 

[arxiv 2022.12] Diffusion Guided Domain Adaptation of Image Generators \[[PDF](https://arxiv.org/pdf/2212.04473.pdf)\] 

[arxiv 2022.12]Training-Free Structured Diffusion Guidance for Compositional Text-to-Image Synthesis [[PDF](https://arxiv.org/pdf/2212.05032.pdf)]




## Repository
***DIFFUSERS*** Hugging-face sota repository. \[[DIFFUSERS](https://github.com/huggingface/diffusers)\]
