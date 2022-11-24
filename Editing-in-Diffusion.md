# Image Editing In Diffusion 


## Inversion 

**[Arxiv.2208; NVIDIA]**  ***An Image is Worth One Word:*** Personalizing Text-to-Image Generation using Textual Inversion [[Paper & Code]](https://github.com/rinongal/textual_inversion)

**[ICLR2022; Stanford & CMU]** ***SDEdit:*** Guided Image Synthesis and Editing with Stochastic Differential Equations [[Project]](https://sde-image-editing.github.io/) [[Paper]](https://arxiv.org/pdf/2108.01073.pdf)

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

[arxiv 2022] EDICT: Exact Diffusion Inversion via Coupled Transformations \[[PDF](https://arxiv.org/abs/2211.12446)\]

## Story-telling
[arxiv 22.11; Ailibaba] Synthesizing Coherent Story with Auto-Regressive Latent Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.10950.pdf), code\]  
<details>
<summary>Details</summary>
Challange: incorporating history captions and scenes for current image generation  
Method: Use history latent as information (Auto-Regressive)  
  x_0 is provided; CLIP for current caption encoding; BLIP for the previous caption and generated image encoding;  
  Adaptive Learning: for new characters: like Dreambooth, using new words to represent it and finetune the G to remember the character.
</details>

## Inpainting 
[arxiv 2022; MSRA]Paint by Example: Exemplar-based Image Editing with Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.13227.pdf), code\]  

## Style transfer 
[arxiv 22.11; kuaishou] ***DiffStyler***: Controllable Dual Diffusion for Text-Driven Image Stylization \[[PDF](https://arxiv.org/pdf/2211.10682.pdf), code\]  

[ICLR 23] TEXT-GUIDED DIFFUSION IMAGE STYLE TRANSFER WITH CONTRASTIVE LOSS [[Paper]](https://openreview.net/pdf?id=iJ_E0ZCy8fi)  

[arxiv 2022] Inversion-Based Creativity Transfer with Diffusion Models \[[PDF](https://arxiv.org/pdf/2211.13203.pdf)\]
