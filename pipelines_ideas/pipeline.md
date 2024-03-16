- Stable Diffusion 3 is built on Dit(Diffusion Transformer) while this version is not.

- This is built on based off on U-net


- Key Points
  - Latent Diffusion
  - Scheduler and all other details --> DDPM
  - Classifier-Free Guidance
  - Functionality: 
    - image -> image
    - text -> image
  - Inpainting
  - Signal:
    - Prompt/context
      - Telling the model 'the direction' of noise prediction
      - This is meant for generating New Data
    - We aim to combine the distribution of the image and the prompt
      - But also we do not want only the joint distribution of those 2, because in essence we are looking for learning the way the original image was generated, joining those 2 distributions would be too off-path in this case
    - Learning both the output of the noise and the difference of conditional and unconditional learnt results
  - CLIP
    - Contrastive Learning: prompt-to-image correspondence, matrix diagonal maximization




TO-DO:
- Mathematical Perspective
  - Joint Distribution of Pixels
    - Higher-Dimension Gaussian - Intuition & Maths Visualization
  - 