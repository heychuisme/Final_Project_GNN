# Final Project

**of lecture Generative Neural Networks for the Sciences**

This project is based on the work *Domain Expansion of Image Generators* of Nitzan et al. Most part of the codes are cited from their GitHub repo which is build based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).

## Abstract

Generative Adversarial Networks (GANs) have revolutionized image generation, demonstrating remarkable capability in producing highly realistic images. Traditionally, these models excel within the confines of their training data, often struggling when tasked with generating images beyond their initial domain. This limitation underscores the necessity for models that can transcend these boundaries without the need for retraining, a concept known as domain expansion. Domain expansion builds upon the principles of domain adaptation, enabling GANs to adapt from generating images in their original domain to creating diverse outputs in previously unexplored domains. This process leverages the untapped potential within the latent space, particularly focusing on dormant directions that, when activated, facilitate the introduction of novel features into the generated images. Our research employs StyleGAN as a foundational model, not only to exploit its advanced image generation capabilities but also to explore domain expansion's potential in broadening the model's versatility across various domains. By factorizing the latent space and harnessing dormant directions, we aim to enhance the model's creative scope, enabling the generation of a wider array of images beyond its original training dataset. This exploration not only enriches the diversity of generated images but also offers insights into the structured yet expansive nature of the latent space, marking a significant advancement in the field of generative modeling.

## Setup

The environment needed can be built via:

```
conda env create -f environment.yml
```

## Training

The train.py contains detailed help list which enables you set various training parameters.

```
python train.py --help
```

Similar to the original project, the adaption configuration used for training can be found in config/config.json as well as further customized.

## Inference

The inference process is the same to the original paper:

```
python generate_aligned.py --ckpt network.pkl --out_dir generated --num 10
```

