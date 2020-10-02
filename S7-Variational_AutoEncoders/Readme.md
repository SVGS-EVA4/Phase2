# Session7 - Variational AutoEncoder 
[![](https://img.shields.io/badge/Website-green.svg)]() [![](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/SVGS-EVA4/Phase2/blob/master/S7-Variational_AutoEncoders/VAE_Final.ipynb) 

Autoencoder is a neural network designed to learn an identity function in an unsupervised way to reconstruct the original input while compressing the data in the process so as to discover a more efficient and compressed representation.

Variational Autoencoders (VAEs) have one fundamentally unique property that separates them from vanilla autoencoders, and it is this property that makes them so useful for generative modeling: their latent spaces are, by design, continuous, allowing easy random sampling and interpolation.

It achieves this by doing something that seems rather surprising at first: making its encoder not output an encoding vector of size n, rather, outputting two vectors of size n: a vector of means, μ, and another vector of standard deviations, σ

![VAE-Architecture]()

## **Assignment**

     To train Variational Auto Encoder to reconstruct an input image of a car.


## **Dataset**



[![](https://img.shields.io/badge/DataSet-blue.svg)](https://drive.google.com/file/d/1G5sKYPPYAteKzWn6fWsACtIF9W635Frx/view?usp=sharing)
[![](https://img.shields.io/badge/DataSet%20Preprocessing-blue.svg)](https://github.com/SVGS-EVA4/Phase2/blob/master/S6-Generative_Adversarial_Networks/Preprocessing.ipynb)


![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S7-Variational_AutoEncoders/asset/ds.PNG)


## **Model Architecture**


![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S7-Variational_AutoEncoders/asset/vae.PNG)


## **Paramenters and Hyperparameters**
* Loss Function: Mean Squared Loss + Kullback–Leibler Divergence Loss
* Epochs: 1300
* Optimizer: Adam
* Learning Rate: 0.001 for first 1000 epochs then 0.0001 for next 300 epochs.
* Batch Size: 128
* Image size: 128


## **VAE Loss Plot**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S7-Variational_AutoEncoders/asset/loss_graph.PNG)

## **Resutls**


![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S7-Variational_AutoEncoders/asset/eval.PNG)
