# Image Captioning

Implemented [pytorch image captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning) on flickr8k dataset. Trained the model from scratch and deployed in aws.

## Dataset

Flickr8K Dataset was used as colab provides around 68GB of disk space in GPU mode.

## Model

### Encoder

We use a pretrained ResNet-101 already available in PyTorch's `torchvision` module. Discard the last two layers (pooling and linear layers), since we only need to encode the image, and not classify it.

We do add an `AdaptiveAvgPool2d()` layer to **resize the encoding to a fixed size**. This makes it possible to feed images of variable size to the Encoder. (We did, however, resize our input images to `256, 256` because we had to store them together as a single tensor.)

Since we may want to fine-tune the Encoder, we add a `fine_tune()` method which enables or disables the calculation of gradients for the Encoder's parameters. We **only fine-tune convolutional blocks 2 through 4 in the ResNet**, because the first convolutional block would have usually learned something very fundamental to image processing, such as detecting lines, edges, curves, etc. We don't mess with the foundations.

### Attention

The Attention network is simple – it's composed of only linear layers and a couple of activations.

Separate linear layers **transform both the encoded image (flattened to `N, 14 * 14, 2048`) and the hidden state (output) from the Decoder to the same dimension**, viz. the Attention size. They are then added and ReLU activated. A third linear layer **transforms this result to a dimension of 1**, whereupon we **apply the softmax to generate the weights** `alpha`.

### Decoder

The output of the Encoder is received here and flattened to dimensions `N, 14 * 14, 2048`. This is just convenient and prevents having to reshape the tensor multiple times.

We **initialize the hidden and cell state of the LSTM** using the encoded image with the `init_hidden_state()` method, which uses two separate linear layers.

At the very outset, we **sort the `N` images and captions by decreasing caption lengths**. This is so that we can process only _valid_ timesteps, i.e., not process the `<pad>`s.

![](https://raw.githubusercontent.com/genigarus/a-PyTorch-Tutorial-to-Image-Captioning/master/img/sorted.jpg)

We can iterate over each timestep, processing only the colored regions, which are the **_effective_ batch size** `N_t` at that timestep. The sorting allows the top `N_t` at any timestep to align with the outputs from the previous step. At the third timestep, for example, we process only the top 5 images, using the top 5 outputs from the previous step.

This **iteration is performed _manually_ in a `for` loop** with a PyTorch [`LSTMCell`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM) instead of iterating automatically without a loop with a PyTorch [`LSTM`](https://pytorch.org/docs/master/nn.html#torch.nn.LSTM). This is because we need to execute the Attention mechanism between each decode step. An `LSTMCell` is a single timestep operation, whereas an `LSTM` would iterate over multiple timesteps continously and provide all outputs at once.

We **compute the weights and attention-weighted encoding** at each timestep with the Attention network. In section `4.2.1` of the paper, they recommend **passing the attention-weighted encoding through a filter or gate**. This gate is a sigmoid activated linear transform of the Decoder's previous hidden state. The authors state that this helps the Attention network put more emphasis on the objects in the image.

We **concatenate this filtered attention-weighted encoding with the embedding of the previous word** (`<start>` to begin), and run the `LSTMCell` to **generate the new hidden state (or output)**. A linear layer **transforms this new hidden state into scores for each word in the vocabulary**, which is stored.

We also store the weights returned by the Attention network at each timestep. You will see why soon enough.

## Training and Validation

See [code](https://github.com/SVGS-EVA4/Phase2/blob/master/S12-Image_Captioning_Text_to_Images/S12_ImageCaptioning.ipynb)

## Inference

**1) Input**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i6.jpg)

**1) Inference**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i1.PNG)


**2) Input**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i7.jpg)

**2) Inference**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i2.PNG)


**3) Input**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i8.jpg)

**3) Inference**

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S12-Image_Captioning_Text_to_Images/asset/i3.PNG)
