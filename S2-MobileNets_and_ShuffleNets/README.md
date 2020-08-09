# MobileNets and ShuffleNets

## Problem Statement:
Training MobileNet-V2(transfer learning) on a custom dataset with 4 classes - Small Copter, Large Copter, Winged Drones and Flying Birds.

<b>Link to dataset:</b> <a href='https://drive.google.com/file/d/1-EvvUU6K6RzNVgEibT3oP1SFb_epRNbI/view?usp=sharing'>dataset.zip</a>

<b>Link to the modified dataset and labels:</b> <a href='https://drive.google.com/file/d/1sJ8EngUpwcTT7tbqRhqQijbm-nGLuVF9/view?usp=sharing'>dataset_padded.zip</a> , <a href='https://drive.google.com/file/d/1-5KNd0rNceRdtxWqvlG_3w9VnY37Bkc5/view?usp=sharing'>Labels </a>

<b>Website:</b> <a href='https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification'>https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification</a>

<b>Insomia Output:</b>

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/insomia_output.JPG'/>

## Result:
* Best Train Accuracy: 89.17%
* Best Test Accuracy: 89.28%
* Epochs: 20 

## Dataset

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S2-MobileNets_and_ShuffleNets/images/dataset_viz1.PNG)

### Resizing Strategy:
* The dataset consisted of non-square images of different sizes.
* So we converted all the images to square images by padding them.
* Later during training we applied the resize augmentation technique of albumentations library to convert all the images to 224x224.
* Approach for padding the images:
    * The padding is performed by overlaying the image at the center of a black image. 
    * For an image which has a larger width compared to it's height, we padded the image along the width of the image.
    * For an image which has a larger height compared to it's width, we padded the image along the height of the image.    
* Following is the code:
   ```python

   img_file_path = '/content/dataset/Small_QuadCopters/Small_QuadCopters_1.jpg'
   img = Image.open(img_file_path).convert('RGB')
   h,w = img.size[0],img.size[1]
   max_len = max(h,w)

   if h == w:
       black_img.save(f'/content/Dataset/{img_save_path}')

   elif h>w:
       diff = int(abs(h-w)/2)
       black = np.zeros((max_len,max_len))
       black_img = Image.fromarray(black,mode='RGB')
       black_img.paste(img,(0,diff))
       black_img.save(f'/content/Dataset/{img_save_path}')

   elif w>h:
       diff = int(abs(h-w)/2)
       black = np.zeros((max_len,max_len))
       black_img = Image.fromarray(black,mode='RGB')
       black_img.paste(img,(diff,0))
       black_img.save(f'/content/Dataset/{img_save_path}')
   ```

* Results:

  Input:

  <img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/pad_input.png'/>

  Output: 

  <img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/pad_output.png'/>
 

## Model Architecture:
* We applied transfer learning on the pretrained Mobilenet v2 model.
* Since the model was trained on Imagenet dataset, the output of model comprised of 1000 classes.
* So we altered the classifier layer of the model to produce an output of 4 classes.Following is the code:

    The original layers:
    ```python
    (classifier): Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=1280, out_features=1000, bias=True) 
      )
    )
    ```
    Modification:
    ```python
    (classifier): Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Sequential(
          (0): Linear(in_features=1280, out_features=256, bias=True)
          (1): ReLU()
          (2): Linear(in_features=256, out_features=4, bias=True)
        )
    ```
* We then trained the entire model without freezing any of the layers.

## Plots:

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/plot_acc.png'/>

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/plot_loss.png'/>

## Misclassified Images: 

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/misclassified_classwise.png'/>
