#  <p align="center">MobileNets and ShuffleNets</p>

## Problem Statement:
<br/>

> Training MobileNet-V2(transfer learning) on a custom dataset with 4 classes - Small Copter, Large Copter, Winged Drones and Flying Birds.
<br/>
<br/>


|Links                 |                                                                                                   |
| ---------------------| --------------------------------------------------------------------------------------------------|
| Raw Dataset          | [click here](https://drive.google.com/file/d/1-EvvUU6K6RzNVgEibT3oP1SFb_epRNbI/view?usp=sharing)  |
| Preprocessed DataSet |[click here](https://drive.google.com/file/d/1sJ8EngUpwcTT7tbqRhqQijbm-nGLuVF9/view?usp=sharing)   |
| Labels               |[click here](https://drive.google.com/file/d/1-5KNd0rNceRdtxWqvlG_3w9VnY37Bkc5/view?usp=sharing)   |
| Final Model          |[click here](https://drive.google.com/file/d/1ZgRsnAXnrx4vruxRbboSqwMOVl0rL_MC/view?usp=sharing)   |


<b>Insomia Output:</b>

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/insomia_output.JPG'/>

## Results:
* Epochs: 20 
* Best Train Accuracy: 89.17%
* Best Test Accuracy: 89.28%
* Classwise Accuracies:
   * Winged_Drones : 90.29%
   * Small_QuadCopters : 76.33%
   * Large_QuadCopters : 80.0%
   * Flying_Birds : 98.93%
   
## Dataset 

### Data Statistics

<img src= "https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S2-MobileNets_and_ShuffleNets/images/dataset_viz1.PNG" width = "400">

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

 <p align="center"><img src = "https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/input_output_pad.jpg"></p>
 

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

<p align="center"><img src= "https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/plots.jpg"></p>

## Misclassified Images: 

<p align="center"><img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/misclassified_classwise.png' height = "1600" /></p>

## Observations:

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S2-MobileNets_and_ShuffleNets/images/conf_matrix.PNG)

* By looking at the misclassified images we can observe that there are lot of misclassification between large quadcopters and small quadcopters. Model is confused between the two, as there is not much difference between them.
* Some are correctly classified, but the actual labels themselves are incorrect.
* Flying birds are mostly classified as winged drones, biased by the wings of birds.

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S2-MobileNets_and_ShuffleNets/images/report.PNG)

* In the above image, 0 denotes **Winged Drone**, 1 denotes **Small Quadcopters**, 2 denotes **Large Quadcopters** and 3 denotes **Flying Birds**. From the above image, we can see that:

1) **Flying Birds** have the perfect precision and recall implying that almost all flying bird images are correctly classified.

2) **Small and Large Quadcopters** seem to have the least precision and recall.

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S2-MobileNets_and_ShuffleNets/images/confusion_matrix.PNG)

* In the above image, we can see the reasons for above mentioned observation. Many of the images in Large Quadcopters are wrongly classified as Winged Drones and Small Quadcopters. If we delve further into this by going into the dataset for inspecting images, we would find many winged drones and small quadcopters images being wrongly labelled by humans as large quadcopters. Similar is the case with small quadcopters.

