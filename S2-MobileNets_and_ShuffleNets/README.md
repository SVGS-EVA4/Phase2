# Mobilenet

Link to dataset: <a href='https://drive.google.com/file/d/1-EvvUU6K6RzNVgEibT3oP1SFb_epRNbI/view?usp=sharing'>dataset.zip</a>

Link to the modified dataset and labels : <a href='https://drive.google.com/file/d/1sJ8EngUpwcTT7tbqRhqQijbm-nGLuVF9/view?usp=sharing'>dataset_padded.zip</a> , <a href='https://drive.google.com/file/d/1-5KNd0rNceRdtxWqvlG_3w9VnY37Bkc5/view?usp=sharing'>Labels </a>

Website: <a href='https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification'>https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification</a>

Insomia Output:

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/insomia_output.JPG'/>

## Stats:
* Best Train Accuracy: 89.17%
* Best Test Accuracy: 89.28%
* Epochs: 20 

## Resizing Strategy:
* The dataset consisted of non-square images of different sizes.
* Following is the code:
```

img_file_path = '/content/dataset/Small_QuadCopters/Small_QuadCopters_1.jpg'
img = Image.open(img_file_path).convert('RGB')

h,w = img.size[0],img.size[1]
max_len = max(h,w)
if h == w:
    return img
    
elif h>w:
    diff = int(abs(h-w)/2)
    black = np.zeros((max_len,max_len))
    black_img = Image.fromarray(black,mode='RGB')

    black_img.paste(img,(0,diff))
    return black_img
elif w>h:
    diff = int(abs(h-w)/2)
    black = np.zeros((max_len,max_len))
    black_img = Image.fromarray(black,mode='RGB')

    black_img.paste(img,(diff,0))

    return black_img

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
```
(classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True) 
  )
)
```
to
```
(classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Sequential(
      (0): Linear(in_features=1280, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=4, bias=True)
    )
```

## Plots:

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/plot_acc.png'/>

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/plot_loss.png'/>

## Misclassified Images: 

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S2-MobileNets_and_ShuffleNets/images/misclassified_classwise.png'/>
