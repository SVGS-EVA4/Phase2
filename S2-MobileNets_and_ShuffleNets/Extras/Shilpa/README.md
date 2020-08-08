Link to dataset: <a href='https://drive.google.com/file/d/1-EvvUU6K6RzNVgEibT3oP1SFb_epRNbI/view?usp=sharing'>dataset.zip</a>

Link to the modified dataset and labels : <a href='https://drive.google.com/file/d/1sJ8EngUpwcTT7tbqRhqQijbm-nGLuVF9/view?usp=sharing'>dataset_padded.zip</a> , <a href='https://drive.google.com/file/d/1-5KNd0rNceRdtxWqvlG_3w9VnY37Bkc5/view?usp=sharing'>Labels </a>

Website: <a href='https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification'>https://865fgqaq94.execute-api.ap-south-1.amazonaws.com/dev/classification</a>
Insomia Output:
<img src=''/>

Stats:
* Best Train Accuracy: %
* Best Test Accuracy: %
* Epochs: 20 

Resizing Strategy:
* The dataset consisted of non-square images of different sizes.
* Following is the code:
```
def toSquare_img(img_bytes):
    print('padding the image')
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
  
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


Model Architecture:
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

```

Misclassified Images: 

Plots:
