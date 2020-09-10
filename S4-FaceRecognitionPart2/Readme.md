# Face Recognition Part 2
[![Website](https://img.shields.io/badge/Website-blue.svg)](https://svgs-eva.s3.ap-south-1.amazonaws.com/face_recognition.html)

# Assignment
1. Refer to this beautiful [blog](https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79) 
2. Collect 10 facial images of 10 people you know (stars, politicians, etc). The more the images you collect, the better your experience would be. Add it to this [LFW](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) dataset. 
3. Train as in the blog and upload the FR model to Lambda


### **We have trained on following 10 people:**
1. APJ Abdul Kalam
2. Barack Obama
3. Michelle Obama
4. Monica Gellar
5. Ross Gellar
6. Phoebe Buffay
7. Rachel Green
8. Chandler Bing
9. Joey Tribiyanni
10. Elon Musk

### **Training**

1. Initially we trained on the LFW + our 10 classes(total ~5k classes) but, but training accuracy was 100% and test accuracy was 50%. Since many classes have very less images(1,2 etc)
2. Then we removed the images which have less than 9 classes and trained. The training acc was 100% and test acc was 97%.

### **Parameters and Hyperparameters**

- Training Accuracy: 100%
- Test Accuaracy : 97.4%
- Loss Function: Cross Entropy Loss
- Epochs: 20
- Optimizer: SGD
- Learning Rate: 0.01
- Batch Size: 128
- Scheduler: StepLR

<img src='https://github.com/SVGS-EVA4/Phase2/blob/master/S4-FaceRecognitionPart2/images/face_recognition.png' alt='Face Recognition'/>


