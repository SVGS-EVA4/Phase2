import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def imshow_images(img,c ):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(15,15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c, fontsize = 15)
    plt.axis('off')



def show_train_data(dataset, classes):

	# get some random training images

  dataiter = iter(dataset)
  images, labels = dataiter.next()
  for i in range(len(classes)):
    index = [j for j in range(len(labels)) if labels[j] == i]
    imshow_images(torchvision.utils.make_grid(images[index[0:5]],nrow=5,padding=20,scale_each=True, pad_value = 0.9),classes[i])


def show_misclassified_images(model, device, dataset, classes,number = 10, class_no = 0):
  misclassified_images = []
  
  for images, labels in dataset:
            images, labels = images.to(device), labels.to(device)
            outputs = model.to(device)(images)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
              if(len(misclassified_images)<number and predicted[i]!=labels[i] and labels[i] == class_no):
                misclassified_images.append([images[i],predicted[i],labels[i]])
            if(len(misclassified_images)>number):
              break
  return misclassified_images

# --------------------------------------Plot Misclassified Images-------------------------------------------------
    
def plot_misclassified_images(misclassified_images, classes, Figsize = (20,20),number = 25,class_no = 0) :

  fig = plt.figure(figsize = Figsize)
  
  for i in range(number):
        sub = fig.add_subplot(5, 5, i+1)
        img = misclassified_images[i][0].cpu()
        img = img *0.29  + 0.6 
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg,(1, 2, 0)),interpolation='none')

        sub.set_title("P={}".format(str(classes[misclassified_images[i][1].data.cpu().numpy()])), fontsize = 12)
        sub.axis('off')
        fig.suptitle(classes[class_no],fontweight="bold", fontsize=25,y=1.02)

        plt.tight_layout()
  plt.savefig(classes[class_no]+'.jpg',bbox_inches='tight')
