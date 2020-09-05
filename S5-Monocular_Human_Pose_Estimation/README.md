# **Simple Baselines for Human Pose Estimation and Tracking**

[![Paper](https://img.shields.io/badge/Paper-blueviolet.svg)](https://arxiv.org/pdf/1804.06208.pdf)


1. Human pose estimation is the process of estimating the configuration of the body (pose) from a single, typically monocular, image. 

2. Pose Tracking is the task of estimating multi-person human poses in videos and assigning unique instance IDs for each keypoint across frames.

3. There has been significant progress on pose estimation and
increasing interests on pose tracking in recent years. At the same time,
the overall algorithm and system complexity increases as well, making
the algorithm analysis and comparison more difficult. This work provides
simple and effective baseline methods

**This work aims to ease complexity in algorithm and architetural problem by asking question *how good could a simple method be?***

This approach involves a few deconvolutional layers added on a backbone network, ResNet.
This approach adds a few deconvolutional layers over the last convolution stage in the ResNet, called **C5**

## **Pose Estimation Using A Deconvolution Head Network**

1. ResNet is the most common backbone network for image feature extraction.
This method simply adds a few deconvolutional layers over the last convolution stage in the ResNet, called C5.They adopted this structure because it is arguably the simplest to generate heatmaps from deep and low resolution features and also adopted in the state-of-the-art Mask R-CNN

2. By default, three deconvolutional layers with batch normalization and
ReLU activation are used. Each layer has 256 filters with 4 × 4 kernel. The
stride is 2. A 1 × 1 convolutional layer is added at last to generate predicted
heatmaps {H1 . . . Hk} for all k key points.

![architecture](https://github.com/SVGS-EVA4/Phase2/blob/master/S5-Monocular_Human_Pose_Estimation/images/simple_pose.png)

3. Comparing the SOTA network architecture with the implemented architechture it differs in n how high resolution feature maps are generated. Both works use upsampling to increase the feature map resolution and put convolutional parameters in other blocks. In contrary, this method combines the upsampling and convolutional parameters into deconvolutional layers in a much simpler way, without using skip layer connections.

4. A commonality of the three methods is that three upsampling steps and also
three levels of non-linearity (from the deepest feature) are used to obtain highresolution feature maps and heatmaps. 

## **Pose Tracking Based on Optical Flow**
![Pose tractking](https://github.com/SVGS-EVA4/Phase2/blob/master/S5-Monocular_Human_Pose_Estimation/images/tracking.png)

1. Multi-person pose tracking in videos first estimates human poses in frames, and
then tracks these human pose by assigning a unique identification number (id)
to them across frames.
2. Ik : kth Frame

P : Human instance P = (J, id), where J = {ji}1:Nj is the coordinates set of Nj body joints and id indicates the tracking id.

When processing the Ik frame, we have the already processed human instances set Pk-1 = {Pik-1}1:Nk-1 in frame Ik-1 and the instances set Pk = {Pik}1:Nk in frame Ik whose id is to be assigned, where Nk-1 and Nk are the instance number in frame Ik-1 and Ik. If one instance Pkj in current frame Ik is linked to the instance Pik-1 in Ik-1 frame, then idk-1i is propagated to idkj, otherwise a new id is assigned to Pkj , indicating a new track.

## **Joint Propagation using Optical Flow**
1. Simply applying a detector designed for single image level to videos could lead to missing detections and false detections due to motion blur and occlusion introduced by video frames.
2. They proposed to generate boxes for the processing frame from nearby frames
using temporal information expressed in optical flow.

3. Given one human instance with joints coordinates set J
k−1
i
in frame I
k−1
and the optical flow field Fk−1→k between frame I
k−1 and I
k
, they could estimate
the corresponding joints coordinates set ˆJ
k
i
in frame I
k by propagating the
joints coordinates set J
k−1
i
according to Fk−1→k. More specifically, for each joint
location (x, y) in J
k−1
i
, the propagated joint location would be (x + δx, y + δy),
where δx, δy are the flow field values at joint location (x, y). Then they computed a
bounding of the propagated joints coordinates set ˆJ
k
i
, and expanded that box by
some extend (15% in experiments) as the candidated box for pose estimation.


## **Flow-based Pose Similarity**

1. Using bounding box IoU(Intersection-over-Union) as the similarity metric (SBbox)
to link instances could be problematic when an instance moves fast thus the
boxes do not overlap, and in crowed scenes where boxes may not have the corresponding relationship with instances

2. Using pose similarity could also be problematic when the pose of the same person is different across frames due to pose changing.

3. So they proposed to use a **Object Keypoint Similarity(OKS)** which is flow based similarity Index
<br/>

![formula](https://github.com/SVGS-EVA4/Phase2/blob/master/S5-Monocular_Human_Pose_Estimation/images/formula.png)

## **Flow-based Pose Tracking Algorithm**

1. For the processing frame in videos, the boxes from a human detector and boxes generated by propagating joints from previous frames using optical flow are unified using a bounding box Non-Maximum Suppression (NMS) operation. The boxes generated by progagating joints serve as the complement of missing detections of the detector. Then then estimate human pose using the cropped and resized images by these boxes through our proposed pose estimation network

![Algorithm](https://github.com/SVGS-EVA4/Phase2/blob/master/S5-Monocular_Human_Pose_Estimation/images/algorithm.png)

2. They store the tracked instances in a double-ended queue(Deque) with fixed length LQ, denoted as Q = [Pk−1,Pk−2, ...,Pk−LQ ]where Pk−i means tracked instances set in previous frame Ik−i and the Q’s length LQ indicates how many previous frames considered when performing matching. The Q could be used to capture previous multi frames’ linking relationship, initialized in the first frame in a video. For the kth frame Ik, they calculate the flow-based pose similarity matrix Msim between the untracked instances set of body joints Jk (id is none) and previous instances sets in Q . Then they assign id to each body joints instance J in Jk to get assigned instance set Pk by using greedy matching and Msim. Finally they update the tracked instances Q by adding up kth frame instances set Pk



## **JointMSELoss**

Mean Squared Error (MSE) is used as the loss between
the predicted heatmaps and targeted heatmaps. The targeted heatmap Hˆk for
joint k is generated by applying a 2D gaussian centered on the kth joint’s ground truth location. 

Since it is calulated for joints it is called as **JointMSELoss**


```python
import torch.nn as nn


class JointsMSELoss(nn.Module):

    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
```
