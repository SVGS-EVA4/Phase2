# Convolutional Sentiment Analysis
---

[![](https://img.shields.io/badge/Website-green.svg)](http://svgs-eva4.s3-website.ap-south-1.amazonaws.com/sentimental_analysis.html)   [![](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/SVGS-EVA4/Phase2/blob/master/S9-Neural%20Embedding/ConvolutionalSentimentAnalysis.ipynb) 

Traditionally, CNNs are used to analyse images and are made up of one or more convolutional layers, followed by one or more linear layers. The convolutional layers use filters (also called kernels or receptive fields) which scan across an image and produce a processed version of the image. This processed version of the image can be fed into another convolutional layer or a linear layer. Each filter has a shape, e.g. a 3x3 filter covers a 3 pixel wide and 3 pixel high area of the image, and each element of the filter has a weight associated with it, the 3x3 filter would have 9 weights. In traditional image processing these weights were specified by hand by engineers, however the main advantage of the convolutional layers in neural networks is that these weights are learned via backpropagation.

The intuitive idea behind learning the weights is that your convolutional layers act like feature extractors, extracting parts of the image that are most important for your CNN's goal, e.g. if using a CNN to detect faces in an image, the CNN may be looking for features such as the existance of a nose, mouth or a pair of eyes in the image.

So why use CNNs on text? In the same way that a 3x3 filter can look over a patch of an image, a 1x2 filter can look over a 2 sequential words in a piece of text, i.e. a bi-gram. In the previous tutorial we looked at the FastText model which used bi-grams by explicitly adding them to the end of a text, in this CNN model we will instead use multiple filters of different sizes which will look at the bi-grams (a 1x2 filter), tri-grams (a 1x3 filter) and/or n-grams (a 1x$n$ filter) within the text.

The intuition here is that the appearance of certain bi-grams, tri-grams and n-grams within the review will be a good indication of the final sentiment.

# Model Building
---
The first major hurdle is visualizing how CNNs are used for text. Images are typically 2 dimensional (we'll ignore the fact that there is a third "colour" dimension for now) whereas text is 1 dimensional. However, we know that the first step in almost all of our previous tutorials (and pretty much all NLP pipelines) is converting the words into word embeddings. This is how we can visualize our words in 2 dimensions, each word along one axis and the elements of vectors aross the other dimension. Consider the 2 dimensional representation of the embedded sentence below:

![](https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/master/assets/sentiment9.png)

We can then use a filter that is **[n x emb_dim]**. This will cover $n$ sequential words entirely, as their width will be `emb_dim` dimensions. Consider the image below, with our word vectors are represented in green. Here we have 4 words with 5 dimensional embeddings, creating a [4x5] "image" tensor. A filter that covers two words at a time (i.e. bi-grams) will be **[2x5]** filter, shown in yellow, and each element of the filter with have a _weight_ associated with it. The output of this filter (shown in red) will be a single real number that is the weighted sum of all elements covered by the filter.

![](https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/master/assets/sentiment12.png)

The filter then moves "down" the image (or across the sentence) to cover the next bi-gram and another output (weighted sum) is calculated. 

![](https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/master/assets/sentiment13.png)

Finally, the filter moves down again and the final output for this filter is calculated.

![](https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/master/assets/sentiment14.png)

In our case (and in the general case where the width of the filter equals the width of the "image"), our output will be a vector with number of elements equal to the height of the image (or lenth of the word) minus the height of the filter plus one, $4-2+1=3$ in this case.

This example showed how to calculate the output of one filter. Our model (and pretty much all CNNs) will have lots of these filters. The idea is that each filter will learn a different feature to extract. In the above example, we are hoping each of the **[2 x emb_dim]** filters will be looking for the occurence of different bi-grams. 

In our model, we will also have different sizes of filters, heights of 3, 4 and 5, with 100 of each of them. The intuition is that we will be looking for the occurence of different tri-grams, 4-grams and 5-grams that are relevant for analysing sentiment of movie reviews.

The next step in our model is to use *pooling* (specifically *max pooling*) on the output of the convolutional layers. This is similar to the FastText model where we performed the average over each of the word vectors, implemented by the `F.avg_pool2d` function, however instead of taking the average over a dimension, we are taking the maximum value over a dimension. Below an example of taking the maximum value (0.9) from the output of the convolutional layer on the example sentence (not shown is the activation function applied to the output of the convolutions).

![](https://raw.githubusercontent.com/bentrevett/pytorch-sentiment-analysis/master/assets/sentiment15.png)

The idea here is that the maximum value is the "most important" feature for determining the sentiment of the review, which corresponds to the "most important" n-gram within the review. How do we know what the "most important" n-gram is? Luckily, we don't have to! Through backpropagation, the weights of the filters are changed so that whenever certain n-grams that are highly indicative of the sentiment are seen, the output of the filter is a "high" value. This "high" value then passes through the max pooling layer if it is the maximum value in the output. 

As our model has 100 filters of 3 different sizes, that means we have 300 different n-grams the model thinks are important. We concatenate these together into a single vector and pass them through a linear layer to predict the sentiment. We can think of the weights of this linear layer as "weighting up the evidence" from each of the 300 n-grams and making a final decision.

# Model Summary
---

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S9-Neural%20Embedding/asset/summary.PNG)

# Model Training
---

[Link to code](https://github.com/SVGS-EVA4/Phase2/blob/master/S9-Neural%20Embedding/ConvolutionalSentimentAnalysis.ipynb)

[Link to trained model](https://drive.google.com/file/d/1FlJyvphI4de1zBqwfEFJzCSFpArmWCDc/view?usp=sharing)

![](https://raw.githubusercontent.com/SVGS-EVA4/Phase2/master/S9-Neural%20Embedding/asset/TrainingHistory.PNG)

# Model Evaluation
---

Below is the result for evaluating on test data:

Test Loss: 0.351 | Test Acc: 85.66%

