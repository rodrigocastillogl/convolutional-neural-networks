# Convolution Neural Networks

## Introduction

When dealing with tabular data (rows corresponding to observations and columns to features)
we anticipate that patterns, that we hope our deep learing model to learn, involve interactions
among features. However we do not assume any structure concerning how the features interact. 
In these cases, a Multilayer Perceptron (MLP) is good idea.

But when our data are images (usually this involves very large input data) we can take advantage 
of the fact that they exhibit rich structure that can be exploited using 
*Convolutional Neural Networks*.

## Convolution

To detect objects in an image, we would like model that recognizes them wherever they appear in 
the image. We can achive this considering the following characteristics for the model:

* *Translation invariance*: in earliest layers, the model should respond similarly to the same 
patch, regardless of where it appears in the image.
* *Locality principle*: in earliest layers, network should focus on local regions , without regard 
for the contents in distant regions. Later, these local representations can be aggragated.
* Deeper layers should be able to capture longer-range features of the image.

We can now desing a model with these characteristics. We can consider a MLP with two-dimensional 
images ${\bf X}$ as inputs an the hidden representation also represented as matrices ${\bf H}$, 
where ${\bf X}$ and ${\bf H}$ have the same shape.

If ${\bf X}_{i,j}$ represents the value of the pixel at location $(i,j)$ of the matrix ${\bf X}$
and every pixel of the hidden matrx depends on every pixel of the input, the we compute the 
hidden representation as

$${\bf H}_{i,j} = {\bf U}_{i,j} + \sum_{k} \sum_{l} \mathtt{W}_{i,j,k,l} {\bf X}_{k,l}$$

where $\mathtt{W}$ is a fourth-order weight tensor and ${\bf U}$ is the bias matrix.

## Convolutional Layer

## Padding and Stride

## Pooling

## Fashion-MNIST example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf)
* [Deep Learning (Goodfellow, Bergio and Courville)](https://www.deeplearningbook.org/)
