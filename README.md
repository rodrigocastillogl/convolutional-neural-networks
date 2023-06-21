# Convolution Neural Networks

## Introduction

When dealing with tabular data (rows corresponding to observations and columns to features) we anticipate 
that patterns, that we hope our deep learing model to learn, involve interactions among features. However 
we do not assume any structure concerning how the features interact. In these cases, a Multilayer Perceptron 
(MLP) is a good idea.

But when our data are images (usually this involves very large input data) we can take advantage of the fact 
that they exhibit rich structure that can be exploited using *Convolutional Neural Networks* (CNNs).

To detect objects in an image, we would like a model that recognizes them wherever they appear in the image. 
We can achive this considering the following characteristics for the model:

* *Translation invariance*: in earliest layers, the model should respond similarly to the same 
patch, regardless of where it appears in the image.
* *Locality principle*: in earliest layers, network should focus on local regions, without regard 
for the contents in distant regions. Later, these local representations can be aggregated.
* Deeper layers should be able to capture longer-range features of the image.

We can desing a model with these characteristics.

**From MLP**. We can consider a MLP with a two-dimensional image $\mathbf{X}$ as input an the hidden 
representation also represented as a matrix $\mathbf{H}$, where $\mathbf{X}$ and $\mathbf{H}$ have the same shape. 
If $\mathbf{X}_{i,j}$ represents the value of the pixel at location $(i,j)$ of the matrix $\mathbf{X}$ and every 
pixel of the hidden matrix depends on every pixel of the input, then we compute the hidden representation as

```math
\mathbf{H}_{i,j} = \mathbf{U}_{i,j} + \sum_{l} \sum_{k} \mathtt{W}_{i,j,k,l} \ \mathbf{X}_{i,j}
```

where $\mathtt{W}$ is a fourth-order weight tensor and $\mathbf{U}$ is the bias matrix.

Now, we can re-index the subscripts $(k,l)$ such that $k=i+a$ and $l=j+b$, this way for a given position we 
compute the value $\mathbf{H}_{i,j}$ by summing over pixels in $\mathbf{X}$ weighted by the tensor $\mathtt{V}$ 
centered at $(i,j)$

```math
\mathbf{H}_{i,j} = \mathbf{U}_{i,j} + \sum_{a} \sum_{b} \mathtt{W}_{i,j,a,b} \ \mathbf{X}_{i+a,j+b}
```

Note that we can assume zeros around the image and if we have an input image of size $n_w \times n_h$, then 
$a \in (-n_w, n_w)$ and $b \in (-n_h, n_h)$. 

**Translation invariance**.  This property implies that the weights of the tensor $\mathtt{V}$ do not 
depend on the position $(i,j)$ where it is centered, not either the bias, this is

```math
\mathbf{H}_{i,j} = u + \sum_{a} \sum_{b} \mathbf{V}_{a,b} \ \mathbf{X}_{i+a,j+b}
```
Now, we have the weights matrix $\mathbf{V}$ that is the same for every location $(i,j)$, and a constat $u$. 
Still, we have that $a \in (-n_w, n_w)$ and $b \in (-n_h, n_h)$.

**Locality**. Finally, adding locality, when we compute the value $\mathbf{H}_ {i,j}$ we do not have to consider 
pixels in the input that are far away from the location $(i,j)$. This means that outside the range $|a|<\Delta$, 
$|b|<\Delta$ we should set $\mathbf{V}_ {a,b}=0$, or equivalently

```math
\mathbf{H}_{i,j} = u + \sum_{|a|<\Delta} \sum_{|b|<\Delta} \mathbf{V}_{a,b} \mathbf{X}_{i+a,j+b}
```

**Multiple channels**. Until now we have not considered that images consists of multiple channels (grayscale images 
have only 1 channel, but typically RGB have 3 channels). Then, an image is a third-order tensor $\mathtt{X}$ 
characterized by height, width and channel. For a hidden representation we use a third-order tensor $\mathtt{H}$.

To support multiple channels, the weighted sum is also computed along the channels. In convolutional layers we apply 
*cross-correlation* between a kernel tensor $\mathtt{V}$ (weights or *learnable parameters*) and the multiple channels 
input image $\mathtt{X}$. This is

```math
\mathtt{H}_{i,j,k} = u_k + \sum_{a = - \Delta}^{\Delta} \sum_{b = - \Delta}^{\Delta} \sum_{c} \mathtt{V}_{a,b,c,k} \mathtt{X}_{i+a,j+b, c}
```

Strictly speaking, in a convolution, we should first flip the kernel, and then perform cross-correlation. But 
since the kernels are learned from the data, the output is not affected if the a simpler cross-correlation is 
used instead.

In sequential CNNs we use several consecutive convolutional layers. In earliest layers we usually interpretate 
that the learned parameters correspond to feature extractor kernels (edges, cornes, etc.). But as we go deeper 
in the CNN it is not clear what is each filter doing.

The output of a convolutional layer is sometimes called *feature map* because it can be seen as the learned 
representation in the espatial dimensions. In CNNs, for any given element $x^{\*}$ of the output of some layer, its 
receptive field refers to all the elements of the input $\mathtt{X}$ that may affect the calculation of $x^{\*}$. 

When any element in a feature map needs a larger receptive field, we can build a deeper network. This way a 
CNN can capture longer-range features of the image.

# Convolutional Layer

We have seen the intuition and motivation of CNNs, from the perspective of digital image processing. Now 
we will define 1D convolution and 2D convolution.

A **1D convolution layer** is mainly defined by three meta-parameters: its kernel size $K$, its number of 
input channels $C_ {in}$, its number of output channels $C_ {out}$. It takes a tensor $\mathtt{X}$ of size 
$C_ {in} \times T$ as input and uses a kernel $\mathtt{V}$ of size $C_ {out} \times K$ to compute the 
output $\mathtt{H}$ as follows

```math
\mathtt{H}_{i,j} = u_i + \sum_{a=-K/2}^{K/2} \sum_{c=1}^{C_{out}} \mathtt{V}_{i,c,a} \mathtt{X}_{i+c,j+a}
```

It takes an input batch $\mathtt{X}$ of size $N \times D_ {in} \times T$ as input and uses a kernel 
$\mathtt{V}$ of size 


```math
\mathtt{H}_{i,j} = u + \sum_{a} \sum_\mathtt{V}_{i,a,b}
```

The resulting size of $\mathtt{H}$ is $ C_ {out} \times (T−K+1)$.


## Padding and Stride

The *padding* specifies how many zero coefficients should be added around the input tensor before processing it, 
particularly to maintain the tensor size when the kernel size is greater than one. Its default value is 0.

The *stride* specifies the step used when going through the input, allowing one to reduce the output size geometrically 
by using large steps. Its default value is 1.

## Pooling

## Fashion-MNIST example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf)
* [Deep Learning (Goodfellow, Bergio and Courville)](https://www.deeplearningbook.org/)
