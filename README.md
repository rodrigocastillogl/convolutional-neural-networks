# Convolution Neural Networks

## Introduction

When dealing with tabular data (rows corresponding to observations and columns to features)
we anticipate that patterns, that we hope our deep learing model to learn, involve interactions
among features. However we do not assume any structure concerning how the features interact. 
In these cases, a Multilayer Perceptron (MLP) is a good idea.

But when our data are images (usually this involves very large input data) we can take advantage 
of the fact that they exhibit rich structure that can be exploited using 
*Convolutional Neural Networks*.

To detect objects in an image, we would like a model that recognizes them wherever they appear 
in the image. We can achive this considering the following characteristics for the model:

* *Translation invariance*: in earliest layers, the model should respond similarly to the same 
patch, regardless of where it appears in the image.
* *Locality principle*: in earliest layers, network should focus on local regions, without regard 
for the contents in distant regions. Later, these local representations can be aggregated.
* Deeper layers should be able to capture longer-range features of the image.

We can desing a model with these characteristics.

**From MLP**. We can consider a MLP with a two-dimensional image $\mathbf{X}$ as input an the hidden 
representation  also represented as a matrix ${\bf H}$, where ${\bf X}$ and $\mathbf{H}$ have the 
same shape. If $\mathbf{X}_{i,j}$ represents the value of the pixel at location $(i,j)$ of the matrix 
$\mathbf{X}$  and every pixel of the hidden matrix depends on every pixel of the input, then we 
compute the hidden representation as

$$\mathbf{H}{i,j} = \mathbf{U}{i,j} + \sum_{l} \sum_{k} \mathtt{W}{i,j,k,l} \ \mathbf{X}_{i,j}$$

where $\mathtt{W}$ is a fourth-order weight tensor and $\mathbf{U}$ is the bias matrix.

Now, we can re-index the subscripts $(k,l)$ such that $k=i+a$ and $l=j+b$, this way for a given 
position we compute the value $\mathbf{H}_{i,j}$ by summing over pixels in $\mathbf{X}$ weighted by 
the tensor $\mathtt{V}$ centered at $(i,j)$

$$\mathbf{H}{i,j} = \mathbf{U}{i,j} + \sum_{a} \sum_{b} \mathtt{W}{i,j,a,b} \ \mathbf{X}_{i+a,j+b}$$

Note that we can assume zeros around the image and if we have an input image of size $n_w \times n_h$, 
then $a \in (-n_w, n_w)$ and $b \in (-n_h, n_h)$. 

**Translation invariance**.  This property implies that the weights of the tensor $\mathtt{V}$ do not 
depend on the position $(i,j)$ where it is centered, not either the bias, this is

$$\mathbf{H}{i,j} = u + \sum_{a} \sum_{b} \mathbf{V}{a,b} \ \mathbf{X}_{i+a,j+b}$$

Now, we have the weights matrix $\mathbf{V}$ that is the same for every location $(i,j)$, and a constat $u$. 
Still, we have that $a \in (-n_w, n_w)$ and $b \in (-n_h, n_h)$.

**Locality**. Finally, adding locality, when we compute the value $\mathbf{H}_{i,j}$ we do not have to consider 
pixels in the input that are far away from the location $(i,j)$. 

This means that outside the range 
$|a|<\Delta$ , $|b|<\Delta$ we should set $\mathbf{V}_{a,b}=0$, or equivalently

$$\mathbf{H}{i,j} = u + \sum_{|a|<\Delta} \sum_{|b|<\Delta} \mathbf{V}{a,b} \mathbf{X}_{i+a,j+b}$$.

Strictly speaking, this equation correspond to the *cross-correlation*.

**Multiple channels**. 

## Convolutional Layer

In convolutional layers we apply cross-correlation between a kernel tensor $\mathtt{V}$ (weights or 
*learnable parameters*) and the multiple-channel image $\mathbf{X}$. This is

$$\mathtt{H}{i,j,d} = \sum_{a = - \Delta}^{\Delta} \sum_{b = - \Delta}^{\Delta} \sum_{c} \mathtt{V}{a,b,c} \mathtt{X}_{i+a,j+b}$$

In a convolution, we should first flip 
the kernel both horizontally and vertically, and then perform cross-correlation. But since the kernels are 
learned from the data, the output is not affected if the a simpler cross-correlation is used instead.




## Padding and Stride

## Pooling

## Fashion-MNIST example

## References
* [Dive into Deep Learning](https://d2l.ai/)
* [The Little Book of Deep Learning](https://fleuret.org/public/lbdl.pdf)
* [Deep Learning (Goodfellow, Bergio and Courville)](https://www.deeplearningbook.org/)
