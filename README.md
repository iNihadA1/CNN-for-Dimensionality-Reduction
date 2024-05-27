# Convolutional Autoencoder with MNIST Dataset
This project demonstrates the use of a convolutional autoencoder to compress and reconstruct images from the MNIST dataset. Additionally, it explores dimensionality reduction and clustering techniques on the encoded representations of the images.

# Table of Contents
- [Introduction](###introduction)
- [Dataset](###Dataset)
- [Model Architecture](###Model-Architecture)
- [Training](###Training)
- [Dimensionality Reduction and Clustering](###Dimensionality-Reduction-and-Clustering)
- [Results](###Results)
- [Requirements](###Requirements)
- [Usage](###Usage)
- [License](###License)

### Introduction
Autoencoders are a type of neural network used to learn efficient codings of input data. This project implements a convolutional autoencoder to encode and decode images from the MNIST dataset. The encoded representations are then analyzed using dimensionality reduction techniques (PCA, t-SNE, UMAP) and clustering algorithms (KMeans).

### Dataset
The MNIST dataset contains 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels. The images are grayscale and normalized to the range [0, 1].

### Model Architecture
The convolutional autoencoder consists of two parts:

1. Encoder: Compresses the input image into a lower-dimensional representation.
2. Decoder: Reconstructs the original image from the encoded representation.

#### Encoder
* Input layer: 28x28x1
* Conv2D: 32 filters, (3x3) kernel, ReLU activation, same padding
* MaxPooling2D: (2x2) pool size, same padding
* Conv2D: 32 filters, (3x3) kernel, ReLU activation, same padding
* MaxPooling2D: (2x2) pool size, same padding

#### Decoder
* Conv2D: 32 filters, (3x3) kernel, ReLU activation, same padding
* UpSampling2D: (2x2) size
* Conv2D: 32 filters, (3x3) kernel, ReLU activation, same padding
* UpSampling2D: (2x2) size
* Conv2D: 1 filter, (3x3) kernel, Sigmoid activation, same padding

### Training
The autoencoder is compiled using the Adam optimizer and binary cross-entropy loss. It is trained for 20 epochs with a batch size of 256.

### Dimensionality Reduction and Clustering

The encoded representations of the images are further analyzed using:
* **PCA** (Principal Component Analysis)
* **t-SNE** (t-distributed Stochastic Neighbor Embedding)
* **UMAP** (Uniform Manifold Approximation and Projection)

The representations are clustered using the KMeans algorithm, and the clustering performance is evaluated using:
* **Silhouette Score**
* **Adjusted Rand Index (ARI)**
* **Normalized Mutual Information (NMI)**

### Results
The results include visualizations of the reconstructed images and plots of the encoded representations in reduced dimensions. Clustering performance metrics are also reported.

### Requirements
* Python 3.x
* numpy
* matplotlib
* keras
* sklearn
* umap-learn

Install the required packages using:
```bash 
pip install numpy matplotlib keras scikit-learn umap-learn
```
### Usage
1. Clone the repository:
    ```bash 
    git clone https://github.com/yourusername/autoencoder-mnist.git cd autoencoder-mnist 
    ``` 
2. Run the script:
    ```bash 
    python autoencoder_mnist.py
    ``` 
### License

This project is licensed under the MIT License. See the LICENSE file for details.    
____

