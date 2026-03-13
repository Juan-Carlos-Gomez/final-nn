# Final project: neural network

## Overview

In this assignment, you will implement a neural network class from (almost) scratch. You will then apply your class to create both:

**(1) a simple 64x16x64 autoencoder.**

**(2) a classifier for transcription factor binding sites.**

You will begin by finishing the API for generating fully connected neural networks from scratch. You will then make Jupyter Notebooks where you create, train, and test your autoencoder and classifier.

## Step 1: finish the neural network API

### For steps 2 and 3

* [x] Finish all methods with a `pass` statement in the `NeuralNetwork` class in the `nn.py` file.

### For step 3

* [x] Finish the `sample_seqs` function in the `preprocess.py` file.
* [x] Finish the `one_hot_encode_seqs` function in the `preprocess.py` file.

## Datasets

### Digits Dataset (Step 2)
- **Source:** scikit-learn's built-in `load_digits()` dataset
- **Type:** Handwritten digit images (0-9)
- **Format:** 8×8 pixel grayscale images (64 features per image)
- **Total Samples:** 1,797 images
- **Purpose:** Unsupervised learning - testing autoencoder reconstruction capability
- **Data Range:** Pixel intensity values [0-16]

### Rap1 Binding Sites Dataset (Step 3)
- **Source:** Biological sequences from yeast (*Saccharomyces cerevisiae*)
- **Type:** DNA sequences (nucleotides: A, T, C, G)
- **Positive Examples:** 137 known Rap1 transcription factor binding motifs
  - File: `data/rap1-lieb-positives.txt`
  - Sequence length: 17 base pairs (bp)
- **Negative Examples:** 3,163 random upstream sequences from yeast genes
  - File: `data/yeast-upstream-1k-negative.fa` (FASTA format)
  - Sequence length: 1,000 bp (processed into 17bp subsequences)
- **Encoding:** One-hot encoding (A=[1,0,0,0], T=[0,1,0,0], C=[0,0,1,0], G=[0,0,0,1])
- **Features after encoding:** 68 (17 bp × 4 nucleotides)
- **Imbalance:** Severe class imbalance (137 pos vs 3,163 neg) - addressed via downsampling

## Step 2: make your autoencoder

### Background

An autoencoder is a neural network that takes an input, encodes it into a lower-dimensional latent space through "encoding" layers, and then attempts to reconstruct the original input using "decoding" layers. Autoencoders are often used for dimensionality reduction.

### Your task

You will train a 64x16x64 autoencoder on the [digits](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset) dataset. All of the following work should be done in a Jupyter Notebook.

### To-do

* [x] Load the digits dataset through sklearn using <code><a href="https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html">sklearn.datasets.load_digits()</a></code>.
* [x] Split the data into training and validation sets.
* [x] Generate an instance of your `NeuralNetwork` class with a 64x16x64 autoencoder architecture.
* [x] Train your autoencoder on the training data.
* [x] Plot your training and validation loss by epoch.
* [x] Quantify your average reconstruction error over the validation set.
* [x] Explain why you chose the hyperparameter values you did.

## Step 3: make your classifier

### Background

Transcription factors are proteins that bind DNA at promoters to drive gene expression. Most preferentially bind to specific sequences while ignoring others. Traditional methods to determine these sequences (called motifs) have assumed that binding sites in the genome are all independent. However, in some cases people have identified motifs where positional interdependencies exist.

### Your task

You will implement a multi-layer fully connected neural network using your `NeuralNetwork` class to predict whether a short DNA sequence is a binding site for the yeast transcription factor Rap1. The training data is incredibly imbalanced, with way fewer positive sequences than negative sequences, so you will implement a sampling scheme to ensure that class imbalance does not affect training. As in step 2, all of the following work should be done in a Jupyter Notebook.

### To-do

* [x] Use the `read_text_file` function from `io.py` to read in the 137 positive Rap1 motif examples.

* [x] Use the `read_fasta_file` function from `io.py` to read in all the negative examples. Note that these sequences are much longer than the positive sequences, so you will need to process them to the same length.

* [x] Balance your classes using your `sample_seq` function and explain why you chose the sampling scheme you did.

* [x] One-hot encode the data using your `one_hot_encode_seqs` function.

* [x] Split the data into training and validation sets.

* [x] Generate an instance of your `NeuralNetwork` class with an appropriate architecture.

* [x] Train your neural network on the training data.

* [x] Plot your training and validation loss by epoch.

* [x] Report the accuracy of your classifier on your validation dataset.

* [x] Explain your choice of loss function and hyperparameters.

## Grading (50 points)

### Neural network implementation (15 points)

* Proper implementation of methods in `NeuralNetwork` class (13 points)
* Proper implementation of `sample_seqs` function (1 point)
* Proper implementation of `one_hot_encode_seqs` function (1 point)

### Autoencoder (10 points)

* Read in data and generate training and validation sets (2 points)
* Successfully train your autoencoder (4 points)
* Plots of training and validation loss (2 points)
* Quantification of reconstruction error (1 point)
* Explanation of hyperparameters (1 point)

### Classifier (15 points)

* Correctly read in all data (2 points)
* Explanation of your sampling scheme (2 points)
* Proper generation of a training set and a validation set (2 point)
* Successfully train your classifier (4 points)
* Plots of training and validation loss (2 points)
* Reporting validation accuracy of the classifier (1 point)
* Explanation of loss function and hyperparameters (2 points)

### Testing (7 points)

Proper unit tests for:

* `_single_forward` method (1 point)
* `forward` method (1 point)
* `_single_backprop` method (1 point)
* `predict` method (1 point)
* `binary_cross_entropy` method (0.5 points)
* `binary_cross_entropy_backprop` method (0.5 points)
* `mean_squared_error` method (0.5 points)
* `mean_squared_error_backprop` method (0.5 points)
* `sample_seqs` function (0.5 points)
* `one_hot_encode_seqs` function (0.5 points)

### Packaging (3 points)

* Installable module (1 point)
* GitHub Actions (installing + testing) (2 points)

### Submission
Please submit a link to your final project repo [here](https://forms.gle/9xWdSinubVTYTwL2A)

## Project Structure

```
final-nn/
├── README.md                          # This file
├── LICENSE                            # License information
├── data/
│   ├── rap1-lieb-positives.txt       # 137 positive Rap1 binding sites
│   └── yeast-upstream-1k-negative.fa # 3,163 negative sequences
├── nn/
│   ├── __init__.py                   # Package initialization
│   ├── nn.py                         # NeuralNetwork class (main implementation)
│   ├── preprocess.py                 # Preprocessing functions (sample_seqs, one_hot_encode_seqs)
│   └── io.py                         # File I/O utilities (read_text_file, read_fasta_file)
├── test/
│   └── test_nn.py                    # Unit tests for neural network
├── autoencoder.ipynb                 # Autoencoder notebook (Step 2)
└── classifier.ipynb                  # Rap1 classifier notebook (Step 3)
```

## Installation & Usage

### Requirements
- Python 3.7+
- NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

### Setup
```bash
# Clone the repository
git clone https://github.com/Juan-Carlos-Gomez/final-nn.git
cd final-nn

# Install dependencies
pip install numpy scikit-learn matplotlib jupyter
```

### Running the Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Open and run:
# - autoencoder.ipynb (for autoencoder training)
# - classifier.ipynb (for Rap1 binding site classifier)
```

### Using the NeuralNetwork Class
```python
from nn.nn import NeuralNetwork

# Create a neural network
nn = NeuralNetwork(layer_sizes=[64, 16, 64])

# Train the network
nn.fit(X_train, y_train, learning_rate=0.01, epochs=100, batch_size=32)

# Make predictions
predictions = nn.predict(X_test)
```

## Key Results & Performance

| Task | Architecture | Best Validation Metric |
|------|--------------|------------------------|
| Autoencoder (digits) | 64 → 16 → 64 | Loss: 0.0739 |
| Classifier (Rap1 sites) | 68 → 32 → 16 → 1 | Accuracy: **100%** |

## Technical Highlights

- **From-scratch implementation:** No external deep learning libraries (TensorFlow/PyTorch)
- **Backpropagation:** Full implementation with gradient computation through all layers
- **Activation Functions:** ReLU (hidden) and Sigmoid (output)
- **Loss Functions:** MSE (regression) and Binary Cross-Entropy (classification)
- **Optimization:** Mini-batch stochastic gradient descent
- **Preprocessing:** Class balancing via downsampling, one-hot encoding for sequences
