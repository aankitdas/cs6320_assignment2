# Neural Networks for Sentiment Analysis

This project implements two types of neural networks for sentiment analysis: Feedforward Neural Network (FFNN) and Recurrent Neural Network (RNN).

## Model Architectures

### Feedforward Neural Network (FFNN)
- **Takes a fixed-length vector as input**
- **Architecture**:
  - Input: \(x \in \mathbb{R}^d\) (\(d\) = vocabulary size)
  - Hidden layer: \(h \in \mathbb{R}^{|h|}\) (\(h\) = hidden dimension)
  - Output layer: \(z \in \mathbb{R}^{|\mathcal{Y}|}\)
  - Final output: \(y\) (probability distribution where \(\sum_{i \in |\mathcal{Y}|} y[i] = 1\))
- **Uses bag-of-words vectorization** for review processing

### Recurrent Neural Network (RNN)
- **Processes sequences of vectors one at a time**
- **Architecture**:
  - Input: \(x_1, x_2, \ldots, x_k\) where \(x_i \in \mathbb{R}^e\) (\(e\) = embedding size)
  - Hidden states: \(h_1, h_2, \ldots, h_k\) where \(h_i \in \mathbb{R}^{|h|}\)
  - Output layer: \(\sum_{i=1}^k z_i\) where \(z_i \in \mathbb{R}^{|\mathcal{Y}|}\)
- **Uses word embeddings** for initialization

## Usage

### Training FFNN
```bash
python ffnn.py --hidden_dim [hparam] --epochs [hparam] \
    --train_data [train data path] --val_data [val data path] --do_train
```

### Training RNN
```bash
python rnn.py --hidden_dim [hparam] --epochs [hparam] \
    --train_data [train data path] --val_data [val data path]
```

## Features
- **Data loading functionality provided**
- **Bag-of-words vectorization** for FFNN
- **Pre-trained word embeddings** for RNN
- **Configurable hyperparameters**
- **Validation during training**

## Requirements
- **Python 3.x**
- **PyTorch**
- **Required data files**:
  - Training data (JSON)
  - Validation data (JSON)
  - Word embeddings (for RNN)

## Authors
- **Aankit Das**
- **Keda Kadu**

*Group 22

