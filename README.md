# Assignment 2 - Neural Networks for Sentiment Analysis

This project implements two types of neural networks for sentiment analysis: Feedforward Neural Network (FFNN) and Recurrent Neural Network (RNN).

## Authors
- **Aankit Das**
- **Keda Kadu**

*Group 22

## Model Architectures

### Feedforward Neural Network (FFNN)
- **Takes a fixed-length vector as input**

- **Uses bag-of-words vectorization** for review processing

### Recurrent Neural Network (RNN)
- **Processes sequences of vectors one at a time**

- **Uses word embeddings** for initialization

## Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/aankitdas/cs6320_assignment2.git
   ```
2. **Install dependencies:**
    ```python
    pip install -r requirements.txt
    ```
3. **Training FFNN**
    ```bash
    python ffnn.py --hidden_dim [hparam] --epochs [hparam] \
        --train_data [train data path] --val_data [val data path] --do_train
    ```

4. **Training RNN**
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



