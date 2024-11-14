import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import time
import pickle



unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax(dim=-1) # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # obtain first hidden layer representation
        hidden_layer = self.activation(self.W1(input_vector))
        # obtain output layer representation
        output_layer = self.W2(hidden_layer)
        # obtain probability dist.
        predicted_vector = self.softmax(output_layer)
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')

    parser.add_argument('--do_eval', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    
    #function to save model
    def save_model(model, filepath):
        torch.save(model.state_dict(), filepath)
        print(f'Model saved to {filepath}.')

    #function to load a saved model
    def load_model(model, filepath):
        model.load_state_dict(torch.load(filepath,weights_only=True))
        model.eval()
        return model
    
    #function to plot 
    def plot_learning_curves(train_losses, val_accuracies, epochs):
    # Create a figure with two subplots side by side
        plt.figure(figsize=(15, 5))
        
        # First subplot for training loss
        plt.subplot(1, 2, 1)
        plt.plot(range(1, epochs + 1), train_losses, 'b-o', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss by Epoch')
        plt.grid(True)
        plt.legend()
        
        # Second subplot for validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(range(1, epochs + 1), val_accuracies, 'orange', marker='o', label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy by Epoch')
        plt.grid(True)
        plt.legend()
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('images/learning_curves_128_25.png')
        plt.show()


    if args.do_train:
        train_losses = []
        val_accuracies = []

        model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
        optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
        print("========== Training for {} epochs ==========".format(args.epochs))
        for epoch in range(args.epochs):
            start_time = time.time()
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            model.train()
            optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            print("Training started for epoch {}".format(epoch + 1))
            random.shuffle(train_data) # Good practice to shuffle order of training data
            minibatch_size = 16 
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / (N // minibatch_size)
            train_losses.append(avg_epoch_loss)    
            
            print("Training completed for epoch {}".format(epoch + 1))
            print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Training time for this epoch: {}".format(time.time() - start_time))


            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            print("Validation started for epoch {}".format(epoch + 1))
            minibatch_size = 32 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
            print("Validation completed for epoch {}".format(epoch + 1))
            print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            print("Validation time for this epoch: {}".format(time.time() - start_time))
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
        
        total_time = time.time() - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")

        #save model after training
        save_path = 'saved_model/ffnn_model.pt'
        save_model(model, save_path)
        plot_learning_curves(train_losses, val_accuracies, args.epochs)


    else:
        print("Skipping training. Evaluate flag is on.")

    # write out to results/test.out

    model = FFNN(input_dim=len(word2index), h=args.hidden_dim)
    model = load_model(model,"saved_model/ffnn_model.pt") #change model accordingly
    if args.do_eval and args.test_data:
        with open(args.test_data) as test_f:
            test = json.load(test_f)
        test_data = [(elt["text"].split(),int(elt["stars"]-1)) for elt in test]
        test_data = convert_to_vector_representation(test_data, word2index)
        
        correct = 0
        total = 0
        for input_vector, gold_label in test_data:
            predicted_vector = model(input_vector)
            predicted_label = torch.argmax(predicted_vector)
            correct += int(predicted_label == gold_label)
            total += 1
        print(f"Test accuracy: {correct / total}")
    
    
    # print("Size of word2index dict:",len(word2index))
    # print("saving word2index...")
    # with open("word2index.pkl","wb") as file:
    #     pickle.dump(word2index,file)