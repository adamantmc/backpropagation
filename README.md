# Backpropagation
Implementation of a neural network trained using backpropagation.

The neural network implementation is in the `nn` module which contains the following files:
- `nn/nn.py`: NeuralNetwork class in which backpropagation is implemented
- `nn/optimizers.py`: Different optimization algorithms
- `nn/activation_functions.py`: Activation functions and their derivatives
- `nn/loss_functions.py`: Loss functions and their derivatives
- `nn/batch_provider.py`: Generator that splits a given `np.array` in batches
- `nn/utils.py`: Debugging utilities

Features:
- Gradient Checking
- L2 Regularization
- He Weight Initialization
- Momentum
- RMSProp
- Adam

## Example Problem
The implementation was tested on the IMDB Movie Review classification task. The dataset (http://ai.stanford.edu/~amaas/data/sentiment/) contains 50.000 movie reviews, split into 25.000 for training and 25.000 for testing.

Each review can be either positive (1) or negative (0). After shuffling the training set, 10% of the reviews (1250) are used as the validation set.

The reviews are processed using `scikit-learn`'s TfidfVectorizer in order to obtain vectors for each piece of text. The size vocabulary has an upper limit of 5000 words, after removing stop-words and very frequent words (max_df=0.5 - if more than half the reviews contain the same word, it's probably irrelevant to the review's label since the labels are distributed uniformly). 

The code that handles the text preprocessing and the network creation is in the `main.py` file.

#### Training Parameters

A 3-layer neural network was used with the following units: [128, 64, 1]. A learning rate of `0.05` was selected with a batch size of 512. The network was trained for 55 epochs. The training and validation set losses can be seen below:

![Alt text](losses.png?raw=true "Training and Validation set loss")

The network achieved an accuracy of `0.8756` on the test set.