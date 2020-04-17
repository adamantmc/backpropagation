from nn.nn import NeuralNetwork
from nn.utils import plot_losses
import os
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def read_reviews(path):
    data = []
    labels = []

    label_dict = {"pos": 1, "neg": 0}

    for label in ["pos", "neg"]:
        review_dir = os.path.join(path, label)
        for file_path in os.listdir(review_dir):
            with open(os.path.join(review_dir, file_path), "r", encoding="utf8") as f:
                data.append(str(f.read()))

            labels.append(label_dict[label])

    return data, labels

def vectorize(train_reviews, test_reviews):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, max_df=0.5)
    vectorizer.fit(train_reviews)

    train_x = vectorizer.transform(train_reviews)
    test_x = vectorizer.transform(test_reviews)

    return train_x, test_x

def accuracy(y_pred, y_true):
    tp, tn, fp, fn = 0, 0, 0, 0

    for p, y in zip(y_pred, y_true):
        if p == y:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y == 1:
                fn += 1
            else:
                fp += 1

    return (tp + tn) / (tp + tn + fp + fn)

train_x_pickle = "train_x.pickle"
train_y_pickle = "train_y.pickle"
test_x_pickle = "test_x.pickle"
test_y_pickle = "test_y.pickle"

if all([os.path.exists(p) for p in [train_x_pickle, train_y_pickle, test_x_pickle, test_y_pickle]]):
    train_x = pickle.load(open("train_x.pickle", "rb"))
    test_x = pickle.load(open("test_x.pickle", "rb"))
    train_y = pickle.load(open("train_y.pickle", "rb"))
    test_y = pickle.load(open("test_y.pickle", "rb"))
else:
    print("Reading Train Reviews")
    train_reviews, train_y = read_reviews("./aclImdb/train")
    print("Reading Test Reviews")
    test_reviews, test_y = read_reviews("./aclImdb/test")

    pickle.dump(train_reviews, open("train_reviews.pickle", "wb"))
    pickle.dump(train_y, open("train_y.pickle", "wb"))
    pickle.dump(test_reviews, open("test_reviews.pickle", "wb"))
    pickle.dump(test_y, open("test_y.pickle", "wb"))

    print("Vectorizing Reviews")
    train_x, test_x = vectorize(train_reviews, test_reviews)
    pickle.dump(train_x, open("train_x.pickle", "wb"))
    pickle.dump(test_x, open("test_x.pickle", "wb"))

train_y = np.asarray(train_y).reshape(-1, 1)
test_y = np.asarray(test_y).reshape(-1, 1)

np.random.seed(2373)

random_indexes = np.random.choice(train_x.shape[0], size=train_x.shape[0], replace=False)
train_x = train_x[random_indexes]
train_y = train_y[random_indexes]

val_index = int(test_x.shape[0] * 0.1)
val_x = train_x[:val_index]
val_y = train_y[:val_index]
train_x = train_x[val_index:]
train_y = train_y[val_index:]

net = NeuralNetwork([128, 64, train_y.shape[1]], epochs=55, activation_dict={-1: "sigmoid"},
                    lr=0.05, batch_size=512, val_x=np.asarray(val_x.todense()), val_y=val_y)
net.fit(train_x, train_y)
plot_losses(net.training_losses, net.validation_losses)

preds = net.predict(np.asarray(test_x.todense()))
print(accuracy(preds, test_y))