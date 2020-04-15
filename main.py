from nn.nn import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Data from: https://www.kaggle.com/c/dogs-vs-cats/data
def load_dataset(image_folder="images/", split_ratio=0.7):
    labels_dict = {"cat": 0, "dog": 1}
    labels_dict = {"cat": [0, 1], "dog": [1, 0]}

    vectors_pickle_file = "image_vectors.pickle"
    labels_pickle_file = "image_labels.pickle"

    if os.path.exists(vectors_pickle_file) and os.path.exists(labels_pickle_file):
        print("Loading image vectors from {}".format(vectors_pickle_file))
        images = pickle.load(open(vectors_pickle_file, "rb"))
        print("Loading image labels from {}".format(labels_pickle_file))
        labels = pickle.load(open(labels_pickle_file, "rb"))
    else:
        from PIL import Image
        from keras.applications.resnet50 import ResNet50, preprocess_input
        from keras import Model

        resnet = ResNet50(weights='imagenet')
        feature_model = Model(inputs=resnet.input, outputs=resnet.get_layer("flatten_1").output)

        files = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if
                 os.path.isfile(os.path.join(image_folder, filename))][:100]
        np.random.shuffle(files)

        labels = [labels_dict[os.path.basename(f).split(".")[0]] for f in files]
        images = None

        step = 2000
        for i in range(0, len(files), step):
            file_batch = files[i:i + step]

            batch_images = []

            for filename in file_batch:
                img = Image.open(filename).resize((224, 224), Image.ANTIALIAS)
                img = np.asarray(img, dtype="float64")
                batch_images.append(preprocess_input(img.copy()))

            image_features = feature_model.predict(np.asarray(batch_images))

            if images is None:
                images = image_features
            else:
                images = np.concatenate((images, image_features), axis=0)

        pickle.dump(images, open(vectors_pickle_file, "wb"))
        pickle.dump(labels, open(labels_pickle_file, "wb"))

    train_last_index = int(split_ratio * len(images))
    no_classes = len(labels_dict["cat"]) if type(labels_dict["cat"]) == list else 1

    train_x = np.asarray(images[:train_last_index])
    train_y = np.asarray(labels[:train_last_index]).reshape(-1, no_classes)

    test_x = np.asarray(images[train_last_index:])
    test_y = np.asarray(labels[train_last_index:]).reshape(-1, no_classes)

    return train_x, train_y, test_x, test_y

# x, y, test_x, test_y = load_dataset("images/")
# x, y = load_breast_cancer(return_X_y=True)
# x = np.asarray(x)
#
# corrs = np.corrcoef(x, y, rowvar=False)
# indexed = []
#
# for c in corrs[-1][:-1]:
#     indexed.append((c, len(indexed)))
#
# indexed.sort(key=lambda x: x[0])
#
# cols = []
# for i in range(8):
#     cols.append(indexed[i][1])
#
# # x = x[:, cols]
#
# print(x.shape)
#
# x = normalize(x)
# print(y)


def load_planar_dataset():
    np.random.seed(1)
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y

np.random.seed(1)
x, y = load_planar_dataset()
net = NeuralNetwork([4, 1], epochs=10000, activation_dict={-1: "sigmoid", 0: "tanh"}, lr=1.2)
print(x.shape, y.shape)
net.fit(x, y)