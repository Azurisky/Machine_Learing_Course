
import argparse
import pickle
import gzip
from collections import Counter, defaultdict
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import time 


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # Load the dataset
        with gzip.open(location, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set

class CNN:
    '''
    CNN classifier
    '''
    def __init__(self, train_x, train_y, test_x, test_y, epoches = 15, batch_size=128):
        '''
        initialize CNN classifier
        '''
        self.batch_size = batch_size
        self.epoches = epoches

        # TODO: reshape train_x and test_x
        # reshape our data from (n, length) to (n, width, height, 1) which width*height = length
        self.train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
        self.test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))

        # normalize data to range [0, 1]
        # self.train_x /= 255
        # self.test_x /= 255

        # TODO: one hot encoding for train_y and test_y
        self.train_y = to_categorical(train_y, 10)
        self.test_y = to_categorical(test_y, 10)
        
        # TODO: build you CNN model
        self.model = Sequential()
        self.model.add(Conv2D(25, 3, 3, input_shape=(28, 28, 1)))
        self.model.add(MaxPool2D(2, 2))
        self.model.add(Conv2D(50, 3, 3))
        self.model.add(MaxPool2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(200, activation='relu'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(100, activation='tanh'))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(10, activation='softmax'))

        # self.model.add(Dense(10, activation=train_model))
        self.model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

    def train(self):
        '''
        train CNN classifier with training data
        :param x: training data input
        :param y: training label input
        :return:
        '''

        # TODO: fit in training data
        self.model.fit(self.train_x, self.train_y,
          batch_size=self.batch_size,
          epochs=self.epoches,
          verbose=1,
          validation_data=(self.test_x, self.test_y))
        pass

    def evaluate(self):
        '''
        test CNN classifier and get accuracy
        :return: accuracy
        '''
        acc = self.model.evaluate(self.test_x, self.test_y)
        return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help='Restrict training to this many examples')
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")
    batch_size = [32, 64, 128, 256, 512]
    model = ["sigmoid", "tanh", "softmax", "relu"]
    # model = ['sigmoid']
    a = []
    b = []
    c = []
    d = []
    # for i in batch_size:
    #     for j in model:
    #         print(i, j)
    cnn = CNN(data.train_x[:args.limit], data.train_y[:args.limit], data.test_x, data.test_y)
    cnn.train()
    acc = cnn.evaluate()
            # if j == "sigmoid":
            #     a += [acc[1]]
            # elif j == "tanh":
            #     b += [acc[1]]
            # elif j == "softmax":
            #     c += [acc[1]]
            # elif j == "relu":
            #     d += [acc[1]]
    print(acc)
    # print(a)
    # print(b)
    # print(c)
    # print(d)
    # plt.plot(batch_size, a, label="sigmoid")
    # plt.plot(batch_size, b, label="tanh")
    # plt.plot(batch_size, c, label="softmax")
    # plt.plot(batch_size, d, label="relu")
    # plt.xlabel('Number of batch size')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()