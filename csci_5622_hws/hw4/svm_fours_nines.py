import argparse
import numpy as np
import time

from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        train_set, valid_set, test_set = pickle.load(f)

        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]

        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]

        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]

        f.close()

def mnist_digit_show(flatimage, outname=None):

    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1,28))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()

if __name__ == "__main__":
    timeStart = time.time()
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = FoursAndNines("../data/mnist.pkl.gz")

    # TODO: Use the Sklearn implementation of support vector machines to train a classifier to
    # distinguish 4's from 9's (using the MNIST data from the KNN homework).
    # Use scikit-learn's Grid Search (http://scikit-learn.org/stable/modules/grid_search.html) to help determine
    # optimial hyperparameters for the given model (e.g. C for linear kernel, C and p for polynomial kernel, and C and gamma for RBF).
    
    parameters = [
    # {'kernel':['linear'], 'C':[1, 10, 100, 1000]},
    # {'kernel':['poly'], 'C':[10000, 100000], 'degree':[2, 3, 4, 5, 6]}
    # {'kernel':['rbf'], 'C':[1, 10, 100, 1000], 'gamma':[0.1, 0.001, 0.0001, 0.00001]}
    ]
    # parameters = [
    # {'kernel':['linear'], 'C':[1, 10]}
    # ]
    
    # print(clf.predict(data.x_test))
    # print(data.y_test)
    # print(clf.score(data.x_valid, data.y_valid))

    # print("# Tuning hyper-parameters for precision")
    # print("")

    
    # svc = SVC()
    # clf = GridSearchCV(svc, parameters)
    # clf.fit(data.x_train, data.y_train)

    # print("Best parameters set found on development set:")
    # # print()
    # print(clf.best_params_)
    # # print()
    # print("Grid scores on development set:")
    # # print()
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #     print("%0.3f (+/-%0.03f) for %r"
    #           % (mean, std * 2, params))
    # # print()

    # print("Detailed classification report:")
    # # print()
    # print("The model is trained on the full development set.")
    # print("The scores are computed on the full evaluation set.")
    # # print()
    # y_true, y_pred = data.y_test, clf.predict(data.x_test)
    # print(classification_report(y_true, y_pred))
    # # print()
    # # -----------------------------------
    # # Plotting Examples
    # # -----------------------------------

    svc = SVC(C=1000, kernel='poly', degree=2)
    svc.fit(data.x_train, data.y_train)
    print(svc.support_)
    print("\n")
    print(svc.support_vectors_)

    # Display in on screen
    print(data.y_train[svc.support_[0]])
    mnist_digit_show(data.x_train[svc.support_[0],:])
    print(data.y_train[svc.support_[-1]])
    mnist_digit_show(data.x_train[svc.support_[-1],:])

    # Plot image to file
    # mnist_digit_show(data.x_train[1,:], "mnistfig.png")
    timeEnd = time.time()
    print(timeEnd - timeStart)
