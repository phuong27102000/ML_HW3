"""
This file is for fashion mnist classification
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_mnist_data
from logistic_np import add_one, LogisticClassifier
import time
#import pdb


class SoftmaxClassifier(LogisticClassifier):
    def __init__(self, w_shape):
        """__init__

        :param w_shape: create w with shape w_shape using normal distribution
        """
        super(SoftmaxClassifier, self).__init__(w_shape)


    def softmax(self, x):
        """softmax
        Compute softmax on the second axis of x

        :param x: input
        """
        # [TODO 2.3]
        # Compute softmax

        #code by Phuong from here
        x_max = np.amax(x, axis=1 ,keepdims=True)
        #code by Phuong to here

        return x_max


    def feed_forward(self, x):
        """feed_forward
        This function compute the output of your softmax regression model

        :param x: input
        """
        # [TODO 2.3]
        # Compute a feed forward pass

        #code by Phuong from here
        z = x @ self.w
        z_max = SoftmaxClassifier.softmax(self,z)
        z0 = np.exp(z - z_max)
        y_hat = z0/np.sum(z0, axis=1, keepdims=True)
        #code by Phuong to here

        return y_hat


    def compute_loss(self, y, y_hat):
        """compute_loss
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the class probabilities of all samples in our data
        """
        # [TODO 2.4]
        # Compute categorical loss

        #code by Phuong from here
        a = np.multiply(y,np.log(y_hat))
        loss = -np.mean(np.sum(a, axis=1))
        #code by Phuong to here

        return loss



    def get_grad(self, x, y, y_hat):
        """get_grad
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        """
        # [TODO 2.5]
        # Compute gradient of the loss function with respect to w

        #code by Phuong from here
        m = y.shape[0]
        grad = ( x.T @ (y_hat - y) )/m
        #code by Phuong to here

        return grad

def plot_loss(train_loss, val_loss):
    plt.figure(1)
    plt.clf()
    plt.plot(train_loss, color='b', label='Training loss')
    plt.plot(val_loss, color='g', label='Validating loss')
    plt.legend()


def draw_weight(w):
    label_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(2, figsize=(8, 6))
    plt.clf()
    w = w[0:(28*28),:].reshape(28, 28, 10)
    for i in range(10):
        ax = plt.subplot(3, 4, i+1)
        plt.imshow(w[:,:,i], interpolation='nearest')
        plt.axis('off')
        ax.set_title(label_names[i])


def normalize(train_x, val_x, test_x):
    """normalize
    This function computes train mean and standard deviation on all pixels then applying data scaling on train_x, val_x and test_x using these computed values
    Note that in this classification problem, the data is already flatten into a shape of (num_samples, image_width*image_height)

    :param train_x: train images, shape=(num_train, image_height*image_width)
    :param val_x: validation images, shape=(num_val, image_height*image_width)
    :param test_x: test images, shape=(num_test, image_height*image_width)
    """
    # [TODO 2.1]
    # train_mean and train_std should have the shape of (1, 1)

    #code by Phuong from here
    train_mean = np.mean(train_x)
    train_std = np.std(train_x)
    train_x_new = (train_x - train_mean)/train_std
    val_x_new = (val_x - train_mean)/train_std
    test_x_new = (test_x - train_mean)/train_std
    #code by Phuong to here

    return train_x_new, val_x_new, test_x_new

def create_one_hot(labels, num_k=10):
    """create_one_hot
    This function creates a one-hot (one-of-k) matrix based on the given labels

    :param labels: list of labels, each label is one of 0, 1, 2,... , num_k - 1
    :param num_k: number of classes we want to classify
    """
    # [TODO 2.2]
    # Create the one-hot label matrix here based on labels

    #code by Phuong from here
    one_hot_labels = np.zeros((labels.size, num_k))
    one_hot_labels[np.arange(labels.size),labels] = 1
    #code by Phuong to here

    return one_hot_labels


def test(y_hat, test_y):
    """test
    Compute the confusion matrix based on labels and predicted values

    :param classifier: the trained classifier
    :param y_hat: predicted probabilites, output of classifier.feed_forward
    :param test_y: test labels
    """

    #confusion_mat = np.zeros((10,10))

    # [TODO 2.7]
    # Compute the confusion matrix here

    #code by Phuong from here
    yy = y_hat - np.amax(y_hat, axis=1, keepdims=True)
    yy = np.equal(yy, 0)
    confusion_mat = test_y.T @ yy
    confusion_mat = confusion_mat/np.sum(confusion_mat, axis=1, keepdims=True)
    #code by Phuong to here

    #confusion_mat = confusion_mat/np.sum(confusion_mat, axis=1)
    np.set_printoptions(precision=2)
    print('Confusion matrix:')
    print(confusion_mat)
    print('Diagonal values:')
    print(confusion_mat.flatten()[0::11])


if __name__ == "__main__":
    np.random.seed(2020)

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())

    # Convert label lists to one-hot (one-of-k) encoding
    train_y = create_one_hot(train_y)
    val_y = create_one_hot(val_y)
    test_y = create_one_hot(test_y)

    # Normalize our data
    train_x, val_x, test_x = normalize(train_x, val_x, test_x)

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    val_x = add_one(val_x)
    test_x = add_one(test_x)

    # Create classifier
    num_feature = train_x.shape[1]
    dec_classifier = SoftmaxClassifier((num_feature, 10))
    momentum = np.zeros_like(dec_classifier.w)

    # Define hyper-parameters and train-related parameters
    num_epoch = 3347
    learning_rate = 0.04
    momentum_rate = 0.5
    epochs_to_draw = 10
    loss_change_valid = 0.000001
    all_train_loss = []
    all_val_loss = []
    plt.ion()

    for e in range(num_epoch):
        tic = time.perf_counter()
        train_y_hat = dec_classifier.feed_forward(train_x)
        val_y_hat = dec_classifier.feed_forward(val_x)

        train_loss = dec_classifier.compute_loss(train_y, train_y_hat)
        val_loss = dec_classifier.compute_loss(val_y, val_y_hat)

        grad = dec_classifier.get_grad(train_x, train_y, train_y_hat)

        # dec_classifier.numerical_check(train_x, train_y, grad)
        # Updating weight: choose either normal SGD or SGD with momentum
        #dec_classifier.update_weight(grad, learning_rate)
        dec_classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)

        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        toc = time.perf_counter()
        print(toc-tic)
        # [TODO 2.6]
        # Propose your own stopping condition here

        #code by Phuong from here
        if e > 10:
            loss_check = val_loss + loss_change_valid
            if np.max(all_val_loss[-5:-1]+[loss_check]) == loss_check:
                print("Training iteration stop at iteration %d" %e)
                break
        #code by Phuong to here

        if (e % epochs_to_draw == epochs_to_draw-1):
            plot_loss(all_train_loss, all_val_loss)
            plt.show()
            plt.pause(0.1)
            print("Epoch %d: train loss is %.5f, validate loss is %.5f" % (e+1, train_loss, val_loss))

    plt.savefig('softmax_np.png')
    y_hat = dec_classifier.feed_forward(test_x)
    test(y_hat, test_y)
