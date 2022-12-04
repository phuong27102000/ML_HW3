"""
This file is for binary classification using TensorFlow
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from util import get_vehicle_data
from logistic_np import *

if __name__ == "__main__":
    np.random.seed(2018)
    tf1.set_random_seed(2018)
    tf1.disable_eager_execution()

    # Load data from file
    # Make sure that vehicles.dat is in data/
    train_x, train_y, test_x, test_y = get_vehicle_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]

    #generate_unit_testcase(train_x.copy(), train_y.copy())
    # logistic_unit_test()

    # Normalize our data: choose one of the two methods before training
    #train_x, test_x = normalize_all_pixel(train_x, test_x)
    train_x, test_x = normalize_per_pixel(train_x, test_x)

    # Reshape our data
    # train_x: shape=(2400, 64, 64) -> shape=(2400, 64*64)
    # test_x: shape=(600, 64, 64) -> shape=(600, 64*64)
    train_x = reshape2D(train_x)
    test_x = reshape2D(test_x)

    # Pad 1 as the last feature of train_x and test_x
    train_x = add_one(train_x)
    test_x = add_one(test_x)

    # [TODO 1.11] Create TF placeholders to feed train_x and train_y when training

    #code by Phuong from here
    x = tf1.placeholder(tf.float32, shape=(None, 4097), name = 'x')
    y = tf1.placeholder(tf.float32, shape=(None, 1), name = 'y')
    #code by Phuong to here

    # [TODO 1.12] Create weights (W) using TF variables

    #code by Phuong from here
    num_feature = train_x.shape[1]
    w_init = LogisticClassifier((num_feature, 1))
    w = tf1.Variable(w_init.w, name='w', dtype=tf.float32, shape=(4097,1))
    #code by Phuong to here

    # [TODO 1.13] Create a feed-forward operator

    #code by Phuong from here
    pred = tf.math.pow(1 + tf.math.exp(-tf.matmul(x,w)),-1)
    #code by Phuong to here

    # [TODO 1.14] Write the cost function
    #cost = -tf.reduce_sum(y*tf.log(pred)+(1-y)*tf.log(1-pred))/num_train

    #code by Phuong from here
    cost = -tf.math.reduce_mean(y*tf.math.log(pred)+(1-y)*tf.math.log(1-pred))
    #code by Phuong to here

    # Define hyper-parameters and train-related parameters
    num_epoch = 1000
    #learning_rate = 0.01
    #momentum_rate = 0.9

    #code by Phuong from here
    learning_rate = 0.0012
    momentum_rate = 0.75
    #code by Phuong to here

    # [TODO 1.15] Create an SGD optimizer
    # Update weight...

    #code by Phuong from here
    optimizer = tf1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # past_momentum = tf1.Variable(tf.zeros((4097,1)), name='momentum', dtype=tf.float32, shape=(4097,1))
    # w_grad = tf.matmul(tf.transpose(x),(pred - y))/num_train
    # momentum = tf1.assign(past_momentum,past_momentum*momentum_rate + learning_rate*w_grad)
    # next_w = tf1.assign(w,w - momentum)
    #code by Phuong to here

    # Some meta parameters
    epochs_to_draw = 100
    all_loss = []
    plt.ion()

    # Start training
    init = tf1.global_variables_initializer()

    with tf1.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.perf_counter()
            # [TODO 1.16] Compute loss and update weights here

            #code by Phuong from here
            loss = sess.run(cost, feed_dict={x: train_x, y: train_y})
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            #sess.run(next_w, feed_dict={x: train_x, y: train_y})
            #code by Phuong to here

            all_loss.append(loss)

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_loss)
                plt.show()
                plt.pause(0.1)
                print("Epoch %d: loss is %.5f" % (e+1, loss))
            toc = time.perf_counter()
            print(toc-tic)

        plt.savefig('logistic_tf.png')
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
