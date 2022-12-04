"""
This file is for multiclass fashion-mnist classification using TensorFlow

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from util import get_mnist_data
from logistic_np import add_one
from softmax_np import *
import time

if __name__ == "__main__":
    np.random.seed(2020)
    tf1.set_random_seed(2020)
    tf1.disable_eager_execution()

    # Load data from file
    # Make sure that fashion-mnist/*.gz files is in data/
    train_x, train_y, val_x, val_y, test_x, test_y = get_mnist_data()
    num_train = train_x.shape[0]
    num_val = val_x.shape[0]
    num_test = test_x.shape[0]

    # generate_unit_testcase(train_x.copy(), train_y.copy())

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

    # [TODO 2.8] Create TF placeholders to feed train_x and train_y when training

    #code by Phuong from here
    len_x = train_x.shape[1]
    len_y = train_y.shape[1]
    x = tf1.placeholder(tf.float32, shape=(None, len_x), name = 'x')
    y = tf1.placeholder(tf.float32, shape=(None, len_y), name = 'y')
    #code by Phuong to here

    # [TODO 2.8] Create weights (W) using TF variables

    #code by Phuong from here
    w_init = LogisticClassifier((len_x, len_y))
    w = tf1.Variable(w_init.w, name='w', dtype=tf.float32, shape=(len_x,len_y))
    #code by Phuong to here

    # [TODO 2.9] Create a feed-forward operator

    #code by Phuong from here
    z = tf.matmul(x, w)
    z_max = tf.math.reduce_max(z, axis=1 ,keepdims=True)
    z0 = tf.math.exp(z - z_max)
    pred = z0/tf.math.reduce_sum(z0, axis=1, keepdims=True)
    #code by Phuong to here

    # [TODO 2.10] Write the cost function

    #code by Phuong from here
    a = tf.math.multiply(y,tf.math.log(pred))
    cost = -tf.math.reduce_mean(tf.math.reduce_sum(a, axis=1)) #node_loss
    #code by Phuong to here

    # Define hyper-parameters and train-related parameters
    #num_epoch = 10000
    num_epoch = 3347
    learning_rate = 0.036
    momentum_rate = 0.45
    loss_change_valid = 0.000001

    # [TODO 2.8] Create an SGD optimizer

    #code by Phuong from here
    optimizer = tf1.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    # grad = tf.matmul(tf.transpose(x),pred - y)/num_train
    # past_momentum = tf1.Variable(tf.zeros((len_x,len_y)), name='momentum', dtype=tf.float32, shape=(len_x,len_y))
    # momentum = tf1.assign(past_momentum,past_momentum*momentum_rate + learning_rate*grad)
    # next_w = tf1.assign(w,w - momentum)
    #code by Phuong to here

    # Some meta parameters
    epochs_to_draw = 10
    all_train_loss = []
    all_val_loss = []
    plt.ion()
    num_val_increase = 0

    # Start training
    init = tf1.global_variables_initializer()

    with tf1.Session() as sess:

        sess.run(init)

        for e in range(num_epoch):
            tic = time.perf_counter()
            # [TODO 2.8] Compute losses and update weights here
            #train_loss = sess.run(cost, feed_dict={x:train_x, y:train_y})
            #val_loss = sess.run(cost, feed_dict={x:val_x, y:val_y})
            # Update weights
            #sess.run(optimizer, feed_dict={x: train_x, y: train_y})

            #code by Phuong from here
            train_loss = sess.run(cost, feed_dict={x:train_x, y:train_y})
            val_loss = sess.run(cost, feed_dict={x:val_x, y:val_y})
            sess.run(optimizer, feed_dict={x: train_x, y: train_y})
            # sess.run([next_w], feed_dict={x: train_x, y: train_y})
            #code by Phuong to here

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)
            toc = time.perf_counter()
            print(toc-tic)
            # [TODO 2.11] Define your own stopping condition here

            #code by Phuong from here
            if e > 10:
                loss_check = val_loss + loss_change_valid
                if max(all_val_loss[-5:-1]+[loss_check]) == loss_check:
                    print("Training iteration stop at iteration %d" %e)
                    break
            #code by Phuong to here

            if (e % epochs_to_draw == epochs_to_draw-1):
                plot_loss(all_train_loss, all_val_loss)
                plt.show()
                plt.pause(0.1)
                print("Epoch %d: train loss is %.5f, validate loss is %.5f" % (e+1, train_loss, val_loss))

        plt.savefig('softmax_tf.png')
        y_hat = sess.run(pred, feed_dict={x: test_x})
        test(y_hat, test_y)
