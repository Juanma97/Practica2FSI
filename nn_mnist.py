#!/usr/bin/env python
# -*- coding: utf-8 -*-
import gzip
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from random import randint

import tensorflow as tf
import numpy as np

from random import randint
import time


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
f.close()

train_x, train_y = train_set
valid_x, valid_y = valid_set
test_x, test_y   = test_set

train_y = one_hot(train_y.astype(int), 10)
valid_y = one_hot(valid_y.astype(int), 10)
test_y = one_hot(test_y.astype(int), 10)


x = tf.placeholder("float", [None,784])
y_ = tf.placeholder("float", [None,10])


W1 = tf.Variable(np.float32(np.random.rand(784,30)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(30)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(30,10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10))* 0.1)
h = tf.nn.sigmoid(tf.matmul(x,W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20
lastError = 1000000
actualError = 0
epoch = 0
finish = True
check = 0
error = []


while finish:
    for jj in range(int(len(train_x) / batch_size)):
        # Entrenamiento
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # Validacion
    actualError = sess.run(loss, feed_dict={x: valid_x, y_: valid_y})
    error.append(actualError)

    epoch = epoch + 1
    if actualError >= lastError:
        check += 1
        if check > 6:
            finish = False
            print("********** Fin del entrenamiento **********")
            print("-Error anterior: " + str(lastError))
            print("-Error actual: " + str(actualError))
            break
    else:
        check = 0
        lastError = actualError

    print("[EPOCH]:", epoch, "Error:", actualError)
    print("-Error anterior: " + str(lastError))
    print("-Error actual: " + str(actualError))
    print("----------------------------------------------------------------------------------")

print("")
print("¡La fase de entrenamiento ha terminado!")
print("----------------------------------------------------------------------------------")
print("")
print("Fase de test")
print("")

done = 0

result = sess.run(y, feed_dict={x: test_x})

print("Test", "Error:", sess.run(loss, feed_dict={x: test_x, y_: test_y}))

for b, r in zip(test_y, result):
    if np.argmax(b) == np.argmax(r):
        done += 1


print("Porcentaje de aciertos: " + str(done / float(len(test_y)) * 100) + "%")
plt.plot(error)
plt.ylabel("ERROR")
plt.show()
print("----------------------------------------------------------------------------------")