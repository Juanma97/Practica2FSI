# -*- coding: utf-8 -*-

from random import randint
import time
import tensorflow as tf
import numpy as np

# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
# funcion que le pasas un 0 y devuelve 1,0,0,0
# n es el numero de bits
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


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code

print("\nSome samples...")
for i in range(20):
    print(x_data[i], " -> ", y_data[i])
print

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)
h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
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
training = int(len(x_data) * 0.7)  # 70% de las muestras (105(0-104))
validation = 0
finish = True
lastError = 100
check = 0

while finish:
    for j in range(10):
        randomNumber = randint(0, training - batch_size)
        batch_xs = x_data[training - batch_size - randomNumber: training - randomNumber]
        batch_ys = y_data[training - batch_size - randomNumber: training - randomNumber]
        trainData = sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    validation += 1
    batch_xs = x_data[training + 1: (len(x_data) - 1) - int(len(x_data) * 0.15)]  # x_data[106: 126]
    batch_ys = y_data[training + 1: (len(x_data) - 1) - int(len(x_data) * 0.15)]  # y_data[106: 126]
    actualError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})

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

    print("[EPOCH]:", validation, "Error:", actualError)
    print("-Error anterior: " + str(lastError))
    print("-Error actual: " + str(actualError))
    print("----------------------------------------------------------------------------------")


print("")
print("¡La fase de entrenamiento ha terminado!")
print("Empezando la fase de test...")
print("")
time.sleep(10)


ciclosTest = 0

batch_xs = x_data[(len(x_data) - 1) - int(len(x_data) * 0.15): len(x_data)]  # x_data[127 : 150]
batch_ys = y_data[(len(x_data) - 1) - int(len(x_data) * 0.15): len(x_data)]  # y_data[127 : 150]
ciclosTest += 1

actualError = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
print("Test:", ciclosTest, "Error:", actualError)
result = sess.run(y, feed_dict={x: batch_xs})
muestra = 1
for b, r in zip(batch_ys, result):
    print(muestra, ".-", b, "-->", r)
    muestra += 1
print("----------------------------------------------------------------------------------")