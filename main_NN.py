'''
Compile environment : anaconda python 3.6
Using Google Tensorflow Package to train a Multi-layer Perceptron,
'''
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DEBUG = True

X = np.array([[3.98,1,2.5],
              [1.2,1.78,3.4],
              [8.9,8.8,7.9],
              [9.8,8.7,9.01],
              [1.22,0.6,2.1],
              [7.8,9.5,8.45]],dtype=np.float32)
y = np.array([[1,1,2,2,1,2]],dtype=np.float32)
y = y.T
test_X=np.array([[5,6,7],[1,1,1],[0.8,9,3.5]],dtype=np.float32)

if DEBUG:
    print('shape X:', X.shape)
    print('shape y:', y.shape)
    print('X:\n', X)
    print('y:\n', y)

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
test_X = sc.transform(test_X)


# Claim the Hyper parameters
LR = 0.1   # Learning rate
epochs = 500

# Define the place holder
tf_X = tf.placeholder(tf.float32, [None, 3])
tf_y = tf.placeholder(tf.float32, [None, 1])

# Define the networks
layer1 = tf.layers.dense(tf_X, 20,activation=tf.nn.relu)
pred = tf.layers.dense(layer1, 1)

# Define the Loss function ==> MSE
loss_func = tf.reduce_mean(tf.square(tf_y-pred))

# Define the Optimizer to optimize the loss
train = tf.train.GradientDescentOptimizer(LR).minimize(loss_func)

# Training
sess = tf.Session()
sess.run(tf.global_variables_initializer())

er = []
val = []
for i in range(epochs):
    feed_dict = {tf_X:X, tf_y:y}
    _, e = sess.run([train, loss_func], feed_dict=feed_dict)
    er.append(e)


def Plot_loss(er, val_er=None):
    er_times = np.linspace(1,len(er),len(er))

    plt.plot(er_times, er, 'r')
    if val_er != None:
        val_er_times = np.linspace(1,len(val_er), len(val_er))
        plt.plot(val_er_times, val_er, 'b')
        plt.legend(['er',['val_er']])

    plt.legend(['er'])
    plt.show()

Plot_loss(er)

print(er[-1])

a = sess.run(pred, feed_dict={tf_X:test_X})
print(a)

