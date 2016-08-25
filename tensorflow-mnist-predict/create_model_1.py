#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST biginners tutorial by Google. 
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""

#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])#先把place hold住之后，又传入值决定
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) #loss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.initialize_all_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init_op)

# Train the model and save the model to disk as a model.ckpt file
# file is stored in the same directory as this python script is started
"""
The use of 'with tf.Session() as sess:' is taken from the Tensor flow documentation
on saving and restoring variables.
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})#feed_dict是把值传入到placehold中
       	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
       	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
       	if i%50==0:
       		print "accuracy",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})     
save_path = saver.save(sess, "model.ckpt")
print ("Model saved in file: ", save_path)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
prediction=tf.argmax(y,1)
print "The last accuracy",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
