#This is a template code. Please save it in a proper .py file.
import rtmaps.types
import numpy as np
from rtmaps.base_component import BaseComponent # base class
from sklearn.utils import shuffle
import cv2
import pickle
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from time import time
import matplotlib.image as mpimg
import csv
import sys
from random import shuffle
from time import time
from sklearn.utils import shuffle


# Python class that will be called from RTMaps.
class rtmaps_python(BaseComponent):
    def __init__(self):
        BaseComponent.__init__(self) # call base class constructor
        #self.add_input("in", rtmaps.types.ANY) # define input
        #self.add_output("out", rtmaps.types.AUTO) # define output

# Birth() will be called once at diagram execution startup
    def Birth(self):
        print("Python Birth")

# Core() is called every time you have a new input
    def Core(self):
        #out = self.inputs["in"].ioelt # create an ioelt from the input
        #self.outputs["out"].write(out) # and write it to the output
        training_file = '/home/bluebox/akash/data/train.p'
#
        testing_file = '/home/bluebox/akash/data/test.p'
        #
        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)
        #    
        X_train, y_train = train['features'], train['labels']

        n_classes = len(np.unique(y_train))

        def Model(x):    

            mu = 0
            sigma = 0.1
            keep_prob = 0.9
            strides_conv = [1, 1, 1, 1]
            strides_pool = [1, 2, 2, 1]
            
            #________________________________Layer 1__________________________________________________

            # Convolutional. Input = 32x32x1. Filter = 5x5x1. Output = 28x28x6.
            # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
            conv1_W = tf.Variable(tf.random_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
            conv1_b = tf.Variable(tf.zeros(6))
            conv1   = tf.nn.conv2d(x, conv1_W, strides=strides_conv, padding='VALID') + conv1_b

            # Apply activation function
            conv1 = tf.nn.relu(conv1)
            #conv1 = tf.nn.dropout(conv1, keep_prob)

            #________________________________Layer 2__________________________________________________
            
             # Convolutional. Input = 28x28x6. Filter = 3x3x6. Output = 14x14x6.
            conv2_W = tf.Variable(tf.random_normal(shape=(3, 3, 6, 12), mean = mu, stddev = sigma))
            conv2_b = tf.Variable(tf.zeros(12))
            conv2   = tf.nn.conv2d(conv1, conv2_W, strides=strides_conv, padding='SAME') + conv2_b

            # Pooling. Input = 28x28x6. Output = 14x14x6.
            conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            #________________________________Layer 3__________________________________________________

            # Convolutional. Input = 14x14x6. Filter = 5x5x12. Output = 10x10x16.
            conv3_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 12, 16), mean = mu, stddev = sigma))
            conv3_b = tf.Variable(tf.zeros(16))
            conv3   = tf.nn.conv2d(conv2, conv3_W, strides=strides_conv, padding='VALID') + conv3_b
            
            # Apply activation function
            conv3 = tf.nn.relu(conv3)

            # Pooling. Input = 10x10x16. Output = 5x5x16.
            conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            #________________________________Layer 4__________________________________________________
            
            # Flatten. Input = 5x5x16. Output = 400.
            fc0   = tf.reshape(conv3, [-1, int(5*5*16)])
            fc0 = tf.nn.dropout(fc0, keep_prob)
            
            # Fully Connected. Input = 400. Output = 120.
            fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
            fc1_b = tf.Variable(tf.zeros(120))
            fc1   = tf.matmul(fc0, fc1_W) + fc1_b
            
            # Apply activation function
            fc1 = tf.nn.relu(fc1)
            fc1 = tf.nn.dropout(fc1, keep_prob)

            #________________________________Layer 5__________________________________________________
                
            # Fully Connected. Input = 120. Output = 84.
            fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
            fc2_b  = tf.Variable(tf.zeros(84))
            fc2    = tf.matmul(fc1, fc2_W) + fc2_b
            
            # Apply activation function
            fc2  = tf.nn.relu(fc2)
            fc2  = tf.nn.dropout(fc2, keep_prob)

            #________________________________Layer 6__________________________________________________
            
            # Fully Connected. Input = 84. Output = 43.
            fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
            fc3_b  = tf.Variable(tf.zeros(n_classes))
            logits = tf.matmul(fc2, fc3_W) + fc3_b
            
            return logits

        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        y = tf.placeholder(tf.int32, (None))
        one_hot_y = tf.one_hot(y, n_classes)


        training_results = pd.read_csv('/home/bluebox/akash/accuracy_per_epoch.csv')

        logits = Model(x)

        images = []
        images_orig = []
        folder = "/home/bluebox/akash/new_images/"
        for image_name in os.listdir(folder):
            #reading in an image and resize it
            image = mpimg.imread(folder + image_name)
            images_orig.append(image)
            
            image = cv2.imread(folder + image_name,0)
            image = cv2.resize(image, (32,32))
            image = image/255
            images.append(image)

        X_data = np.asarray(images)
        X_data = X_data.reshape((len(images),32,32,1))
        print ("New images after reshape: ", X_data.shape)

        signs=[]
        with open('/home/bluebox/akash/signnames.csv', 'rt') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                signs.append((row['SignName']))
                
        with tf.Session() as sess:
            sess = tf.get_default_session()
            tf.train.Saver().restore(sess, tf.train.latest_checkpoint('/home/bluebox/akash/.'))
            signs_classes = sess.run(tf.argmax(logits, 1), feed_dict={x: X_data})
        for i in range(len(images)):

            print(signs[signs_classes[i]])
            plt.axis('off')

        sess.close()
           
        
# Death() will be called once at diagram execution shutdown
    def Death(self):
        pass
