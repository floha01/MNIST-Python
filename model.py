import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os

# Load Data

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Find out the split ratio
print("x_train " + str(x_train.shape))
print("y_train " + str(y_train.shape))
print("x_test " + str(x_test.shape))
print("y_test " + str(y_test.shape))

# Plotting the data
num_images = 3

# Show one Image at a time

for i in range(num_images):
    plt.imshow(x_train[i], cmap="grey")
    plt.show()

# Show all images in one plot
for i in range(num_images):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i], cmap="gray")
plt.show()
