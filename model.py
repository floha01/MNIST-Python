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

# Normalize Train and Test data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the model

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dense(10, activation="softmax"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
