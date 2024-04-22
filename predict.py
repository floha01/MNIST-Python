import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
import numpy as np

# Load Model
model = tf.keras.models.load_model("mnist.keras")

# Load test images
test_folder = "test-img"
file_list = os.listdir(test_folder)

for file_name in file_list:
    image_path = os.path.join(test_folder, file_name)
    try:
        image = cv2.imread(image_path)
        if image is not None:
            image = image[:, :, 0]
            image = np.invert(image)
            image = np.array([image])
            prediction = model.predict(image)
            pred_number = np.argmax(prediction)
            plt.imshow(image[0], cmap="binary")
            plt.title(f"This is probably a {pred_number}")
            plt.show()
        else:
            print(f"Error: Image could not be loaded {image_path}")
    except Exception as e:
        print(f"Error during prcessing of {image_path}: {str(e)}")
