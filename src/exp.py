import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img

#img = load_img("./database/train/d6/d6_45angle_0000.jpg")
#plt.imshow(img)
#plt.show()
#print(cv2.imread("./database/train/d6/d6_45angle_0000.jpg").shape)


train = ImageDataGenerator(rescale=1/255)
validation = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory("./database/train/", target_size=(480,480), batch_size=32)
test_dataset = validation.flow_from_directory("./database/valid/", target_size=(480,480), batch_size=32)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(480,480,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation="relu")

])

print(train_dataset.class_indices)

