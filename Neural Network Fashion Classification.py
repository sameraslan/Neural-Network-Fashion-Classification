import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Scaling down the values to between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

'''print(train_images[7])

plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()'''

#input layer has 28x28 = 784 neurons
#output layer has 10 neurons (0-9), each representing a different class of fashion item

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', Loss='sparse_categorical_crossentropy', metrics=['accuracy'])