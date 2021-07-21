import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')
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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

# 5 epochs results in 87% accuracy and 20 epochs in 89%
#print("Tested: ", test_acc)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: ", test_labels[i])
    plt.title("Prediction: ", class_names[np.argmax(prediction[i])])
    plt.show()
    
#print(prediction[0])
#print(class_names[np.argmax(prediction[0])])