import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

model = tf.keras.models.load_model('the_basics_mnist.model')

data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = data.load_data()

num = random.randint(0, len(x_test))
predictions = model.predict(x_test)

pred = np.argmax(predictions[num])
string_prediction = f'Model has predicted: {pred}'
plt.imshow(x_test[num])
plt.title(string_prediction)
plt.show()
