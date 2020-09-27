import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

ans = input('do you wanna check running time? (y/n)')
start = time.perf_counter()

model = tf.keras.models.load_model('mnist_cnn', compile = False)
image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
image = image / 255.0

test_image = np.expand_dims(image, axis = 0)
predict = model(test_image.reshape(1,28,28))
print('predict ',predict, np.argmax(predict, axis=1))
plt.imshow(test_image[0], cmap='Greys')

if ans =='y' or ans =='Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
else:
    plt.show()
