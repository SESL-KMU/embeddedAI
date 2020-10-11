import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt
import time

interpreter = tflite.Interpreter(model_path="./converted_model.tflite")
interpreter.allocate_tensors()

image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
image = image / 255.0
test_image = np.expand_dims(image, axis=0)

ans = input('do you wanna check running time? (y/n)')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(test_image.reshape(1, 28, 28), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

if ans =='y' or ans =='Y':
    for _ in range(3):
        start = time.perf_counter()
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(output_data)
        elapsed = time.perf_counter() - start
        print('Elapsed %.3f seconds.' % elapsed)
else:
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)

plt.imshow(test_image[0], cmap='Greys')
plt.savefig('savefig_default.png')
plt.show()

