import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
from matplotlib import pyplot as plt
import time

ans = input('do you wanna check running time? (y/n)')
start = time.perf_counter()

fb = open('./B.txt', 'r')
data = ()
date = ()
while True:
    line = fb.readline()
    if not line:
        break
    if line[0] is 'D':
        continue
    line = np.array(line.split(','))
    line = np.delete(line, np.where(line == ''))
    Close = float(line[4])
    Date = line[0]
    data = np.append(data, Close)
    date = np.append(date, Date)

print(data)

if ans == 'n' or ans == 'N':
    plt.show()

stock_data = data[:, np.newaxis]
stock_data_min = stock_data.min()
stock_data_max_min_sub = stock_data.max() - stock_data.min()
stock_data = (stock_data - stock_data_min) / stock_data_max_min_sub

past_history = 365
future_target = 30

interpreter = tflite.Interpreter(model_path="./converted_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array([stock_data[-past_history:]], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


def create_time_steps(length):
    time_steps = []
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps


def multi_step_plot(history, true_future, prediction, title):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.title(title)

    plt.plot(num_in, np.array(history), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), label='True Future')
    if prediction.any():
        plt.plot(np.arange(len(prediction)), np.array(prediction), label='Predicted Future')
        plt.legend(loc='upper left')
    plt.savefig('savefig_default.png')
    plt.show()


TRAIN_SPLIT = int(len(stock_data) * 0.7)


def data_geenrator(dataset, start_index, end_index, history_size, target_size, single_step=False):
    _data_ = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):

        indices = range(i - history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        _data_.append(np.reshape(dataset[indices], (history_size, 1)))
        if single_step:
            labels.append(dataset[i + target_size])
        else:
            labels.append(dataset[i:i + target_size])
    return np.array(_data_), np.array(labels).squeeze()


x_val, y_val = data_geenrator(stock_data, TRAIN_SPLIT, None,
                              past_history,
                              future_target)

multi_step_plot(x_val[0], y_val[0],
                output_data[0], 'Validation Predict')

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)

# multi_step_plot(stock_data[-past_history:], np.array([0]),
#                 output_data[0],
#                 'Future month Predict')

img = cv2.imread('./savefig_default.png')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
