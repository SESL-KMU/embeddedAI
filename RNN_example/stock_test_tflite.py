import numpy as np
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
plt.plot(data)
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
    plt.show()


if ans == 'n' or ans == 'N':
    multi_step_plot(stock_data[-past_history:], np.array([0]),
                    output_data[0], 'Future month Predict')

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
