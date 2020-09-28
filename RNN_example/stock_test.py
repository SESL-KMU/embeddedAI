import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
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


simple_lstm_model = tf.keras.models.load_model('stock_rnn', compile=False)
print(simple_lstm_model.predict(np.array([stock_data[-past_history:]]))[0])

if ans == 'n' or ans == 'N':
    multi_step_plot(stock_data[-past_history:], np.array([0]),
                    simple_lstm_model.predict(np.array([stock_data[-past_history:]]))[0],
                    'Future month Predict')

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
