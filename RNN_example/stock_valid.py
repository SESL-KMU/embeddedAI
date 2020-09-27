import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

ans = input('do you wanna check running time? (y/n)')
start = time.perf_counter()

df = pd.read_csv('B.csv')
df = df[df['Close'].notnull()]
print(df.head())
df_close = df[['Close']]
df_close.index = df['Date']
print(df_close)
df_close.plot(subplots=True)
if ans == 'n' or ans == 'N':
    plt.show()

stock_data = df_close.values
stock_data_min = stock_data.min()
stock_data_max_min_sub = stock_data.max()-stock_data.min()
stock_data = (stock_data-stock_data_min)/stock_data_max_min_sub

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
  if ans == 'n' or ans == 'N':
      plt.show()

simple_lstm_model = tf.keras.models.load_model('stock_rnn', compile = False)

TRAIN_SPLIT = int(len(stock_data)*0.7)

def data_geenrator(dataset, start_index, end_index, history_size, target_size, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):

        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        if single_step:
            labels.append(dataset[i + target_size])
        else:
            labels.append(dataset[i:i + target_size])
    return np.array(data), np.array(labels).squeeze()

x_val, y_val = data_geenrator(stock_data, TRAIN_SPLIT, None,
                                       past_history,
                                       future_target)

BATCH_SIZE = 32

val_tensor = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_tensor = val_tensor.batch(BATCH_SIZE).repeat()

for x, y in val_tensor.take(1):
    plot = multi_step_plot(x[0].numpy(), y[0].numpy(),
                           simple_lstm_model.predict(x)[0], 'Validation Predict')

if ans == 'y' or ans == 'Y':
    elapsed = time.perf_counter() - start
    print('Elapsed %.3f seconds.' % elapsed)
