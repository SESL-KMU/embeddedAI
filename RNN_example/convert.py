import tensorflow as tf

saved_model_dir = './stock_rnn/'
saved_model = "./stock_rnn/saved_model.pb"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

open('./converted_model.tflite', 'wb').write(tflite_model)
