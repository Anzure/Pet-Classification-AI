from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import time
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision

tf.random.set_seed(1337)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

IMG_SIZE = 90


def load_data():
    pickle_in = open("X_train.pickle", "rb")
    train_data = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y_train.pickle", "rb")
    train_labels = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("X_test.pickle", "rb")
    test_data = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y_test.pickle", "rb")
    test_labels = pickle.load(pickle_in)
    pickle_in.close()

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    return (train_data, train_labels), (test_data, test_labels)


def train_model(input_size, conv_layer, conv_size, dense_layer, dense_size, dropout_size, epochs, l2):
    model_name = f"{input_size}-{conv_layer}-{conv_size}-{dense_layer}-{dense_size}-{dropout_size}-{epochs}-{l2}_run0"
    tensorboard = TensorBoard(log_dir="logs/{}".format(f"{model_name} {int(time.time())}"))

    # Neural network
    model = Sequential()
    model.add(Conv2D(input_size, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Dropout(dropout_size))

    for l in range(conv_layer):
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(conv_size, (3, 3), activation='relu'))
        model.add(Dropout(dropout_size))

    model.add(Flatten())

    for n in range(dense_layer):
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=keras.regularizers.l2(l2)))
        model.add(Dropout(dropout_size))

    model.add(Dense(2, activation='softmax'))

    # Train model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(training_data, training_labels, batch_size=32, epochs=epochs,
              validation_data=(testing_data, testing_labels), callbacks=[tensorboard])
    model.save('katteroghunder.model')


# Load data
(training_data, training_labels), (testing_data, testing_labels) = load_data()
print(f"Training data size: {len(training_data)}")
print(f"Testing data size: {len(testing_data)}")

# Train best models
train_model(32, 3, 64, 1, 64, 0.3, 50, 0.05)
sys.exit(0)

# Training parameters
input_sizes = [32, 64]
conv_layers = [3, 4]
conv_sizes = [64, 128]
dense_layers = [1]
dense_sizes = [64, 128]
epoch_lengths = [20]
dropout_sizes = [0.2, 0.3, 0.4]
l2_sizes = [0.01, 0.05]

# Train multiple models
for input_size in input_sizes:
    for conv_layer in conv_layers:
        for conv_size in conv_sizes:
            for dense_layer in dense_layers:
                for dense_size in dense_sizes:
                    for epoch_length in epoch_lengths:
                        for dropout_size in dropout_sizes:
                            for l2_size in l2_sizes:
                                try:
                                    train_model(input_size, conv_layer, conv_size, dense_layer, dense_size,
                                                dropout_size,
                                                epoch_length, l2_size)
                                except Exception as e:
                                    pass
