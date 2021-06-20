from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import numpy as np
import pickle
import time
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

IMG_SIZE = 70


def load_data():
    pickle_in = open("X.pickle", "rb")
    data = pickle.load(pickle_in)
    pickle_in.close()

    pickle_in = open("y.pickle", "rb")
    labels = pickle.load(pickle_in)
    pickle_in.close()

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def train_model(input_size, conv_layer, conv_size, dense_layer, dense_size, dropout_size, epochs):
    model_name = f"{input_size}-{conv_layer}-{conv_size}-{dense_layer}-{dense_size}-{dropout_size}-{epochs}_run13"
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
        model.add(Dense(dense_size, activation='relu', kernel_regularizer=keras.regularizers.l2(0.05)))
        model.add(Dropout(dropout_size))

    model.add(Dense(2, activation='softmax'))

    # Train model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(training_data, training_labels, batch_size=32, epochs=epochs, validation_split=0.3,
              callbacks=[tensorboard])
    model.save('katteroghunder.model')


# Load data
dataset = load_data()
training_data = dataset[0]
training_labels = dataset[1]
print(f"Training data: {len(training_data)}")

# Train best models
train_model(32, 3, 64, 1, 64, 0.2, 30)
sys.exit(0)

# Training parameters
input_sizes = [16]
conv_layers = [3]
conv_sizes = [64]
dense_layers = [1]
dense_sizes = [64]
epoch_lengths = [15]
dropout_sizes = [0.5, 0.7, 0.8]

# Train multiple models
for input_size in input_sizes:
    for conv_layer in conv_layers:
        for conv_size in conv_sizes:
            for dense_layer in dense_layers:
                for dense_size in dense_sizes:
                    for epoch_length in epoch_lengths:
                        for dropout_size in dropout_sizes:
                            try:
                                train_model(input_size, conv_layer, conv_size, dense_layer, dense_size, dropout_size,
                                            epoch_length)
                            except Exception as e:
                                pass
