from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

IMG_SIZE = 70


def load_data():
    pickle_in = open("X.pickle", "rb")
    data = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    labels = pickle.load(pickle_in)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


def train_model(input_size, conv_layer, conv_size, dense_layer, dense_size):
    NAME = f"{input_size}-{conv_layer}-{conv_size}-{dense_layer}-{dense_size}.run2"
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

    # Neural network
    model = Sequential()
    model.add(Conv2D(input_size, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))

    for l in range(conv_layer):
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(conv_size, (3, 3), activation='relu'))

    model.add(Flatten())

    for n in range(dense_layer):
        model.add(Dense(dense_size, activation='relu'))

    model.add(Dense(2, activation='softmax'))

    # Train model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(training_data, training_labels, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
    # model.save('katteroghunder.model')


# Load data
dataset = load_data()
training_data = dataset[0]
training_labels = dataset[1]
print(f"Training data: {len(training_data)}")

# Training parameters
input_sizes = [16, 32, 64]
conv_layers = [2, 3]
conv_sizes = [64, 128]
dense_layers = [1, 2]
dense_sizes = [32, 64]

# Train multiple models
for input_size in input_sizes:
    for conv_layer in conv_layers:
        for conv_size in conv_sizes:
            for dense_layer in dense_layers:
                for dense_size in dense_sizes:
                    try:
                        train_model(input_size, conv_layer, conv_size, dense_layer, dense_size)
                    except Exception as e:
                        pass
