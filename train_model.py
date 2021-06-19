from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import pickle
import time

IMG_SIZE = 70
NAME = "70p-64x3cnn-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


def load_data():
    pickle_in = open("X.pickle", "rb")
    data = pickle.load(pickle_in)

    pickle_in = open("y.pickle", "rb")
    labels = pickle.load(pickle_in)

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


# Load data
dataset = load_data()
training_data = dataset[0]
training_labels = dataset[1]
print(f"Training data: {len(training_data)}")

# Neural network
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Train model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(training_data, training_labels, batch_size=32, epochs=10, validation_split=0.3, callbacks=[tensorboard])
model.save('katteroghunder.model')
