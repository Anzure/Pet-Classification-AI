from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import time
from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband
import keras_tuner as kt
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
    model_name = f"{input_size}-{conv_layer}-{conv_size}-{dense_layer}-{dense_size}-{dropout_size}-{epochs}-{l2}_run1"
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
training_labels = keras.utils.to_categorical(training_labels, num_classes=2)
testing_labels = keras.utils.to_categorical(testing_labels, num_classes=2)
print(f"Training data size: {len(training_data)}")
print(f"Testing data size: {len(testing_data)}")


# Train best model
#train_model(32, 3, 64, 1, 64, 0.3, 50, 0.05)
#sys.exit(0)

# Custom hyper model
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(
            Conv2D(hp.Choice('units', [32, 64, 128]), (3, 3), activation='relu',
                   input_shape=(IMG_SIZE, IMG_SIZE, 3)))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


# Keras tuning
early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
hypermodel = HyperResNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), classes=2)

tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=20,
    directory='tuner',
    project_name='tuning')

tuner.search(x=training_data, y=training_labels, epochs=15, batch_size=32,
             validation_data=(testing_data, testing_labels), callbacks=[early_stopping])
