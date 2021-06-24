from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras import utils, callbacks
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf
from kerastuner.applications import HyperResNet
from kerastuner.tuners import Hyperband
import keras_tuner as kt
from load_data import get_data
from settings import image_size

tf.random.set_seed(1337)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


# Custom hyper model
class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(
            Conv2D(hp.Choice('units', [32, 64, 128]), (3, 3), activation='relu',
                   input_shape=(image_size(), image_size(), 3)))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


# Load data
(training_data, training_labels), (testing_data, testing_labels) = get_data()
training_labels = utils.to_categorical(training_labels, num_classes=2)
testing_labels = utils.to_categorical(testing_labels, num_classes=2)
print(f"Training data size: {len(training_data)}")
print(f"Testing data size: {len(testing_data)}")

# Training parameters
hypermodel = HyperResNet(input_shape=(image_size(), image_size(), 3),
                         classes=2)
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy',
                                         patience=5)

# Tuning parameters
tuner = Hyperband(
    hypermodel,
    objective='val_accuracy',
    max_epochs=30,
    directory='tuner',
    project_name='hypertrain')

# Tune model
tuner.search(x=training_data, y=training_labels, epochs=25, batch_size=32,
             validation_data=(testing_data, testing_labels))
tuner.search_space_summary()
tuner.results_summary()
