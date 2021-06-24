from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf
import time
import sys as system
from load_data import get_data
from settings import image_size

tf.random.set_seed(1337)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)


def train_model():
    model_name = f"training.{int(time.time())}"
    tensorboard = TensorBoard(log_dir="logs/{}".format(f"{model_name}"))

    # Neural network
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size(), image_size(), 3)))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.3))

    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))
    model.add(Dropout(0.3))

    model.add(Dense(2, activation='softmax'))

    # Train model
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x=training_data,
              y=training_labels,
              batch_size=32,
              epochs=30,
              validation_data=(testing_data, testing_labels),
              callbacks=[tensorboard])
    model.save('kjaledyr.model')


# Load data
(training_data, training_labels), (testing_data, testing_labels) = get_data()
print(f"Training data size: {len(training_data)}")
print(f"Testing data size: {len(testing_data)}")

# Train model
train_model()
system.exit(0)
