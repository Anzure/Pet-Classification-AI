import os
import cv2
from tqdm import tqdm
import pickle
import random
import numpy as np

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 70
VALIDATION_SPLIT = 0.3


def load_data():
    training_map = []
    testing_map = []
    for category in CATEGORIES:
        dir_path = os.path.join(category)
        class_num = CATEGORIES.index(category)
        files = os.listdir(dir_path)
        size = len(files)
        progress = 0
        for img in tqdm(files):
            try:
                image = cv2.imread(os.path.join(dir_path, img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = np.array(image) / 255
                if progress < size * VALIDATION_SPLIT:
                    testing_map.append([image, class_num])
                else:
                    training_map.append([image, class_num])
                progress += 1
            except Exception as e:
                pass
    random.shuffle(training_map)
    random.shuffle(testing_map)
    return training_map, testing_map


def split_data(full_dataset):
    data_list = []
    label_list = []
    for feature, label in training_dataset:
        data_list.append(feature)
        label_list.append(label)
    return data_list, label_list


def save_data(train_data, train_labels, test_data, test_labels):
    pickle_out = open("X_train.pickle", "wb")
    pickle.dump(train_data, pickle_out)
    pickle_out.close()

    pickle_out = open("y_train.pickle", "wb")
    pickle.dump(train_labels, pickle_out)
    pickle_out.close()

    pickle_out = open("X_test.pickle", "wb")
    pickle.dump(test_data, pickle_out)
    pickle_out.close()

    pickle_out = open("y_test.pickle", "wb")
    pickle.dump(test_labels, pickle_out)
    pickle_out.close()
    return


# Load data
(training_dataset, testing_dataset) = load_data()
print("Processing data...")
(training_data, training_labels) = split_data(training_dataset)
(testing_data, testing_labels) = split_data(testing_dataset)
print(f"Training dataset size: {len(training_dataset)}")
print(f"Testing dataset size: {len(testing_dataset)}")

# Save data
print("Saving data...")
save_data(training_data, training_labels, testing_data, testing_labels)
print("Saved data!")
