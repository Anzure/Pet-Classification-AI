import os
import cv2
from tqdm import tqdm
import pickle
import random
import numpy as np

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 70


def load_data():
    labeled_data_map = []
    for category in CATEGORIES:
        dir_path = os.path.join(category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(dir_path)):
            try:
                grey_image = cv2.imread(os.path.join(dir_path, img), cv2.IMREAD_GRAYSCALE)
                grey_image = cv2.resize(grey_image, (IMG_SIZE, IMG_SIZE))
                labeled_data_map.append([grey_image, class_num])
            except Exception as e:
                pass
    return labeled_data_map


def save_data(data, labels):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()
    return


# Load data
dataset = load_data()
random.shuffle(dataset)
print(f"Dataset size: {len(dataset)}")

# Prepare data
training_data = []
training_labels = []
for features, label in dataset:
    training_data.append(features)
    training_labels.append(label)
training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Save data
save_data(training_data, training_labels)
