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
                image = cv2.imread(os.path.join(dir_path, img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                image = np.array(image) / 255
                labeled_data_map.append([image, class_num])
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
#training_data = np.array(training_data).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Save data
save_data(training_data, training_labels)
