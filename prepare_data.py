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
                # image90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                # image90 = cv2.blur(image, (5, 5))
                # image180 = cv2.rotate(image, cv2.ROTATE_180)
                # image180 = cv2.GaussianBlur(image, (5, 5), 0)
                # image270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # image270 = cv2.medianBlur(image, 5)
                image = np.array(image) / 255
                # labeled_data_map.append([np.array(image90) / 255, class_num])
                # labeled_data_map.append([np.array(image180) / 255, class_num])
                # labeled_data_map.append([np.array(image270) / 255, class_num])
                labeled_data_map.append([image, class_num])
            except Exception as e:
                pass
    return labeled_data_map


def save_data(data, labels):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(data, pickle_out)
    pickle_out.flush()
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.flush()
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

# Save data
print("Saving data...")
save_data(training_data, training_labels)
print("Saved data!")
