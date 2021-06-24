import os
import cv2
from tqdm import tqdm
import pickle
import random
import numpy as np
from settings import image_size, categories, scaled_image_size

VALIDATION_SPLIT = 0.2


def load_data():
    training_map = []
    testing_map = []
    for category in categories():
        dir_path = os.path.join(category)
        class_num = categories().index(category)
        files = os.listdir(dir_path)
        size = len(files)
        progress = 0
        for img in tqdm(files):
            try:
                src_image = cv2.imread(os.path.join(dir_path, img))
                src_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB)
                if progress < size * VALIDATION_SPLIT:
                    image = cv2.resize(src_image, (scaled_image_size(), scaled_image_size()))
                    image = cv2.resize(image, (image_size(), image_size()))
                    image = np.array(image) / 255
                    testing_map.append([image, class_num])
                else:
                    for r in range(0, 2):
                        n = random.randint(60, 90)
                        image = cv2.resize(src_image, (n, n))
                        # if random.randint(0, 5) == 1:
                        #    image = cv2.blur(image, (5, 5))
                        # if random.randint(0, 5) == 1:
                        #    image = cv2.GaussianBlur(image, (5, 5), 0)
                        # if random.randint(0, 5) == 1:
                        #    image = cv2.medianBlur(image, 5)
                        image = cv2.resize(image, (image_size(), image_size()))
                        image = np.array(image) / 255
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
    for feature, label in full_dataset:
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
