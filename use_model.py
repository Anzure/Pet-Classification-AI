from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
from settings import image_size, categories
import os
from tqdm import tqdm


def get_image(target_path):
    image = cv2.imread(target_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size(), image_size()))
    return image


# Use model
model = keras.models.load_model("kjaledyr.model")
dir_path = os.path.join("Test")
count = 0
for file_path in tqdm(os.listdir(dir_path)):
    # Load image
    test_image = get_image(os.path.join(dir_path, file_path))
    test_input = np.array([test_image]) / 255
    # Predict outcome
    prediction = model.predict(test_input)[0]
    print("Debug", prediction)
    print("Argmax", np.argmax(prediction))
    result = categories()[np.argmax(prediction)]
    confidence = int(prediction[np.argmax(prediction)] * 100)
    print(f"Resultat: {result} ({confidence}%)")
    # Show results
    count += 1
    plt.subplot(4, 3, count)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_image, cmap=plt.cm.binary)
    plt.xlabel(f"{result} ({confidence}%)")
plt.tight_layout()
plt.show()
