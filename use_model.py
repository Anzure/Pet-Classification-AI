from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import numpy as np
from settings import image_size, categories, scaled_image_size


def get_image(target_path):
    image = cv2.imread(target_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size))
    return image


# Load data
file_path = r'C:\Dev\AI\mu3.jpg'
test_image = get_image(file_path)
test_input = np.array([test_image]) / 255

# Use model
model = keras.models.load_model("katteroghunder.model")
prediction = model.predict(test_input)[0]
print("Debug", prediction)
print("Argmax", np.argmax(prediction))
result = categories[np.argmax(prediction)]
confidence = int(prediction[np.argmax(prediction)] * 100)
print(f"Resultat: {result}")

# Show results
plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(test_image, cmap=plt.cm.binary)
plt.xlabel(f"Resultat: {result} ({confidence}%)")
plt.show()
