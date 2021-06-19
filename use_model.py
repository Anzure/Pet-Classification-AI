from tensorflow import keras
import cv2
import matplotlib.pyplot as plt

CATEGORIES = ["Hund", "Katt"]
IMG_SIZE = 70


def get_image(target_path):
    color_image = cv2.imread(target_path, cv2.IMREAD_ANYCOLOR)
    color_image = cv2.resize(color_image, (IMG_SIZE, IMG_SIZE))
    grey_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    grey_image = cv2.resize(grey_image, (IMG_SIZE, IMG_SIZE))
    image = grey_image.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return image, grey_image, color_image


# Load data
file_path = r'C:\Dev\AI\mjau.jpg'
test_data = get_image(file_path)
test_image = test_data[0]

# Use model
model = keras.models.load_model("katteroghunder.model")
predictions = model.predict([test_image])
test_result = CATEGORIES[int(predictions[0][0])]
print(predictions[0][0])
print(f"Resultat: {test_result}")

# Show results
plt.subplot(2, 1, 1)
plt.xticks([])
plt.yticks([])
plt.imshow(test_data[2], cmap=plt.cm.binary)
plt.xlabel("Test bilde")
plt.subplot(2, 1, 2)
plt.xticks([])
plt.yticks([])
plt.imshow(test_data[1], cmap=plt.cm.binary)
plt.xlabel(f"Resultat: {test_result}")
plt.show()
