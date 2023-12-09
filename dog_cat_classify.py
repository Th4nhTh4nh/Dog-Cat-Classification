import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np


# model = load_model("dog_cat.h5")

# image_path = 'GPD.jpg'# Đường dẫn tới ảnh đầu vào
# image_path = "GPD.jpg"


def predictImage(image_path):
    model = load_model("dog_cat.h5")
    image = load_img(image_path, target_size=(192, 192))
    # Đảm bảo rằng kích thước ảnh phù hợp với model
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    image_array = image_array / 255.0  # Chuẩn hóa dữ liệu
    predictions = model.predict(image_array)

    class_names = ["Cat", "Dog"]  # Danh sách các lớp trong mô hình (theo thứ tự)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100

    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2f}%")

    return predicted_class, confidence


# predictImage("/uploads/GPD.jpg")
