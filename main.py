import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Image size
IMG_SIZE = 224

# Dataset paths
train_dir = "dataset/train"
test_dir = "dataset/test"

# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=4,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=4,
    class_mode='categorical'
)

print("Classes:", train_data.class_indices)

# Load pretrained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Build model
model = Sequential([
    base_model,
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(train_data, epochs=5, validation_data=test_data)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

# Save model
model.save("traffic_model.h5")
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("traffic_model.h5")

# IMPORTANT: class names must match your folders EXACTLY
class_names = ['horn', 'no_entry', 'speed_limit', 'stop']

# Load test image (change path if needed)
img_path = "dataset/test/stop/stopt_1.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.reshape(img, (1, 224, 224, 3))

# Predict
prediction = model.predict(img)
predicted_class = class_names[np.argmax(prediction)]

print("Predicted Sign:", predicted_class)