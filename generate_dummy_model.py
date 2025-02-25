import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Create a Dummy Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')  # Assuming 5 categories: Healthy, Acne, Rash, Jaundice, Stress
])

# Save the model
model.save("disease_detection_model.h5")
print("✅ Dummy model created: 'disease_detection_model.h5'")
