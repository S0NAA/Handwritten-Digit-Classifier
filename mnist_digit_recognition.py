# mnist_digit_recognition.py

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# -----------------------------
# 2. Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0           # Normalize
y_train = to_categorical(y_train, 10)                       # One-hot encode
y_test = to_categorical(y_test, 10)

# -----------------------------
# 3. Build the model
model = Sequential([
    tf.keras.Input(shape=(28,28)),  # Modern way instead of input_shape in Flatten
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# -----------------------------
# 4. Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# -----------------------------
# 5. Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

# -----------------------------
# 6. Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# -----------------------------
# 7. Save the model
model.save("mnist_model.h5")

# -----------------------------
# 8. Visualize predictions (first 25 test images)
predictions = model.predict(x_test[:25])

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {predictions[i].argmax()}")
plt.show()
