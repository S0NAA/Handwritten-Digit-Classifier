# MNIST Digit Recognition

A deep learning project that **recognizes handwritten digits (0–9)** using the MNIST dataset. The model is built with **TensorFlow and Keras** and can predict digits from the test dataset or your own images.

---

## **Project Overview**

This project demonstrates the use of a **neural network** to classify handwritten digits. The model is trained on **60,000 MNIST images** and evaluated on **10,000 test images**, achieving over **97% accuracy**. Predictions are visualized in a 5x5 grid for easy interpretation.

---

## **How It Works**

1. **Dataset**: Uses the MNIST dataset of 28x28 grayscale images of digits.  
2. **Preprocessing**: Normalizes pixel values and one-hot encodes labels.  
3. **Model Architecture**:
   - Flatten layer: Converts image to 784-length vector.
   - Dense layer: 128 neurons with ReLU activation.
   - Output layer: 10 neurons with softmax for digit classification.  
4. **Training**: Uses `categorical_crossentropy` loss and `adam` optimizer.  
5. **Prediction**: Model predicts digits from test images and displays a visual grid.  
6. **Model Saving**: The trained model is saved as `mnist_model.h5` for future use.

---
## Activate the virtual environment
1. **Windows:**
    - venv\Scripts\activate
2. **Mac/Linux:**
    - source venv/bin/activate

## Install dependencies
pip install -r requirements.txt

## Run the program
python mnist_digit_recognition.py
