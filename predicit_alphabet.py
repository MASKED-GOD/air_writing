import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import os
import cv2

# Load the dataset
dataset_path = r"D:\python\ML projects\air_canvas\letter-recognition.data.csv"
data = pd.read_csv(dataset_path, header=None)

# Preprocessing the data
X = data.iloc[:, 1:].values  # Features
y = data.iloc[:, 0].values   # Labels

# Convert the labels to one-hot encoding
y = pd.get_dummies(y).values

# Split the dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the model
model = Sequential([
    Dense(64, input_shape=(16,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(26, activation='softmax')  # 26 output classes (A-Z)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=250, batch_size=64, validation_data=(X_test, y_test))

# Save the model
model.save('letter_recognition_model.h5')

def classify_submission(irregularity_score):
    HUMAN_THRESHOLD = 0.25  # Example threshold
    BOT_THRESHOLD = 0.15    # Example threshold

    print(f"Evaluating with irregularity score: {irregularity_score}")

    if irregularity_score > HUMAN_THRESHOLD:
        return 'Human'
    elif irregularity_score < BOT_THRESHOLD:
        return 'Bot'
    else:
        return 'Uncertain'

def evaluate_irregularity(image_path, model):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"The file at {image_path} does not exist or is not a file.")
    
    try:
        # Preprocess the canvas image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (4, 4))  # Adjusting the size to match input shape
        img_normalized = img_resized / 255.0
        img_flattened = img_normalized.flatten().reshape(1, -1)
    except PermissionError:
        raise PermissionError(f"Permission denied for the file at {image_path}. Check file permissions.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the image: {e}")
    
    # Predict using the model
    predictions = model.predict(img_flattened)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Calculate confidence and irregularity score
    confidence = np.max(predictions)
    irregularity_score = calculate_irregularity_score(img)  # New function for irregularity
    
    # Classify as Human or Bot
    classification = classify_submission(irregularity_score)
    
    return predicted_class, irregularity_score, classification, confidence

def calculate_irregularity_score(img):
    """
    Calculate the irregularity score based on the contour analysis.
    """
    edges = cv2.Canny(img, 100, 200)  # Apply Canny edge detection
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    straight_line_score = 0
    total_contours = 0
    
    for contour in contours:
        if len(contour) >= 5:  # Minimum number of points to fit an ellipse
            ellipse = cv2.fitEllipse(contour)
            (center, axes, orientation) = ellipse
            
            # Calculate deviation from a perfect straight line
            deviation = np.std(np.diff(contour, axis=0))
            straight_line_score += deviation
            total_contours += 1
    
    if total_contours > 0:
        straight_line_score /= total_contours  # Average score per contour
    
    return straight_line_score

# Example usage:
image_path = r"D:\python\ML projects\air_canvas\img_to_check\captcha_box.png"
try:
    predicted_class, irregularity_score, classification, confidence = evaluate_irregularity(image_path, model)
    print(f"Predicted class: {predicted_class}")
    print(f"Irregularity score: {irregularity_score}")
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence}")

except Exception as e:
    print(f"Error: {e}")

model.summary()
