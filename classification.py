import os
import numpy as np
import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

# Define paths and categories
image_dir = '/home/ashwin/ibmz/IBMZ2024/images'

# Define your categories based on the naming convention
categories = {
    'amazon': 'rainforest',  # Amazon corresponds to rainforest
    'sahel': 'desert',       # Sahel corresponds to desert
    'us': 'city',            # US corresponds to city
    'europe': 'city'         # Europe corresponds to city
}

# Reverse mapping to numeric labels
category_map = {v: idx for idx, v in enumerate(set(categories.values()))}  # Create mapping for labels

IMG_SIZE = 128  # Define the target image size
data = []
labels = []

# Load and preprocess the images
for img_name in os.listdir(image_dir):
    img_path = os.path.join(image_dir, img_name)

    # Extract category from the filename (the part after the last '-')
    category_key = img_name.split('-')[-1].lower()  # Get the category from the last part of the filename
    category_key = category_key.split('.')[0]  # Remove the file extension if present

    # Check if the extracted category is in our defined mapping
    if category_key in categories:
        class_label = category_map[categories[category_key]]  # Get numeric label
    else:
        print(f"Warning: Category '{category_key}' not recognized for image '{img_name}'. Skipping.")
        continue

    try:
        # Load the image and resize it to IMG_SIZE x IMG_SIZE
        img = Image.open(img_path).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))

        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Append the image data and its label to the respective lists
        data.append(img_array)
        labels.append(class_label)
        
    except Exception as e:
        print(f"Error loading image {img_name}: {e}")
        continue

# Convert lists to numpy arrays for model input
data = np.array(data)
labels = np.array(labels)

# Normalize the image data to be between 0 and 1
data = data / 255.0

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print the shapes to confirm
print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
# (numOfImgs, HeightOfImg, WidOfImg, NumOfChannels)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")







from tensorflow.keras import layers, models

# Define the CNN model
def create_model():
    model = models.Sequential()
    
    # First convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Second convolutional layer
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Third convolutional layer
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output
    model.add(layers.Flatten())
    
    # Fully connected layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(len(category_map), activation='softmax'))  # Output layer

    return model

# Create the model
model = create_model()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_split=0.2)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")





# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()


