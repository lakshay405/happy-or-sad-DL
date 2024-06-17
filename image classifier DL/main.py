!pip install tensorflow tensorflow-gpu opencv-python matplotlib
!pip list

import tensorflow as tf
import os

# Configure GPU Memory Consumption Growth to prevent OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus: 
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.list_physical_devices('GPU')

import cv2
import imghdr

# Directory and image extensions
data_directory = 'data' 
image_extensions = ['jpeg', 'jpg', 'bmp', 'png']

# Loop through each image class in the directory
for image_class in os.listdir(data_directory): 
    for image in os.listdir(os.path.join(data_directory, image_class)):
        image_path = os.path.join(data_directory, image_class, image)
        try: 
            img = cv2.imread(image_path)
            img_type = imghdr.what(image_path)
            if img_type not in image_extensions: 
                print('Image extension not allowed {}'.format(image_path))
                os.remove(image_path)
        except Exception as e: 
            print('Issue with image {}'.format(image_path))
            # os.remove(image_path)

import numpy as np
from matplotlib import pyplot as plt

# Load images as TensorFlow dataset
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Visualize a batch of images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
plt.show()

# Normalize the image data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()

# Split dataset into train, validation, and test sets
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2)
test_size = int(len(data) * 0.1)
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# Build the convolutional neural network model
model = Sequential()
model.add(Conv2D(16, (3,3), strides=1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), strides=1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), strides=1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Set up TensorBoard callback for visualization
log_dir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Train the model
history = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Plot training history: Loss
fig = plt.figure()
plt.plot(history.history['loss'], color='teal', label='Train Loss')
plt.plot(history.history['val_loss'], color='orange', label='Val Loss')
plt.title('Model Loss', fontsize=20)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Plot training history: Accuracy
fig = plt.figure()
plt.plot(history.history['accuracy'], color='teal', label='Train Accuracy')
plt.plot(history.history['val_accuracy'], color='orange', label='Val Accuracy')
plt.title('Model Accuracy', fontsize=20)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate model metrics on test data
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()
for batch in test.as_numpy_iterator(): 
    X, y = batch
    y_pred = model.predict(X)
    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)
    accuracy.update_state(y, y_pred)
print(f'Precision: {precision.result()}, Recall: {recall.result()}, Accuracy: {accuracy.result()}')

# Load and predict on a new image
import cv2
img = cv2.imread('154006829.jpg')
plt.imshow(img)
plt.show()

# Resize and preprocess the image for prediction
resized_img = tf.image.resize(img, (256,256))
plt.imshow(resized_img.numpy().astype(int))
plt.show()

# Make prediction using the saved model
y_pred = model.predict(np.expand_dims(resized_img / 255, 0))
if y_pred > 0.5: 
    print(f'Predicted class: Malignant')
else:
    print(f'Predicted class: Benign')

# Save and load the trained model
model.save(os.path.join('models', 'image_classifier.h5'))
new_model = tf.keras.models.load_model(os.path.join('models', 'image_classifier.h5'))
new_model.predict(np.expand_dims(resized_img / 255, 0))
