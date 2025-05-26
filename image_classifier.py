import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import datasets, layers, models
import numpy as np 
import matplotlib.pyplot as plt
import os

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                'dog', 'frog', 'horse', 'ship', 'truck']

model_path = "cifar10_cnn_model.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
    print("loaded model from disk")
else:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, 
              validation_data=(test_images, test_labels))
    model.save(model_path)
    print("saved model to disk")
# Evaluate the model
def classify_image(image):
    image_arry = tf.expand_dims(image, 0)  
    predictions = model.predict(image_arry)
    predicted_class = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class]

def show_image_with_prediction(image, prediction):
    predicted_class = classify_image(image)
    plt.figure(figsize=(4, 4))
    plt.imshow(image, interpolation='nearest')
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

image_path  = "/Users/naveen/Projects/ai-ml/my_test/car/new.jpeg"

prediction = classify_image(test_images[4])
show_image_with_prediction(test_images[4], prediction)

    