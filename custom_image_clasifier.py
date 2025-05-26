import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os

# -------------------- CONFIG --------------------

DATASET_PATH = "/Users/naveen/Projects/ai-ml/my_test"
MODEL_PATH = "realworld_classifier_mobilenetv2.h5"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 123

# -------------------- LOAD RAW DATA --------------------

train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds_raw = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds_raw.class_names
print("📦 Class labels:", class_names)

# -------------------- DATA AUGMENTATION --------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# -------------------- NORMALIZE + PREPROCESS --------------------

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds_raw.map(lambda x, y: (data_augmentation(x, training=True), y))
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y)).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)

val_ds = val_ds_raw.map(lambda x, y: (normalization_layer(x), y)).cache().prefetch(tf.data.AUTOTUNE)

# -------------------- BUILD MODEL --------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

# -------------------- COMPILE & TRAIN --------------------

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

# -------------------- EVALUATE --------------------

val_loss, val_acc = model.evaluate(val_ds)
print(f"📊 Validation Accuracy: {val_acc:.2f}")

# -------------------- SAVE MODEL --------------------

model.save(MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")

# -------------------- PREDICT FUNCTION --------------------

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return

    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Preprocess the input image
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(predictions[0]), 2)

    # Show image + prediction
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class} ({confidence}%)")
    plt.axis('off')
    plt.show()

    print(f"🔮 Predicted: {predicted_class}, Confidence: {confidence}%")
    print("📊 All class probabilities:")
    for i, prob in enumerate(predictions[0]):
        print(f"  {class_names[i]}: {round(prob * 100, 2)}%")

    return predicted_class

# -------------------- RUN PREDICTION --------------------

img_path = input("📸 Enter full image path to classify: ").strip()
predict_image(img_path)