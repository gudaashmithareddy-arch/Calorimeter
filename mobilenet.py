import os
import numpy as np
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
CATEGORIES = ['waffles', 'pancakes', 'cup_cakes', 'pizza', 'donuts','apple_pie','baklava','ceviche','fried_rice','ice_cream','macarons','mussels','omelette',
                  'dumplings','edamame','falafel']  # Your categories
DIRECTORY = r"C:\Users\pavan\Downloads\archive\Food 101 dataset\test\test"  # Update the path accordingly

    # Label encoding
label_binarizer = LabelBinarizer()
    
# Initialize label binarizer
label_binarizer.fit(CATEGORIES)
def train_food_model_with_mobilenetv2():
    # Parameters
    EPOCHS = 50
    INIT_LR = 1e-4
    BS = 32
    IMAGE_SIZE = (128, 128)
   
    # Data preparation
    data = []
    labels = []

    print("[INFO] Loading dataset...")
    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                image = load_img(img_path, target_size=IMAGE_SIZE)
                image = img_to_array(image) / 255.0
                data.append(image)
                labels.append(category)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    labels = label_binarizer.transform(labels)
    num_classes = len(label_binarizer.classes_)

    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(np.array(data), np.array(labels), test_size=0.2, random_state=42)

    # Data augmentation
    aug = ImageDataGenerator(
        rotation_range=25, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2,
        zoom_range=0.2, horizontal_flip=True,
        fill_mode="nearest"
    )

    # Load the pre-trained MobileNetV2 model
    print("[INFO] Loading MobileNetV2...")
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Freeze the base model's layers

    # Add custom layers on top of MobileNetV2
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),  # Replaces Flatten layer
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    print("[INFO] Compiling the model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    print("[INFO] Training the model...")
    history = model.fit(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, verbose=1
    )

    # Unfreeze some base layers and fine-tune
    print("[INFO] Fine-tuning the model...")
    base_model.trainable = True
    for layer in base_model.layers[:100]:  # Freeze the first 100 layers
        layer.trainable = False

    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=INIT_LR / 10), metrics=["accuracy"])
    history_fine_tune = model.fit(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=5, verbose=1
    )

    # Save the model
    model.save("food_classifier_mobilenetv2.h5")
    print("[INFO] Model saved as 'food_classifier_mobilenetv2.h5'.")
    return model

train_food_model_with_mobilenetv2()