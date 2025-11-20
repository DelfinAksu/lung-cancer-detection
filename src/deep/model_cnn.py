from keras import layers, models
from src.config import IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, NUM_CLASSES

def build_cnn_model():
    """
    Custom CNN architecture for 4-class lung cancer classification.
    Input: CT images resized to (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
    Output: 4-class softmax probabilities.
    """
    model = models.Sequential(name="LungCancerCNN")

    # ----- Input layer -----
    model.add(layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))

    # ----- Conv Block 1 -----
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # ----- Conv Block 2 -----
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))

    # ----- Conv Block 3 -----
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))  # overfitting'e karşı

    # ----- Fully Connected -----
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))

    # ----- Output layer -----
    # 4 sınıf -> NUM_CLASSES nöron + softmax
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))

    # ----- Compile -----
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",  # çünkü label'lar one-hot (categorical)
        metrics=["accuracy"],
    )

    return model