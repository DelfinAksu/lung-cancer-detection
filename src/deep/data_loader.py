from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.preprocessing.preprocess import preprocess_image

from src.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMG_HEIGHT,
    IMG_WIDTH,
    BATCH_SIZE,
    SEED,
)


def get_data_generators():
    """
    Creates train and validation data generators with CLAHE preprocessing.
    """

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        seed=SEED
    )

    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    print("Class indices:", train_generator.class_indices)
    return train_generator, val_generator


def get_test_generator():
    """
    Creates test data generator with CLAHE preprocessing.
    """

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_image
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    return test_generator
