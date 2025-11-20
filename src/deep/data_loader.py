from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    Train ve validation için Keras ImageDataGenerator oluşturur.
    Kaggle'dan gelen klasör yapısını (train/valid + 4 sınıf klasörü) kullanır.
    """

    # ----- 1) TRAIN tarafı: augmentation + normalization -----
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,      # piksel değerlerini [0,255] -> [0,1]
        rotation_range=10,        # hafif döndürme
        width_shift_range=0.05,   # yatay kaydırma
        height_shift_range=0.05,  # dikey kaydırma
        zoom_range=0.1,           # hafif zoom
        horizontal_flip=True      # yatay çevirme
    )

    # ----- 2) VALIDATION tarafı: sadece normalization -----
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    # ----- 3) TRAIN generator -----
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,                          # data/raw/chest_ct/train
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",           # 4 sınıf -> one-hot vektör
        color_mode="rgb",
        shuffle=True,
        seed=SEED
    )

    # ----- 4) VALIDATION generator -----
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,                            # data/raw/chest_ct/valid
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    # Sınıf isimlerini görmek için (raporda da kullanabiliriz)
    print("Class indices:", train_generator.class_indices)
    # Örn: {'adenocarcinoma': 0, 'large.cell.carcinoma': 1, ...}
    return train_generator, val_generator


def get_test_generator():
    """
    Creates test data generator (only rescaling, no augmentation).
    """
    test_datagen = ImageDataGenerator(rescale=1/255)

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return test_gen