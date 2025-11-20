import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.deep.data_loader import get_data_generators
from src.deep.model_cnn import build_cnn_model
from src.config import EPOCHS, MODELS_DIR


def train_model():
    """
    Trains the custom CNN model using the Kaggle chest CT dataset.
    Uses:
      - data_loader.get_data_generators()
      - build_cnn_model()
      - 20 epochs (from config)
    Saves the best model under results/models/.
    """

    # 1) Klasörleri hazırla
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 2) Data generator'ları al
    train_gen, val_gen = get_data_generators()

    # 3) Modeli oluştur
    model = build_cnn_model()

    # 4) Callback'ler
    checkpoint_path = os.path.join(MODELS_DIR, "lung_cancer_cnn_best.h5")

    checkpoint_cb = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )

    earlystop_cb = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    reducelr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    )

    callbacks = [checkpoint_cb, earlystop_cb, reducelr_cb]

    # 5) Eğit
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # 6) Validation set üzerinde son performansı yazdır
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    print(f"Final validation loss: {val_loss:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")

    # 7) Son modeli de (callback dışında) kaydetmek istersen:
    final_model_path = os.path.join(MODELS_DIR, "lung_cancer_cnn_last.h5")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return model, history