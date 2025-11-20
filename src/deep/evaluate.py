import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

from src.deep.data_loader import get_test_generator
from src.config import MODELS_DIR

def evaluate_model():
    """
    Loads the BEST model and evaluates it on the test set.
    Returns predictions and true labels for confusion matrix later.
    """

    # 1) Load test generator
    test_gen = get_test_generator()

    # 2) Load best model
    best_model_path = os.path.join(MODELS_DIR, "lung_cancer_cnn_best.h5")
    print(f"Loading best model: {best_model_path}")
    model = load_model(best_model_path)

    # 3) Evaluate on test set
    loss, acc = model.evaluate(test_gen, verbose=1)
    print(f"\n🔥 Test Loss: {loss:.4f}")
    print(f"🔥 Test Accuracy: {acc:.4f}")

    # 4) Predictions for confusion matrix
    preds = model.predict(test_gen)
    pred_labels = np.argmax(preds, axis=1)
    true_labels = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, pred_labels)
    print("\nConfusion Matrix:")
    print(cm)

    # --- Classification Report ---
    print("\nClassification Report:")
    print(classification_report(true_labels, pred_labels, target_names=class_names))

    return true_labels, pred_labels, class_names