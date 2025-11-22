from src.deep.train_cnn import train_model
from src.deep.evaluate import evaluate_model


def main():
    # 1) TRAINING PHASE (with CLAHE preprocessing)
    print("Lung Cancer Detection Project - Training with CLAHE preprocessing")

    model, history = train_model()

    # 2) TEST EVALUATION (on test set, also with CLAHE)
    print("\nTraining finished. Running Test Evaluation with CLAHE-preprocessed data...")
    true_labels, pred_labels, class_names = evaluate_model()


if __name__ == "__main__":
    main()

