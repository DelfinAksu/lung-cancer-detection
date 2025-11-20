from src.deep.evaluate import evaluate_model

def main():
    print("Running Test Evaluation...")
    true_labels, pred_labels, class_names = evaluate_model()

if __name__ == "__main__":
    main()