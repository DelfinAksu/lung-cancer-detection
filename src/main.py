from src.deep.data_loader import get_data_generators

def main():
    print("Lung Cancer Detection Project - Data Loader Test")
    
    # Data generator'ları oluştur
    train_gen, val_gen = get_data_generators()

    # Birkaç bilgi yazdıralım
    print(f"Train samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print("Class indices:", train_gen.class_indices)

if __name__ == "__main__":
    main()