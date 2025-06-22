from train import train_pegasus, train_t5, train_komodo
from evaluate import Evaluator
from dataset import SummarizationDataset, TranslationDataset, KomodoDataset
import pandas as pd

def main():
    # Pilih model
    print("Pilih model yang akan digunakan:")
    print("1. Pegasus")
    print("2. T5")
    print("3. Komodo-7B")
    model_choice = input("Masukkan pilihan (1/2/3): ")
    model_type = "pegasus" if model_choice == "1" else "t5" if model_choice == "2" else "komodo"

    # Pilih task
    print("\nPilih task yang akan dilakukan:")
    print("1. Summarization")
    print("2. Translation")
    task_choice = input("Masukkan pilihan (1/2): ")
    task_type = "summarization" if task_choice == "1" else "translation"

    # Path dataset
    train_path = f"datasets/{task_type}_train.csv"
    test_path = f"datasets/{task_type}_test.csv"

    # Output directory
    output_dir = input("Masukkan nama direktori output (contoh: pegasus-summarization): ")

    # Jalankan training
    if model_choice == "1":
        print("\nTraining Pegasus...")
        train_pegasus(task_type, train_path, test_path, output_dir)
    elif model_choice == "2":
        print("\nTraining T5...")
        train_t5(task_type, train_path, test_path, output_dir)
    elif model_choice == "3":
        print("\nTraining Komodo-7B...")
        train_komodo(task_type, train_path, test_path, output_dir)
    else:
        print("Pilihan model tidak valid.")
        return

    # Evaluasi model
    print("\nEvaluasi model...")
    evaluator = Evaluator(task_type, model_type)
    test_data = pd.read_csv(test_path)

    if model_type in ["pegasus", "t5"]:
        if task_type == "summarization":
            test_dataset = SummarizationDataset(test_data, model.tokenizer)
        else:
            test_dataset = TranslationDataset(test_data, model.tokenizer)
    elif model_type == "komodo":
        dataset_handler = KomodoDataset(task_type)
        _, test_ds = dataset_handler.load_and_format_dataset(train_path, test_path)
        test_dataset = test_ds

    results = evaluator.evaluate(model, tokenizer, test_dataset, alpaca_prompt=dataset_handler.alpaca_prompt if model_type == "komodo" else None)
    results.to_csv(f"{output_dir}-results.csv", index=False)
    print(f"Hasil evaluasi disimpan di: {output_dir}-results.csv")

if __name__ == "__main__":
    main()