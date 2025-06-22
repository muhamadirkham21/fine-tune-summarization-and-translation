from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, SFTTrainer, TrainingArguments, EarlyStoppingCallback
from models.pegasus import PegasusModel
from models.t5 import T5Model
from models.komodo import KomodoModel
from dataset import SummarizationDataset, TranslationDataset, KomodoDataset
import torch

def train_pegasus(task_type, train_path, test_path, output_dir):
    # Load dataset
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Initialize model
    model = PegasusModel("google/pegasus-xsum")

    # Create datasets
    if task_type == "summarization":
        train_dataset = SummarizationDataset(train_data, model.tokenizer)
        test_dataset = SummarizationDataset(test_data, model.tokenizer)
    elif task_type == "translation":
        train_dataset = TranslationDataset(train_data, model.tokenizer)
        test_dataset = TranslationDataset(test_data, model.tokenizer)
    else:
        raise ValueError("Task type not supported. Choose 'summarization' or 'translation'.")

    # Training parameters
    training_params = {
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "num_train_epochs": 10,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "predict_with_generate": True,
        "fp16": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "push_to_hub": False,
    }

    # Train
    model.train(train_dataset, test_dataset, output_dir, training_params)

def train_t5(task_type, train_path, test_path, output_dir):
    # Load dataset
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Initialize model
    model = T5Model("mesolitica/t5-base-standard-bahasa-cased")

    # Create datasets
    if task_type == "summarization":
        train_dataset = SummarizationDataset(train_data, model.tokenizer)
        test_dataset = SummarizationDataset(test_data, model.tokenizer)
    elif task_type == "translation":
        train_dataset = TranslationDataset(train_data, model.tokenizer)
        test_dataset = TranslationDataset(test_data, model.tokenizer)
    else:
        raise ValueError("Task type not supported. Choose 'summarization' or 'translation'.")

    # Training parameters
    training_params = {
        "learning_rate": 1e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 10,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "predict_with_generate": True,
        "fp16": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "push_to_hub": False,
    }

    # Train
    model.train(train_dataset, test_dataset, output_dir, training_params)

def train_komodo(task_type, train_path, test_path, output_dir):
    # Load dataset
    dataset_handler = KomodoDataset(task_type)
    train_ds, test_ds = dataset_handler.load_and_format_dataset(train_path, test_path)

    # Load model
    model = KomodoModel("Yellow-AI-NLP/komodo-7b-base")
    model.get_peft_model()

    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        warmup_steps=5,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        learning_rate=1e-5,
        fp16=False,
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=output_dir,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="epoch",
        save_total_limit=5,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model.model,
        tokenizer=model.tokenizer,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        dataset_text_field="texts",
        max_seq_length=1024,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model disimpan di: {output_dir}")