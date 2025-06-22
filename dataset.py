#datast.py
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset
import pandas as pd

class BaseDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length, target_max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.target_max_length = target_max_length

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text, max_length):
        return self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )

class SummarizationDataset(BaseDataset):
    def __init__(self, data, tokenizer, text_max_length=450, summary_max_length=64):
        super().__init__(data, tokenizer, text_max_length, summary_max_length)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text_enc = self._tokenize(row["text"], self.source_max_length)
        summary_enc = self._tokenize(row["summary"], self.target_max_length)

        labels = summary_enc["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": text_enc["input_ids"].flatten(),
            "attention_mask": text_enc["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

class TranslationDataset(BaseDataset):
    def __init__(self, data, tokenizer, src_max_length=64, tgt_max_length=64):
        super().__init__(data, tokenizer, src_max_length, tgt_max_length)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        src_enc = self._tokenize(row["indonesian"], self.source_max_length)
        tgt_enc = self._tokenize(row["english"], self.target_max_length)

        return {
            "input_ids": src_enc["input_ids"].flatten(),
            "attention_mask": src_enc["attention_mask"].flatten(),
            "labels": tgt_enc["input_ids"].flatten()
        }

class KomodoDataset:
    def __init__(self, task_type):
        self.task_type = task_type
        self.alpaca_prompt = self._get_prompt_template(task_type)

    def _get_prompt_template(self, task_type):
        if task_type == "summarization":
            return """Below are instructions that explain the task. Write a response that precisely completes the request.
### Instruction:
Summarize the following News article ""{}""!?
### Response: {} """
        elif task_type == "translation":
            return """Below are instructions that explain the task. Write a response that precisely completes the request.
### Instruction:
What does the sentence ""{}"" mean in English?
### Response: {} """
        else:
            raise ValueError("Task type not supported. Choose 'summarization' or 'translation'.")

    def load_and_format_dataset(self, train_path, test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_ds = HFDataset.from_pandas(train_data, split="train")
        test_ds = HFDataset.from_pandas(test_data, split="test")

        def formatting_prompts_func(examples):
            inputs = examples["text"] if self.task_type == "summarization" else examples["indonesian"]
            outputs = examples["summary"] if self.task_type == "summarization" else examples["english"]
            texts = [self.alpaca_prompt.format(input, output) + EOS_TOKEN for input, output in zip(inputs, outputs)]
            return {"texts": texts}

        train_ds = train_ds.map(formatting_prompts_func, batched=True)
        test_ds = test_ds.map(formatting_prompts_func, batched=True)

        return train_ds, test_ds