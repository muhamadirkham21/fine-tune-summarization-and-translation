# models/base_model.py
from abc import ABC, abstractmethod
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

class BaseSeq2SeqModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.initialize_model()

    @abstractmethod
    def initialize_model(self):
        pass

    def train(self, train_dataset, eval_dataset, output_dir, training_params):
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model
        )

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            **training_params
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def generate(self, input_text, **gen_kwargs):
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).input_ids

        outputs = self.model.generate(inputs, **gen_kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)