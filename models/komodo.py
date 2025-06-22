from unsloth import FastLanguageModel
import torch

class KomodoModel:
    def __init__(self, model_name, max_seq_length=1024, load_in_4bit=True):
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            load_in_4bit=self.load_in_4bit,
            token="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",  # Ganti dengan token Anda jika diperlukan
        )
        return model, tokenizer

    def get_peft_model(self, r=32, lora_alpha=16, lora_dropout=0, use_gradient_checkpointing=True):
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing=use_gradient_checkpointing,
            random_state=3407,
        )
        return self.model