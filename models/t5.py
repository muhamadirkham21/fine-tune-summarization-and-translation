from transformers import T5ForConditionalGeneration, AutoTokenizer
from .base_model import BaseSeq2SeqModel

class T5Model(BaseSeq2SeqModel):
    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
