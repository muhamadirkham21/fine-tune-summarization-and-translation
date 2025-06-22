from transformers import PegasusForConditionalGeneration, AutoTokenizer
from .base_model import BaseSeq2SeqModel

class PegasusModel(BaseSeq2SeqModel):
    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name)