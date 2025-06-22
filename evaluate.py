import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.stem import PorterStemmer
from tqdm import tqdm
import re

class Evaluator:
    def __init__(self, task_type, model_type):
        self.task_type = task_type
        self.model_type = model_type
        self.stemmer = PorterStemmer()
        self.smoothie = SmoothingFunction().method4
        if task_type == "summarization":
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def extract_response(self, text):
        """
        Ekstrak teks setelah ## Response untuk Komodo-7B.
        """
        if isinstance(text, list):
            text = text[0] if text else ""
        match = re.search(r'## Response:\s*(.*?)(?=\s{2,}|\n|$)', text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def evaluate(self, model, tokenizer, test_dataset, alpaca_prompt=None):
        """
        Evaluasi model berdasarkan task dan jenis model.
        """
        results = []
        if self.model_type in ["pegasus", "t5"]:
            # Evaluasi untuk Pegasus dan T5
            if self.task_type == "summarization":
                rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
                for i in tqdm(range(len(test_dataset))):
                    inputs = tokenizer(
                        test_dataset[i]['text'],
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=150, use_cache=True)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reference_summary = test_dataset[i]['summary']
                    rouge_scores = self.scorer.score(reference_summary, generated_text)
                    results.append({
                        'Input': test_dataset[i]['text'],
                        'Generated Summary': generated_text,
                        'Reference Summary': reference_summary,
                        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
                        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
                        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
                    })
                return pd.DataFrame(results)
            elif self.task_type == "translation":
                bleu_scores = []
                for i in tqdm(range(len(test_dataset))):
                    inputs = tokenizer(
                        test_dataset[i]['indonesian'],
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=70, use_cache=True)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    reference = [self.stemmer.stem(word) for word in test_dataset[i]['english'].lower().split()]
                    hypothesis = [self.stemmer.stem(word) for word in generated_text.lower().split()]
                    bleu = sentence_bleu([reference], hypothesis, smoothing_function=self.smoothie)
                    results.append({
                        'Input': test_dataset[i]['indonesian'],
                        'Generated Translation': generated_text,
                        'Reference Translation': test_dataset[i]['english'],
                        'BLEU Score': bleu,
                    })
                return pd.DataFrame(results)

        elif self.model_type == "komodo":
            # Evaluasi khusus untuk Komodo-7B
            if self.task_type == "summarization":
                rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
                for i in tqdm(range(len(test_dataset))):
                    prompt = alpaca_prompt.format(test_dataset[i]['text'], "")
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=150, use_cache=True)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_response = self.extract_response(generated_text)
                    reference_summary = test_dataset[i]['summary']
                    rouge_scores = self.scorer.score(reference_summary, generated_response)
                    results.append({
                        'Input': test_dataset[i]['text'],
                        'Generated Summary': generated_response,
                        'Reference Summary': reference_summary,
                        'ROUGE-1': rouge_scores['rouge1'].fmeasure,
                        'ROUGE-2': rouge_scores['rouge2'].fmeasure,
                        'ROUGE-L': rouge_scores['rougeL'].fmeasure,
                    })
                return pd.DataFrame(results)
            elif self.task_type == "translation":
                bleu_scores = []
                for i in tqdm(range(len(test_dataset))):
                    prompt = alpaca_prompt.format(test_dataset[i]['indonesian'], "")
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to("cuda")
                    outputs = model.generate(**inputs, max_new_tokens=70, use_cache=True)
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_response = self.extract_response(generated_text)
                    reference = [self.stemmer.stem(word) for word in test_dataset[i]['english'].lower().split()]
                    hypothesis = [self.stemmer.stem(word) for word in generated_response.lower().split()]
                    bleu = sentence_bleu([reference], hypothesis, smoothing_function=self.smoothie)
                    results.append({
                        'Input': test_dataset[i]['indonesian'],
                        'Generated Translation': generated_response,
                        'Reference Translation': test_dataset[i]['english'],
                        'BLEU Score': bleu,
                    })
                return pd.DataFrame(results)
        else:
            raise ValueError("Model type not supported. Choose 'pegasus', 't5', or 'komodo'.")