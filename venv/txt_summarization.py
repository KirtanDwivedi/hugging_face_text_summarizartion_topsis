import csv
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
from bert_score import score

# Load dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train']
ARTICLE = train_data[0]['article']
summary = train_data[0]['highlights']

# Models initialization --------------------------------------------------------
# 1. BART
model_name1 = "facebook/bart-base"
model1 = AutoModelForSeq2SeqLM.from_pretrained(model_name1)
tokenizer1 = AutoTokenizer.from_pretrained(model_name1)
summarizer1 = pipeline("summarization", model=model1, tokenizer=tokenizer1)
summary1 = summarizer1(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)

# 2. PEGASUS
model_name2 = "google/pegasus-xsum"
model2 = AutoModelForSeq2SeqLM.from_pretrained(model_name2)
tokenizer2 = AutoTokenizer.from_pretrained(model_name2)
summarizer2 = pipeline("summarization", model=model2, tokenizer=tokenizer2)
summary2 = summarizer2(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)

# 3. T5
model_name3 = "t5-small"
model3 = AutoModelForSeq2SeqLM.from_pretrained(model_name3)
tokenizer3 = AutoTokenizer.from_pretrained(model_name3)
summarizer3 = pipeline("summarization", model=model3, tokenizer=tokenizer3)
summary3 = summarizer3(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)

# 4. LongT5
model_name4 = "google/long-t5-tglobal-base"
model4 = AutoModelForSeq2SeqLM.from_pretrained(model_name4)
tokenizer4 = AutoTokenizer.from_pretrained(model_name4)
summarizer4 = pipeline("summarization", model=model4, tokenizer=tokenizer4)
summary4 = summarizer4(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)

# 5. LED
model_name5 = "allenai/led-base-16384"
model5 = AutoModelForSeq2SeqLM.from_pretrained(model_name5)
tokenizer5 = AutoTokenizer.from_pretrained(model_name5)
summarizer5 = pipeline("summarization", model=model5, tokenizer=tokenizer5)
summary5 = summarizer5(ARTICLE, max_length=130, min_length=30, do_sample=False, truncation=True)

# Evaluation metrics -----------------------------------------------------------
rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")

summaries = [summary1[0]['summary_text'], 
             summary2[0]['summary_text'], 
             summary3[0]['summary_text'], 
             summary4[0]['summary_text'], 
             summary5[0]['summary_text']]

domains = ['politics', 'technology', 'health', 'entertainment']

# CSV writing -------------------------------------------------------------------
for domain in domains:
    with open(f"{domain}.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Alternative', 'Criterion1', 'Criterion2', 'Criterion3', 'Criterion4', 'Criterion5', 'Criterion6'])
        
        for i, generated_summary in enumerate(summaries):
            # Get domain-specific reference
            domain_data = train_data.filter(lambda example: domain in example['article'].lower())
            if len(domain_data) == 0:
                continue  # Skip if no relevant articles are found
            
            reference_summary = domain_data[0]['highlights']
            
            # Calculate metrics
            f1 = rouge.compute(predictions=[generated_summary], references=[reference_summary])
            f2 = bleu.compute(predictions=[generated_summary], references=[[reference_summary]])  # Fixed format
            f3 = meteor.compute(predictions=[generated_summary], references=[reference_summary])
            F1 = score([generated_summary], [reference_summary], lang="en")
            f4 = F1[2].mean().item()
            
            # Length-based features
            generated_length = len(generated_summary.split())
            reference_length = len(reference_summary.split())
            f5 = abs(generated_length - reference_length)
            
            original_article_length = len(domain_data[0]['article'].split())
            f6 = generated_length / original_article_length
            
            writer.writerow([f'M{i+1}', f1['rouge1'], f2['bleu'], f3['meteor'], f4, f5, f6])
