import requests
import pandas as pd
from transformers import pipeline, BertForMaskedLM, BertTokenizerFast
import pprint

# Load the MatBERT model from the local directory
matbert_model_path = r"C:\Users\dmich\Desktop\ammonia-text-miner\MatBERT\matbert-base-cased"  

# Load the tokenizer and model
model = BertForMaskedLM.from_pretrained(matbert_model_path)
tokenizer = BertTokenizerFast.from_pretrained(matbert_model_path, do_lower_case=False)
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)
pprint.pprint(unmasker("Conventional [MASK] synthesis is used to fabricate material LiMn2O4."))