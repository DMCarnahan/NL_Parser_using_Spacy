import requests
import pandas as pd
import logging
from transformers import pipeline
from datasets import load_dataset
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import PyPDF2
import io
import os
from materials_entity_recognition import MatRecognition

# Load the tokenizer and model
model = MatRecognition()
tokenizer = model.get_tokenizer()  # Ensure tokenizer is correctly initialized

# Update the path to the BERT model
bert_model_path = r"C:\Users\dmich\Desktop\text-miner\MatBERT-synthesis-classifier"  # Replace with the correct path or repo ID

# Create the NER pipeline
ner_model = pipeline("ner", model=bert_model_path, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

def chunk_string(string, length):
    return [string[0+i:length+i] for i in range(0, len(string), length)]

def extract_parameters(abstract):
    chunks = chunk_string(abstract,512)

    size = "Unknown"
    reducing_agent = "Unknown"
    precursor = "Unknown"

    # print(ner_model)

    model = ner_model.model
    labels = model.config.id2label

    # # Print all possible entities
    # for id, label in labels.items():
    #     print(f"ID: {id}, Label: {label}")

    for i in range(0,len(chunks)):
        chunk = chunks[i]
        ner_results = ner_model(chunk)      
        # print(f"NER Results: {ner_results}")
        for entity in ner_results:
            print(f"Entity: {entity}")
            if entity['entity'] == 'SIZE':
                size = entity['word']
            elif entity['entity'] == 'REDUCING_AGENT':
                reducing_agent = entity['word']
            elif entity['entity'] == 'PRECURSOR':
                precursor = entity['word']
    
    return size, reducing_agent, precursor

def extract_text_from_local_pdf(file_path):
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ''.join(page.extract_text() or '' for page in reader.pages)
        return text
    except Exception as e:
        logging.error(f"Error reading local PDF file: {e}")
        return None

def process_result(result):
    logging.debug(f"Raw result: {json.dumps(result, indent=2)}")
    try:
        title = result.get('title', 'Unknown')
        doi = result.get('doi', 'Unknown')
        open_access = result.get('openaccess', 'false').lower() == 'true'
        
        # Check URL for hints of open access
        pdf_url = None
        for url in result.get('url', []):
            if url.get('format') == 'pdf' and 'openurl/pdf' in url.get('value', ''):
                pdf_url = url['value']
                open_access = True
                break
        
        logging.debug(f"Title: {title}, DOI: {doi}, Open Access: {open_access}, PDF URL: {pdf_url}")
        
        return {
            "title": title,
            "doi": doi,
            "open_access": open_access,
            "pdf_url": pdf_url if open_access else None
        }
    except Exception as e:
        logging.error(f"Error processing result: {e}")
        return None

def save_dataset(data, filename="extracted_parameters.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")

def load_dataset(filename="extracted_parameters.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        logging.info(f"Data loaded from {filename}")
        return df
    else:
        logging.error(f"File {filename} does not exist")
        return None

def main():
    # Example local PDF file path
    local_pdf_path = r"C:\Users\dmich\Downloads\s41467-024-45066-9.pdf"
    
    # Extract text from the local PDF
    full_text = extract_text_from_local_pdf(local_pdf_path)
    
    if full_text:
        logging.debug(f"Extracted text from local PDF: {full_text[:500]}...")  # Log the first 500 characters
        # Process the extracted text
        size, reducing_agent, precursor = extract_parameters(full_text)
        logging.info(f"Extracted Parameters - Catalyst: {size}, Reducing_agent: {reducing_agent}, Precursor: {precursor}")
    else:
        logging.error("Failed to extract text from the local PDF.")

    # Comment out or remove the code related to searching Springer
    # api_key = "56dd87b4ca55f70891d98ba04dd9c042"
    # query = "gold nanoparticles"
    # search_results = search_springer_nature(query, api_key)

    # logging.debug(f"Number of records: {len(search_results['records'])}")
    # logging.debug(json.dumps(search_results['records'], indent=2))

    # if not search_results:
    #     logging.error("No search results found.")
    #     logging.debug(json.dumps(search_results, indent=2))  # Log raw API response
    #     return
    
    # if 'records' not in search_results:
    #     logging.error("'records' key is missing in the search results.")
    #     logging.debug(json.dumps(search_results, indent=2))  # Log raw API response
    #     return

    # if not search_results['records']:
    #     logging.error("'records' key is present but contains no data.")
    #     logging.debug(json.dumps(search_results, indent=2))  # Log raw API response
    #     return
    
    # logging.debug(f"Search results: {search_results['records']}")

    # data = []
    # for result in search_results.get('records', []):
    #     logging.debug(f"Raw result: {json.dumps(result, indent=2)}")
    #     title = result.get('title', 'Unknown')
    #     doi = result.get('doi', 'Unknown')
    #     open_access = result.get('openaccess', 'false').lower() == 'true'
        
    #     data.append({
    #         "title": title,
    #         "doi": doi,
    #         "open_access": open_access
    #     })

    # logging.debug(f"Final Data Collected: {data}")

    # if data:
    #     logging.info(f"{len(data)} records collected. Sample: {data[0]}")
    # else:
    #     logging.warning("No records were added to the data list.")

    # # Save the extracted parameters to a dataset
    # if data:
    #     df = pd.DataFrame(data)
    #     df.to_csv("extracted_parameters.csv", index=False)
    #     logging.info(f"{len(data)} records saved successfully with PDF URLs included.")
    # if data:
    #     save_dataset(data)
    # else:
    #     logging.warning("No valid data to save. CSV was not created.")

    # for result in search_results['records']:
    #     logging.debug(f"Processing record: {json.dumps(result, indent=2)}")
    #     process_result(result)
    # logging.debug(f"Final Collected Data: {data}")

if __name__ == "__main__":
    main()
