import logging
import json

# Set logging to DEBUG
logging.basicConfig(level=logging.DEBUG)

def process_result(result):
    logging.debug(f"Raw result: {json.dumps(result, indent=2)}")
    try:
        title = result.get('title', 'Unknown')
        doi = result.get('doi', 'Unknown')
        open_access = result.get('openaccess', 'false').lower() == 'true'
        
        logging.debug(f"Title: {title}, DOI: {doi}, Open Access: {open_access}")
        
        if open_access and doi != 'Unknown':
            logging.debug(f"Valid result: {title}")
            return {
                "title": title,
                "doi": doi,
                "open_access": open_access
            }
        else:
            logging.debug(f"Skipping non-open-access record or missing DOI: {title}")
            return None
    except Exception as e:
        logging.error(f"Error processing result: {e}")
        return None


# Test with your provided record
test_record = {
    "contentType": "Article",
    "identifier": "doi:10.1007/s11664-024-11601-z",
    "title": "Synthesis of MWCNTs/Si_3N_4 Nanocomposites Via Click Chemistry",
    "openaccess": "false",
    "doi": "10.1007/s11664-024-11601-z"
}

result = process_result(test_record)
print("Processed Result:", result)