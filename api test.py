import requests
import json
import logging

api_key = "56dd87b4ca55f70891d98ba04dd9c042"
query = "ammonia synthesis"
url = f"https://api.springernature.com/meta/v2/json?q={query}&api_key={api_key}"

response = requests.get(url)
if response.status_code == 200:
    try:
        result = response.json()
        logging.debug(json.dumps(result, indent=2))
        
        if 'records' in result:
            print(f"Number of records: {len(result['records'])}")
            print(json.dumps(result['records'], indent=2))
        else:
            print("'records' key is missing in the API response")
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
else:
    print(f"API request failed with status code {response.status_code}")
