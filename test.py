import requests
import json

# Define the API endpoint
api_url = "http://127.0.0.1:8000/predict"

# Define the test data (same as above)
test_data = {
    "features": [350.00, 1, 23, 116, 1016, 1, 30]
}

# Convert the data to JSON format
data = json.dumps(test_data)

# Define the headers (if needed)
headers = {'Content-Type': 'application/json'}

# Make the POST request
response = requests.post(api_url, data=data, headers=headers)

# Check the response
if response.status_code == 200:
    print("API call successful!")
    print("Response:", response.json())
else:
    print("API call failed with status code:", response.status_code)
