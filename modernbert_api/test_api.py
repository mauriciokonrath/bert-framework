import requests

url = "http://127.0.0.1:8000/predict"
data = {"text": "The capital of France is [MASK]."}

response = requests.post(url, json=data)

print("Status code:", response.status_code)
print("Response:", response.json())
