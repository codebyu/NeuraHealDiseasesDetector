import requests

# API Endpoint
url = "http://127.0.0.1:5000/predict"

# Send an Image to the API
files = {'image': open('test_image.jpg', 'rb')}  # Replace with actual image
response = requests.post(url, files=files)

# Print API Response
print("Response:", response.json())
