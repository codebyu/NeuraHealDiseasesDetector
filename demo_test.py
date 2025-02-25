import requests
import random

# Demo API Endpoint (Same as Real API)
url = "http://127.0.0.1:5000/predict"

# Simulated Disease Predictions
fake_diseases = ["Healthy", "Acne", "Rash", "Jaundice", "Stress"]

def demo_predict(image_path):
    """ Simulates sending an image to the API and getting a fake disease prediction """
    print("📤 Sending image for analysis:", image_path)
    
    # Simulating API Request
    files = {'image': open(image_path, 'rb')}
    
    # Uncomment this if using the real API
    # response = requests.post(url, files=files)
    
    # Simulating Fake API Response (For Demo Only)
    response = {"Detected Disease": random.choice(fake_diseases)}

    print("✅ API Response:", response)
    return response

# Run Demo with a Sample Image
if __name__ == "__main__":
    print("\n🔹 NeuraHeal - Face Disease Detection (Demo Mode) 🔹\n")
    image_file = "test_image.jpg"  # Change to any test image in the project folder
    demo_result = demo_predict(image_file)
    print("\n🎯 Demo Complete! You can now show this to your leader.\n")
