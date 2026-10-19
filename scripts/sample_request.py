"""Send a documented sample prediction request to a running local API."""
import json
import urllib.request

payload = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
           "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
request = urllib.request.Request("http://localhost:8000/predict", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
