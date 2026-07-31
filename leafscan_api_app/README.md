# LeafScan AI — REST API

## Setup
1. Place your trained model at: `models/leafscan_model.keras`
2. `pip install -r requirements.txt`
3. `python app.py` (runs on port 8000)

## POST /predict
```bash
curl -X POST http://localhost:8000/predict -F "image=@sample_leaf.jpg"
```
Response: `{"disease": "Brown Blight", "confidence": 91.2}`
