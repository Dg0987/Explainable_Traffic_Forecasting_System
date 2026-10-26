from fastapi.testclient import TestClient
import sys
sys.path.append('backend')
from app import app
import time
import numpy as np

client = TestClient(app)

models_to_test = ['lstm', 'tft']

for model in models_to_test:
    print(f'\n--- Testing /train ({model}) ---', flush=True)
    resp = client.post('/train', json={'model_type': model})
    print(f'Start Response: {resp.status_code} {resp.json()}', flush=True)

    # Wait for training to finish
    while True:
        time.sleep(5)
        status = client.get('/train/status').json()
        if status.get('status') == 'success':
            print(f'{model} Training finished: {status.get("message")}', flush=True)
            break
        elif status.get('error'):
            print(f'{model} Training error: {status}', flush=True)
            break
        else:
            print(f'{model} Training progress: {status.get("progress")}% epoch: {status.get("epoch")}', flush=True)

    print(f'\n--- Testing /predict ({model}) ---', flush=True)
    dummy_data = np.zeros((12, 8)).tolist()
    resp = client.post('/predict', json={'data': dummy_data, 'model_type': model})
    print(f'Predict Response: {resp.status_code} {resp.json()}', flush=True)
