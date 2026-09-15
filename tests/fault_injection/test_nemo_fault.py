import requests
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get('NVIDIA_API_KEY', '').strip('"')
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def call_model(model):
    data = {
        'model': model,
        'messages': [{'role': 'user', 'content': 'Write a 2 sentence explanation of feature engineering.'}],
        'max_tokens': 100
    }
    try:
        r = requests.post('https://integrate.api.nvidia.com/v1/chat/completions', headers=headers, json=data, timeout=10)
        if r.status_code == 200:
            return r.json()['choices'][0]['message']['content']
        return f'Error {r.status_code}: {r.text}'
    except Exception as e:
        return str(e)

import pytest

def test_nemotron_lightning():
    if not api_key:
        pytest.skip("No NVIDIA_API_KEY")
    result = call_model('nvidia/nemotron-3.5-lightning-30b-a3b')
    assert result, "Response should not be empty"

def test_nemotron_ultra():
    if not api_key:
        pytest.skip("No NVIDIA_API_KEY")
    result = call_model('nvidia/nemotron-3-ultra-550b-a55b')
    assert result, "Response should not be empty"
