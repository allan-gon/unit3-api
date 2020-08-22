from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_valid_input():
    """Return 200 Success when input is valid."""
    response = client.post(
        '/predict',
        json={
            'title': 'First string',
            'body': 'Another string',
        }
    )
    body = response.json()
    assert response.status_code == 200
    assert body['prediction'] in ['r/AMA', 'r/Politics', 'r/PCMasterrace']


def test_invalid_input():
    """Return 422 Validation Error when given non-strings"""
    response = client.post(
        '/predict',
        json={
            'title': -3.14,
            'body': 'this is a string',
        }
    )
    body = response.json()
    assert response.status_code == 422
    assert 'title' in body['detail'][0]['loc']
