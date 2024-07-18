# Backend for Assisted Process Information Annotation

## Installation

Tested for Python 3.9

```ssh
pip install -r requirements.txt
python -m spacy download en_core_web_sm
FLASK_APP=/app/api/api.py flask run --host=0.0.0.0 --port=5000
```

Backend server is now running at http://127.0.0.1:5000

If the frontend tries to store annotation results, this server will try to do so in director `/results`.
You can change this in `api/api.py` line 66.

## Preprocess an input document

POST request to http://127.0.0.1:5000/annotate

Body should contain the following json structure.

```json
{
    "text": "The Customer Service Representative sends a Mortgage offer to the customer and waits for a reply . If the customer calls or writes back declining the mortgage , the case details are updated",
    "option": "GoodAI"
}
```

## Retrieve results

GET request to http://127.0.0.1:5000/annotate