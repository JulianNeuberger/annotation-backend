import json
import os
import typing

import stanza
from flask import Flask, request, jsonify
from flask_cors import CORS

from api.utils import convert_stanza_to_dataclass, get_predictions_for_input, load_document

app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})


def create_nlp_pipeline() -> stanza.Pipeline:
    return stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})


nlp: typing.Optional[stanza.Pipeline] = None


@app.route('/annotate', methods=['POST'])
def annotate_text():
    global nlp
    if nlp is None:
        nlp = create_nlp_pipeline()

    data = request.json
    option = data['option']
    text = data['text']

    stanza_document = nlp(text)
    document = convert_stanza_to_dataclass(stanza_document)

    ai_options = {
        "BadAI": "bad model",
        "AverageAI": "average model",
        "GoodAI": "good model"
    }

    if option in ai_options:
        document = get_predictions_for_input(document, ai_options[option])

    return jsonify(document.to_json_serializable())


@app.route('/get-document', methods=['GET'])
def get_document():
    document_path = "/Users/jannic/Developer/Bachelorarbeit/pet-baselines/api/new_data/all.new.jsonl"
    document = load_document(document_path)
    document_dict = document.to_json_serializable()
    return jsonify(document_dict)


@app.route('/test-cors', methods=['GET'])
def test_cors():
    return jsonify({"message": "CORS is working"})


@app.route("/results", methods=["POST"])
def save_results():
    result_json = request.json
    user_id = result_json["userId"]
    task_id = result_json["taskId"]

    results_folder = f"/results/{user_id}"
    os.makedirs(results_folder, exist_ok=True)
    results_file = f"{results_folder}/{task_id}.json"

    with open(results_file, "w", encoding="utf8") as f:
        json.dump(result_json, f)

    return ""


@app.route("/results", methods=["GET"])
def load_results():
    response = {}
    for user_id in os.listdir("/results"):
        response[user_id] = {}

        for task_results in os.listdir(f"/results/{user_id}"):
            with open(f"/results/{user_id}/{task_results}", "r", encoding="utf8") as f:
                result = json.load(f)

            task_id = task_results.replace(".json", "")
            response[user_id][task_id] = result

    return response


if __name__ == '__main__':
    nlp = create_nlp_pipeline()
    app.run(port=5001)
