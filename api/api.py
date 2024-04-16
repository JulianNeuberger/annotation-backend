from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import stanza
from api.annotation_loader import read_annotated_documents_from_json

from api.new_data.pet import PetDictExporter
from api.utils import adjust_data_for_data_model, convert_stanza_to_dataclass, get_predictions_for_input, load_document, retrain_model, save_data_to_json
from data import model

app = Flask(__name__)
CORS(app, resources={r'/*': {"origins": '*'}})

def create_nlp_pipeline():
    return stanza.Pipeline(lang='en', processors={'tokenize': 'spacy'})

nlp = None


@app.route('/annotate', methods=['POST'])
def annotate_text():
    global nlp
    if nlp is None:
        nlp = create_nlp_pipeline()

    data = request.json
    option = data['option']
    text = data['text']

    doc = nlp(text)
    dataclass_doc = convert_stanza_to_dataclass(doc)

    if option == 'NoAI':
        return jsonify(dataclass_doc.to_json_serializable())
    
    elif option == 'BadAI':
        result: model.Document = get_predictions_for_input(dataclass_doc, 'bad model')
        
        formatted_result = result.to_json_serializable()

        return jsonify(formatted_result)
    
    elif option == 'AverageAI':
        result: model.Document = get_predictions_for_input(dataclass_doc, 'average model')
        
        formatted_result = result.to_json_serializable()

        return jsonify(formatted_result)
    
    else:
        result: model.Document = get_predictions_for_input(dataclass_doc, 'good model')
        
        formatted_result = result.to_json_serializable()

        return jsonify(formatted_result)


@app.route('/retrain', methods=['POST'])
def retrain():
    modified_data = request.json

    save_data_to_json(adjust_data_for_data_model(modified_data), 'api/annotated_data.json')

    print("Documents saved to json file")

    number_of_docs = retrain_model()

    return jsonify({"message": f"Completed retraining successful! Number of documents: {number_of_docs}"})


@app.route('/get-document', methods=['GET'])
def get_document():
    document_path = "/Users/jannic/Developer/Bachelorarbeit/pet-baselines/api/new_data/all.new.jsonl"
    document = load_document(document_path)

    exporter = PetDictExporter()
    document_dict = exporter.export_document(document)

    return jsonify(document_dict)


@app.route('/test-cors', methods=['GET'])
def test_cors():
    return jsonify({"message": "CORS is working"})

if __name__ == '__main__':
    nlp = create_nlp_pipeline()
    app.run(port=5001)