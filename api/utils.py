import json
import os
import typing
import stanza
from api.annotation_loader import read_annotated_documents_from_json
from api.new_data.pet import NewPetFormatImporter
import data
from data import model
from data.loader import _read_entities_from_json, _read_mentions_from_json, _read_relations_from_json

from data.model import Document, Entity, Mention, Relation, Sentence, Token
from api.isolated_piplines import IsolatedPipline
import pipeline


def convert_stanza_to_dataclass(stanza_doc):
    sentences = []
    token_index = 0 

    for i, stanza_sentence in enumerate(stanza_doc.sentences):
        sentence = Sentence(tokens=[])

        for stanza_token in stanza_sentence.tokens:
            for word in stanza_token.words:
                token = Token(
                    text=word.text,
                    index_in_document=token_index,
                    pos_tag=word.xpos if word.xpos else '',
                    bio_tag='O', 
                    sentence_index=i
                )
                sentence.tokens.append(token)
                token_index += 1

        sentences.append(sentence)

    document_text = stanza_doc.text if stanza_doc.text else ''
    return Document(
        text=document_text,
        name="Document Name",  
        sentences=sentences,
        mentions=[],
        entities=[],
        relations=[]
    )


def get_predictions_for_input(input_text: data.Document, model_type: str):
    ner_pd = IsolatedPipline(name=f'complete-pipeline', steps=[
                pipeline.CrfMentionEstimatorStep(name=f'crf mention extraction {model_type}'),
                pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
                pipeline.CatBoostRelationExtractionStep(name=f'cat-boost re {model_type}', use_pos_features=False,
                                                        context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                        depth=8, class_weighting=0, num_passes=1)])
    pipeline_result = ner_pd.run(test_documents=[input_text], ground_truth_documents=[input_text], training_only=False)
    
    last_step_result = None
    if pipeline_result.step_results:
        last_step_key = list(pipeline_result.step_results.keys())[-1]
        last_step_result = pipeline_result.step_results[last_step_key]
    
    else:
        print("No results in pipeline_result.step_results")
    
    return last_step_result[0]


def retrain_model():
    training_set = connect_training_data()

    print("Train CRF model")
    ner_pd = IsolatedPipline(name=f'mention-extraction model', steps=[
            pipeline.CrfMentionEstimatorStep(name=f'crf mention extraction')])
    ner_pd.run(train_documents=training_set, training_only=True)

    print("Train CatBoost model")
    catboost_pd = IsolatedPipline(name='catboost', steps=[pipeline.CatBoostRelationExtractionStep(name=f'cat-boost re', use_pos_features=False,
                                                    context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)])
    catboost_pd.run(train_documents=training_set, training_only=True)

    print("Models retrained")

    return len(training_set)


def connect_training_data():
    train_data = data.loader.read_documents_from_json(f'./jsonl/fold_{4}/train.json')
    annotated_train_data = read_annotated_documents_from_json('api/annotated_data.json')
    combined_train_data = train_data + annotated_train_data
    
    return combined_train_data


def save_data_to_json(modified_data: Document, file_path: str):
    new_data = modified_data.to_json_serializable()
    
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(new_data)
    
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

    
def adjust_data_for_data_model(modified_data: dict) -> Document:
    document = Document(text=modified_data.get('text', ''), name=modified_data.get('name', ''))

    for sentence_data in modified_data.get('sentences', []):
        sentence = Sentence(tokens=[
            Token(
                text=token['text'],
                index_in_document=token['index_in_document'],
                pos_tag=token['pos_tag'],
                bio_tag=token['bio_tag'],
                sentence_index=token['sentence_index']
            ) for token in sentence_data['tokens']
        ])
        document.sentences.append(sentence)

    for mention_data in modified_data.get('mentions', []):
        mention = Mention(
            ner_tag=mention_data['tag'],
            sentence_index=mention_data['sentence_index'],
            token_indices=mention_data['token_indices']
        )
        document.mentions.append(mention)

    for entity_data in modified_data.get('entities', []):
        entity = Entity(mention_indices=entity_data['mention_indices'])
        document.entities.append(entity)

    for relation_data in modified_data.get('relations', []):
        relation = Relation(
            head_entity_index=relation_data['head'],
            tail_entity_index=relation_data['tail'],
            tag=relation_data['tag'],
            evidence=relation_data['evidence']
        )
        document.relations.append(relation)

    return document

def load_document(file_path):
    importer = NewPetFormatImporter(file_path)
    documents = importer.do_import()
    return documents[4] if documents else None