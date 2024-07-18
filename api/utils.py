import json
import os
import typing

import data
import pipeline
from api.isolated_piplines import IsolatedPipeline
from data.model import Document, Token


def convert_stanza_to_dataclass(stanza_doc):
    tokens: typing.List[Token] = []
    for i, stanza_sentence in enumerate(stanza_doc.sentences):
        for stanza_token in stanza_sentence.tokens:
            for word in stanza_token.words:
                token = Token(
                    text=word.text,
                    index_in_document=len(tokens),
                    pos_tag=word.xpos if word.xpos else '',
                    sentence_index=i
                )
                tokens.append(token)

    document_text = stanza_doc.text if stanza_doc.text else ''
    return Document(
        text=document_text,
        id="",
        category="",
        name="",
        tokens=tokens,
        mentions=[],
        entities=[],
        relations=[]
    )


def get_predictions_for_input(document: data.Document, model_type: str) -> data.Document:
    ner_pd = IsolatedPipeline(name=f'complete-pipeline', steps=[
        pipeline.CrfMentionEstimatorStep(name=f'crf mention extraction {model_type}'),
        pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                 resolved_tags=['Actor', 'Activity Data'],
                                                 cluster_overlap=.5,
                                                 mention_overlap=.5,
                                                 ner_strategy='frequency'),
        pipeline.CatBoostRelationExtractionStep(name=f'cat-boost re {model_type}', use_pos_features=False,
                                                context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                depth=8, class_weighting=0, num_passes=1)])
    pipeline_result = ner_pd.run(test_documents=[document], ground_truth_documents=[document], training_only=False)

    assert pipeline_result.step_results, "No results in pipeline_result.step_results"
    last_step_key = list(pipeline_result.step_results.keys())[-1]
    last_step_result = pipeline_result.step_results[last_step_key]
    return last_step_result.predictions[0]


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


def load_document(file_path) -> typing.Optional[data.Document]:
    documents = data.loader.read_documents_from_json_file(file_path)
    return documents[4] if documents else None
