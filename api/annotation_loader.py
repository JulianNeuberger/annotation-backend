import json
import typing

from data import model


def read_annotated_documents_from_json(file_path: str) -> typing.List[model.Document]:
    documents = []
    with open(file_path, 'r', encoding='utf8') as f:
        json_data = json.load(f) 

    for item in json_data:
        document = _read_document_from_json(item)
        documents.append(document)

    return documents


def _read_document_from_json(json_data: typing.Dict) -> model.Document:
    text = json_data.get('text', '')
    name = json_data.get('name', '')

    mentions = _read_mentions_from_json(json_data.get('mentions', []))
    entities = _read_entities_from_json(json_data.get('entities', []))
    relations = _read_relations_from_json(json_data.get('relations', []))
    sentences = _read_sentences_from_json(json_data.get('sentences', []))

    return model.Document(
        text=text,
        name=name,
        sentences=sentences,
        mentions=mentions,
        entities=entities,
        relations=relations
    )


def _read_sentences_from_json(json_sentences: typing.List[typing.Dict]) -> typing.List[model.Sentence]:
    sentences = []
    for json_sentence in json_sentences:
        tokens = [model.Token(
            text=token['text'],
            index_in_document=token['index_in_document'],
            pos_tag=token['pos_tag'],
            bio_tag=token['bio_tag'],
            sentence_index=token['sentence_index']
        ) for token in json_sentence['tokens']]
        sentence = model.Sentence(tokens=tokens)
        sentences.append(sentence)
    return sentences


def _read_mentions_from_json(json_mentions: typing.List[typing.Dict]) -> typing.List[model.Mention]:
    mentions = []
    for json_mention in json_mentions:
        mention = model.Mention(
            ner_tag=json_mention['tag'],
            sentence_index=json_mention['sentence_index'],
            token_indices=json_mention['token_indices']
        )
        mentions.append(mention)
    return mentions


def _read_entities_from_json(json_entities: typing.List[typing.Dict]) -> typing.List[model.Entity]:
    entities = []
    for json_entity in json_entities:
        entity = model.Entity(mention_indices=json_entity['mention_indices'])
        entities.append(entity)
    return entities


def _read_relations_from_json(json_relations: typing.List[typing.Dict],
                              ) -> typing.List[model.Relation]:
    relations = []
    for json_relation in json_relations:
        relation = model.Relation(
            head_entity_index=json_relation['head'],
            tail_entity_index=json_relation['tail'],
            tag=json_relation['tag'],
            evidence=json_relation['evidence']
        )
        relations.append(relation)
    return relations