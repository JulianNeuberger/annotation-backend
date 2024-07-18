import json
import os
import typing

import nltk

from data import model

try:
    nltk.data.find("averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")


def read_names(filename) -> typing.List[typing.List[str]]:
    data = open(filename).readlines()
    data = [item.strip().split("\t") for item in data]
    return data


def read_documents_from_json_file(file_path: str) -> typing.List[model.Document]:
    documents = []
    with open(file_path, "r", encoding="utf8") as f:
        for json_line in f:
            json_data = json.loads(json_line)
            documents.append(read_document_from_json(json_data))

    return documents


def read_documents_from_folder(folder_path: str) -> typing.List[model.Document]:
    documents = []
    print(f"reading from {folder_path}")
    for file_name in os.listdir(folder_path):
        print(file_name)
        if not os.path.isfile(os.path.join(folder_path, file_name)):
            continue
        file_path = os.path.join(folder_path, file_name)
        read_documents = read_documents_from_json_file(file_path)
        print(len(read_documents))
        documents.extend(read_documents)
    return documents


def read_document_from_json(json_data: typing.Dict) -> model.Document:
    mentions = _read_mentions_from_json(json_data["mentions"])
    entities = _read_entities_from_json(json_data["entities"])
    relations = _read_relations_from_json(json_data["relations"])
    tokens = _read_tokens_from_json(json_data["tokens"])
    return model.Document(
        name=json_data["name"],
        text=json_data["text"],
        id=json_data["id"],
        category=json_data["category"],
        tokens=tokens,
        mentions=mentions,
        relations=relations,
        entities=entities,
    )


def _read_tokens_from_json(
    json_tokens: typing.List[typing.Dict],
) -> typing.List[model.Token]:
    tokens = []
    for i, json_token in enumerate(json_tokens):
        tokens.append(
            model.Token(
                text=json_token["text"],
                pos_tag=json_token["posTag"],
                index_in_document=i,
                sentence_index=json_token["sentenceIndex"],
            )
        )
    return tokens


def _read_mentions_from_json(
    json_mentions: typing.List[typing.Dict],
) -> typing.List[model.Mention]:
    mentions = []
    for json_mention in json_mentions:
        mention = _read_mention_from_json(json_mention)
        mentions.append(mention)
    return mentions


def _read_entities_from_json(
    json_entities: typing.List[typing.Dict],
) -> typing.List[model.Entity]:
    entities = []
    for json_entity in json_entities:
        entity = _read_entity_from_json(json_entity)
        entities.append(entity)
    return entities


def _read_mention_from_json(json_mention: typing.Dict) -> model.Mention:
    return model.Mention(
        ner_tag=json_mention["type"],
        token_document_indices=json_mention["tokenDocumentIndices"],
    )


def _read_entity_from_json(json_entity: typing.Dict) -> model.Entity:
    return model.Entity(json_entity["mentionIndices"])


def _read_relations_from_json(
    json_relations: typing.List[typing.Dict],
) -> typing.List[model.Relation]:
    relations = []
    for json_relation in json_relations:
        head_mention_index = json_relation["headMentionIndex"]
        tail_mention_index = json_relation["tailMentionIndex"]

        relations.append(
            model.Relation(
                head_mention_index=head_mention_index,
                tail_mention_index=tail_mention_index,
                tag=json_relation["type"],
            )
        )

    return relations
