import json

from data import model


def dump_document_to_json(document: model.Document) -> str:
    as_dict = {
        "id": document.name,
        "text": document.text,
        "tokens": [
            {
                "text": t.text,
                "stanza_pos": t.pos_tag,
                "ner": t.bio_tag,
                "sentence_id": t.sentence_index,
            }
            for t in document.tokens
        ],
        "mentions": [
            {
                "ner": m.ner_tag,
                "sentence_id": m.sentence_index,
                "token_indices": m.token_indices,
            }
            for m in document.mentions
        ],
        "entities": [{"mention_indices": e.mention_indices} for e in document.entities],
        "relations": [
            {
                "head_entity": {
                    "mention_indices": document.entities[
                        r.head_entity_index
                    ].mention_indices
                },
                "tail_entity": {
                    "mention_indices": document.entities[
                        r.tail_entity_index
                    ].mention_indices
                },
                "type": r.tag,
                "evidence": r.evidence,
            }
            for r in document.relations
        ],
    }

    return json.dumps(as_dict)
