import typing

import data


def decode_predictions(
    document: data.Document, predictions: typing.List[typing.List[str]]
) -> data.Document:
    print(predictions)
    assert len(document.sentences) == len(predictions)

    decoded_document: data.Document = data.Document(
        name=document.name,
        text=document.text,
        category=document.category,
        id=document.id,
    )

    for sent_id, (sentence, predicted_tags) in enumerate(
        zip(document.sentences, predictions)
    ):
        current_mention: typing.Optional[data.Mention] = None
        for token, bio_tag in zip(sentence, predicted_tags):
            current_token = data.Token(
                text=token.text,
                pos_tag=token.pos_tag,
                index_in_document=token.index_in_document,
                sentence_index=sent_id,
            )
            decoded_document.tokens.append(current_token)

            bio_tag = bio_tag.strip()
            tag = bio_tag.split("-", 1)[-1]

            is_entity_start = bio_tag.startswith("B-")

            should_finish_entity = is_entity_start or tag == "O"

            if should_finish_entity and current_mention is not None:
                decoded_document.mentions.append(current_mention)
                current_mention = None

            if is_entity_start:
                current_mention = data.Mention(ner_tag=tag)

            if current_mention is not None:
                current_mention.token_document_indices.append(
                    current_token.index_in_document
                )

        if current_mention is not None:
            decoded_document.mentions.append(current_mention)
    return decoded_document
