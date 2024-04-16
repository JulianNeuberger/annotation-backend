import typing

import data


def decode_predictions(document: data.Document,
                       predictions: typing.List[typing.List[str]]) -> data.Document:
    assert len(document.sentences) == len(predictions)

    decoded_document: data.Document = data.Document(name=document.name, text=document.text)

    for sent_id, (sentence, predicted_tags) in enumerate(zip(document.sentences, predictions)):
        decoded_sentence = data.Sentence()
        current_mention: typing.Optional[data.Mention] = None
        for token_index, (token, bio_tag) in enumerate(zip(sentence.tokens, predicted_tags)):
            current_token = data.Token(
                text=token.text,
                pos_tag=token.pos_tag,
                bio_tag=bio_tag,
                index_in_document=token.index_in_document,
                sentence_index=sent_id
            )
            decoded_sentence.tokens.append(current_token)

            bio_tag = bio_tag.strip()
            tag = bio_tag.split('-', 1)[-1]
            is_entity_start = bio_tag.startswith('B-')

            should_finish_entity = is_entity_start or tag == 'O'

            if should_finish_entity and current_mention is not None:
                decoded_document.mentions.append(current_mention)
                current_mention = None

            if is_entity_start:
                current_mention = data.Mention(ner_tag=tag, sentence_index=sent_id)

            if current_mention is not None:
                current_mention.token_indices.append(token_index)

        if current_mention is not None:
            decoded_document.mentions.append(current_mention)

        decoded_document.sentences.append(decoded_sentence)

    return decoded_document
