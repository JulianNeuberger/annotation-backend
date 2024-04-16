import itertools
import random
import re
import typing

import numpy as np
import torch
import tqdm
from torch import nn
import datasets

import data
import transformers
from transformers import modeling_outputs

from relations import sampler


class NeuralRelationEstimator:
    def __init__(self, checkpoint: str,
                 entity_tags: typing.List[str], relation_tags: typing.List[str],
                 negative_sampling_rate: float = 40):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.negative_sampling_rate = negative_sampling_rate

        self.entity_tags = entity_tags
        self.no_relation_tag = 'NO REL'
        self.relation_tags = [r.lower() for r in relation_tags] + [self.no_relation_tag]
        self.config: transformers.LongformerConfig = transformers.AutoConfig.from_pretrained(checkpoint,
                                                                                             output_attentions=True,
                                                                                             output_hidden_states=True)
        self.tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint,
                                                                                                      config=self.config)
        self.feature_extractor = transformers.AutoModel.from_pretrained(checkpoint, config=self.config)
        new_tokens = ['[HEAD]', '[TAIL]']
        for entity_tag in entity_tags:
            new_tokens.append(opening_entity_tag(entity_tag))
            new_tokens.append(closing_entity_tag(entity_tag))

        # TODO: should we check that the tags are not already in the vocab?
        self.tokenizer.add_tokens(new_tokens)
        self.feature_extractor.resize_token_embeddings(len(self.tokenizer))

        self.classifier = RelationExtractionModel(self.feature_extractor, self.config.hidden_size, self.relation_tags)

        self.classifier.to(self.device)

    def train(self, documents: typing.List[data.Document]) -> 'NeuralRelationEstimator':
        train_args = transformers.TrainingArguments(
            output_dir='./',
            per_gpu_train_batch_size=1,
            fp16=True,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True
        )
        train_dataset = documents_to_dataset(documents, self.relation_tags, self.no_relation_tag,
                                             self.tokenizer, self.negative_sampling_rate)
        trainer = transformers.Trainer(
            args=train_args,
            model=self.classifier,
            train_dataset=train_dataset.select([1])
        )
        trainer.train()
        return self

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        eval_args = transformers.TrainingArguments(
            output_dir='./',
            per_gpu_eval_batch_size=4,
            fp16=True
        )
        eval_dataset = all_combinations_dataset_from_documents(documents, self.tokenizer)
        trainer = transformers.Trainer(
            args=eval_args,
            model=self.classifier
        )
        output = trainer.predict(eval_dataset)
        print(type(output))
        return self._parse_predictions(documents, eval_dataset, output.predictions)

    def _parse_predictions(self, documents: typing.List[data.Document],
                           dataset: datasets.Dataset, predictions: np.ndarray) -> typing.List[data.Document]:
        assert predictions.shape[0] == dataset.num_rows

        ret: typing.Dict[str, data.Document] = {
            doc.name: doc.copy() for doc in documents
        }
        no_rel_index = self.relation_tags.index(self.no_relation_tag)

        for data_row, prediction in zip(dataset, predictions):
            doc_id = data_row['doc_id']

            predicted_class_index = np.argmax(prediction)
            if predicted_class_index == no_rel_index:
                continue

            predicted_class = self.relation_tags[predicted_class_index]

            document = ret[doc_id]
            head_entity_index = document.entity_index_for_mention(document.mentions[data_row['head_mention_index']])
            tail_entity_index = document.entity_index_for_mention(document.mentions[data_row['tail_mention_index']])
            relation = data.Relation(
                head_entity_index=head_entity_index,
                tail_entity_index=tail_entity_index,
                tag=predicted_class,
                evidence=[]
            )
            if document.contains_relation(relation):
                continue
            document.relations.append(relation)

        return list(ret.values())


class RelationExtractionModel(nn.Module):
    def gradient_checkpointing_enable(self):
        self.feature_extractor.gradient_checkpointing_enable()

    def __init__(self, feature_extractor: nn.Module,
                 hidden_size: int, relation_tags: typing.List[str]):
        super().__init__()
        relation_representation_size = 2 * hidden_size
        self.feature_extractor = feature_extractor
        self.post_transformer = nn.Linear(in_features=relation_representation_size,
                                          out_features=relation_representation_size)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(in_features=relation_representation_size,
                                    out_features=len(relation_tags))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, global_attention_mask, head_index, tail_index, labels=None):
        features = self.feature_extractor(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          global_attention_mask=global_attention_mask)
        sequence_output = features.last_hidden_state
        e1_hidden = sequence_output[torch.arange(sequence_output.size(0)), head_index]
        e2_hidden = sequence_output[torch.arange(sequence_output.size(0)), tail_index]

        features = torch.concat((e1_hidden, e2_hidden), dim=1)
        features = self.post_transformer(features)
        features = self.activation(features)

        logits = self.classifier(features)

        if labels is None:
            return modeling_outputs.SequenceClassifierOutput(logits=logits)
        else:
            return modeling_outputs.SequenceClassifierOutput(logits=logits, loss=self.loss.forward(logits, labels))


def opening_entity_tag(entity_tag: str) -> str:
    entity_tag = re.sub(r'\s+', '_', entity_tag)
    entity_tag = entity_tag.upper()
    return f'[{entity_tag}]'


def closing_entity_tag(entity_tag: str) -> str:
    entity_tag = re.sub(r'\s+', '_', entity_tag)
    entity_tag = entity_tag.upper()
    return f'[/{entity_tag}]'


def mention_pair_to_sample(document: data.Document,
                           tokenizer: transformers.PreTrainedTokenizer,
                           head_mention_index: int,
                           tail_mention_index: int) -> typing.Dict[str, typing.Any]:
    tokens = [t.text for t in document.tokens]

    head_mention = document.mentions[head_mention_index]
    tail_mention = document.mentions[tail_mention_index]

    head_start = min(head_mention.document_level_token_indices(document))
    head_end = max(head_mention.document_level_token_indices(document)) + 1
    tail_start = min(tail_mention.document_level_token_indices(document))
    tail_end = max(tail_mention.document_level_token_indices(document)) + 1

    if head_start < tail_start:
        # head before tail, insert in reverse order to make indices easier
        tokens.insert(tail_end, f'{closing_entity_tag(tail_mention.ner_tag)}')
        tokens.insert(tail_start, f'[TAIL] {opening_entity_tag(tail_mention.ner_tag)}')

        tokens.insert(head_end, f'{closing_entity_tag(head_mention.ner_tag)}')
        tokens.insert(head_start, f'[HEAD] {opening_entity_tag(head_mention.ner_tag)}')
    else:
        # tail before head, insert in reverse order to make indices easier
        tokens.insert(head_end, f'{closing_entity_tag(head_mention.ner_tag)}')
        tokens.insert(head_start, f'[HEAD] {opening_entity_tag(head_mention.ner_tag)}')

        tokens.insert(tail_end, f'{closing_entity_tag(tail_mention.ner_tag)}')
        tokens.insert(tail_start, f'[TAIL] {opening_entity_tag(tail_mention.ner_tag)}')

    document_text = ' '.join(tokens)
    input_ids = torch.tensor(tokenizer.encode(document_text,
                                              padding='max_length', truncation=True, max_length=4096))
    num_input_ids = input_ids.shape
    head_index = (input_ids == tokenizer.convert_tokens_to_ids('[HEAD]')).nonzero().squeeze().item()
    tail_index = (input_ids == tokenizer.convert_tokens_to_ids('[TAIL]')).nonzero().squeeze().item()
    attention_mask = torch.ones(num_input_ids, dtype=torch.long, device=input_ids.device)
    global_attention_mask = torch.zeros(num_input_ids, dtype=torch.long, device=input_ids.device)
    global_attention_mask[head_index] = 1
    global_attention_mask[tail_index] = 1

    return {
        'text': document_text,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'global_attention_mask': global_attention_mask,
        'head_index': head_index,
        'tail_index': tail_index,
        'head_mention_index': head_mention_index,
        'tail_mention_index': tail_mention_index
    }


def documents_to_dataset(documents: typing.List[data.Document],
                         relation_tags: typing.List[str],
                         no_relation_tag: str,
                         tokenizer: transformers.PreTrainedTokenizer,
                         negative_sampling_rate: float) -> datasets.Dataset:
    def positive_samples_from_document(document: data.Document) -> typing.List[typing.Dict[str, typing.Any]]:
        ret = []

        for relation in document.relations:
            head_entity = document.entities[relation.head_entity_index]
            tail_entity = document.entities[relation.tail_entity_index]
            for head_mention_index, tail_mention_index in itertools.product(head_entity.mention_indices,
                                                                            tail_entity.mention_indices):
                head_mention = document.mentions[head_mention_index]
                tail_mention = document.mentions[tail_mention_index]

                # make sure we can learn from this mention pair,
                # i.e. they are evidence for the relation between their entities
                if head_mention.sentence_index not in relation.evidence:
                    continue
                if tail_mention.sentence_index not in relation.evidence:
                    continue

                sample = {
                    'label': relation_tags.index(relation.tag),
                    'doc_id': document.name,
                    **mention_pair_to_sample(document, tokenizer, head_mention_index, tail_mention_index)
                }

                ret.append(sample)
        return ret

    def negative_sample_from_document(document: data.Document,
                                      num_positive_samples: int,
                                      negative_rate: float) -> typing.List[typing.Dict[str, typing.Any]]:
        samples = []
        for head, tail in sampler.negative_sample(document,
                                                  num_positive=num_positive_samples,
                                                  negative_rate=negative_rate):
            samples.append({
                **mention_pair_to_sample(document, tokenizer, head, tail),
                'label': relation_tags.index(no_relation_tag),
                'doc_id': document.name
            })
        return samples

    ret = []
    for d in tqdm.tqdm(documents, desc='building dataset'):
        positive_samples = positive_samples_from_document(d)
        negative_samples = negative_sample_from_document(d, len(positive_samples), negative_sampling_rate)

        ret.extend(negative_samples)
        ret.extend(positive_samples)

    random.shuffle(ret)
    return datasets.Dataset.from_list(ret)


def all_combinations_dataset_from_documents(documents: typing.List[data.Document],
                                            tokenizer: transformers.PreTrainedTokenizer) -> datasets.Dataset:
    samples = []
    for document in documents:
        for head_mention_index, tail_mention_index in itertools.combinations(range(len(document.mentions)), 2):
            samples.append({
                **mention_pair_to_sample(document, tokenizer, head_mention_index, tail_mention_index),
                'doc_id': document.name
            })
    return datasets.Dataset.from_list(samples)
