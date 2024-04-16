import collections
import functools
import itertools
import os
import random
import typing

import catboost
import nltk
import numpy as np
import spacy
from gensim import downloader
from spacy import tokens

import data
from relations import sampler

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class CatBoostRelationEstimator:
    def __init__(self,
                 name: str,
                 negative_sampling_rate: float,
                 num_trees: int,
                 context_size: int,
                 use_pos_features: bool,
                 use_embedding_features: bool,
                 num_passes: int,
                 verbose: bool,
                 relation_tags: typing.List[str] = None,
                 ner_tags: typing.List[str] = None,
                 learning_rate: float = None,
                 class_weights: typing.Dict[str, float] = None,
                 depth: int = 8,
                 seed: int = 42):
        self._no_relation_tag = 'NO REL'
        if class_weights is not None:
            class_weights[self._no_relation_tag] = 1.0
        self._class_weights = class_weights
        self._model: typing.Dict[int, catboost.CatBoostClassifier] = {}
        self._num_trees = num_trees
        self._learning_rate = learning_rate
        self._depth = depth
        self._negative_rate = negative_sampling_rate
        self._target_tags = [t.lower() for t in relation_tags]
        self._ner_tags = [t.lower() for t in ner_tags]
        self._verbose = verbose
        self._name = name
        self._context_size = context_size
        self._use_pos_features = use_pos_features
        self._num_passes = num_passes
        self._use_embedding_features = use_embedding_features
        self._seed = seed
        self._embedding_size = 25
        self._embedder: typing.Optional = None
        if self._use_embedding_features:
            self._embedder = downloader.load('glove-twitter-25')
        self._nlp = spacy.load('en_core_web_sm')

    @staticmethod
    def load_model(config: dict):
        estimator = CatBoostRelationEstimator(**config)

        num_passes = config['num_passes']
        name = config['name']
        print(name)
        for pass_id in range(num_passes):
            model_path_str = f'api/models/catboost/{name}.cbm'
            estimator._model[pass_id] = catboost.CatBoostClassifier()
            estimator._model[pass_id].load_model(model_path_str)
        return estimator

    def train(self, documents: typing.List[data.Document]) -> 'CatBoostRelationEstimator':
        random.seed(self._seed)
        for pass_id in range(self._num_passes):
            print(f'Training pass #{pass_id + 1}/{self._num_passes}')
            self._model[pass_id] = catboost.CatBoostClassifier(iterations=self._num_trees,
                                                               verbose=False,
                                                               random_state=self._seed,
                                                               depth=self._depth,
                                                               class_weights=self._class_weights,
                                                               learning_rate=self._learning_rate)

            samples = self._get_samples(documents, pass_id)
            xs = [x for x, _ in samples]
            ys = [y for _, y in samples]

            cat_features_start = 4
            num_cat_features = 4
            if self._use_pos_features:
                num_cat_features += 2
            num_cat_features += self._context_size * 4

            cat_features = list(range(cat_features_start, cat_features_start + num_cat_features))
            self._model[pass_id].fit(xs, ys, cat_features=cat_features, verbose=self._verbose)

            # Save the model 
            model_save_path = f'api/models/catboost/{self._name}.cbm' 
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            self._model[pass_id].save_model(model_save_path)

            # Load the model for verification
            loaded_model = catboost.CatBoostClassifier()
            loaded_model.load_model(model_save_path)

            print("Model saved")

        return self

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) > 0 for d in documents])
        assert all([len(d.relations) == 0 for d in documents])


        last_pass = [d.copy(clear_relations=True) for d in documents]
        for pass_id in range(self._num_passes):
            print(f'Prediction pass #{pass_id + 1}/{self._num_passes}')
            predict_on_documents = [d.copy(clear_relations=True) for d in documents]
            last_pass = self._predict(pass_id, predict_on_documents, last_pass)
        return last_pass

    def _predict(self, pass_id: int, documents: typing.List[data.Document],
                 last_passes: typing.List[data.Document]) -> typing.List[data.Document]:
        for document, last_pass in zip(documents, last_passes):
            spacy_sentences = self._get_spacy_sentences(document)
            feature_builder = functools.partial(self._build_features,
                                                document=document,
                                                last_pass=last_pass,
                                                spacy_sentences=spacy_sentences)
            argument_indices: typing.List[typing.Tuple[int, int]] = []
            for mention_index_pair in itertools.combinations(range(len(document.mentions)), 2):
                argument_indices.append(mention_index_pair)  # forward
                argument_indices.append(mention_index_pair[::-1])  # backward

            xs = list(map(feature_builder, argument_indices))
            ys = self._model[pass_id].predict(xs)
            document.relations = self._get_relations_from_predictions(argument_indices, ys, document)
        return documents

    @staticmethod
    def _sub_sample_document_relations(document: data.Document, rate: float) -> data.Document:
        assert 0.0 <= rate <= 1.0
        ret = document.copy(clear_relations=True)
        for r in document.relations:
            if random.random() <= rate:
                ret.relations.append(r.copy())
        return ret

    def _get_samples(self, documents: typing.List[data.Document],
                     pass_id: int) -> typing.List[typing.Tuple[typing.List, str]]:
        samples = []
        document: data.Document
        for document in documents:
            spacy_sentences = self._get_spacy_sentences(document)

            if self._num_passes > 1:
                sub_sampling_rate = pass_id / (self._num_passes - 1)
                teacher_forced_last_pass = self._sub_sample_document_relations(document, sub_sampling_rate)
                print(len(teacher_forced_last_pass.relations), len(document.relations))
            else:
                teacher_forced_last_pass = document
            feature_builder = functools.partial(self._build_features,
                                                document=document,
                                                last_pass=teacher_forced_last_pass,
                                                spacy_sentences=spacy_sentences)

            samples_in_document = 0
            for relation in document.relations:
                head_entity = document.entities[relation.head_entity_index]
                tail_entity = document.entities[relation.tail_entity_index]
                mention_index_pairs = itertools.product(head_entity.mention_indices, tail_entity.mention_indices)

                mention_index_pairs = [
                    (h, t)
                    for h, t in mention_index_pairs
                    if document.mentions[h].sentence_index in relation.evidence
                       and document.mentions[t].sentence_index in relation.evidence
                ]

                xs = list(map(feature_builder, mention_index_pairs))
                ys = [relation.tag] * len(xs)

                samples.extend(zip(xs, ys))

                samples_in_document += len(ys)

            assert len(document.relations) <= samples_in_document

            negative_samples = sampler.negative_sample(document,
                                                       num_positive=samples_in_document,
                                                       negative_rate=self._negative_rate,
                                                       verbose=self._verbose)

            xs = list(map(feature_builder, negative_samples))
            ys = [self._no_relation_tag] * len(xs)
            samples.extend(zip(xs, ys))

        random.shuffle(samples)
        return samples

    def _get_relations_from_predictions(self, indices: typing.List[typing.Tuple[int, int]],
                                        ys: np.ndarray, document: data.Document) -> typing.List[data.Relation]:
        assert len(indices) == len(ys)

        relations = []

        for (head_mention_index, tail_mention_index), tag in zip(indices, ys):
            tag = tag[0]
            assert type(tag) == str, f'Expected prediction to be string, got "{tag}" ({type(tag)})'
            if tag == self._no_relation_tag:
                # predicted no relation between the two
                continue

            assert tag in self._target_tags

            head_mention = document.mentions[head_mention_index]
            tail_mention = document.mentions[tail_mention_index]

            head_index = document.entity_index_for_mention(document.mentions[head_mention_index])
            tail_index = document.entity_index_for_mention(document.mentions[tail_mention_index])

            relations.append(data.Relation(
                head_entity_index=head_index,
                tail_entity_index=tail_index,
                tag=tag,
                evidence=list({head_mention.sentence_index, tail_mention.sentence_index})
            ))

        return relations

    def _get_spacy_sentences(self, document: data.Document) -> typing.List[tokens.Doc]:
        spacy_sentences: typing.List[tokens.Doc] = []
        batch = [
            tokens.Doc(self._nlp.vocab, [t.text for t in sentence.tokens])
            for sentence in document.sentences
        ]
        doc: tokens.Doc
        for doc in self._nlp.pipe(batch):
            spacy_sentences.append(doc)
        return spacy_sentences

    def _build_features(self,
                        mention_index_pair: typing.Tuple[int, int],
                        document: data.Document,
                        last_pass: data.Document,
                        spacy_sentences: typing.List[tokens.Doc]) -> typing.List:
        assert document.name == last_pass.name

        head_mention_index, tail_mention_index = mention_index_pair
        head = document.mentions[head_mention_index]
        tail = document.mentions[tail_mention_index]

        context = []
        for i in range(-self._context_size, self._context_size + 1):
            if i == 0:
                continue

            mention_index = head_mention_index + i
            if mention_index < 0:
                context.append(None)
            elif mention_index >= len(document.mentions):
                context.append(None)
            else:
                context.append(document.mentions[mention_index])

            mention_index = tail_mention_index + i
            if mention_index < 0:
                context.append(None)
            elif mention_index >= len(document.mentions):
                context.append(None)
            else:
                context.append(document.mentions[mention_index])

        _, head_spacy = self.root_token_for_mention(head, document, spacy_sentences[head.sentence_index])
        _, tail_spacy = self.root_token_for_mention(tail, document, spacy_sentences[tail.sentence_index])

        features = [
            tail_mention_index - head_mention_index,
            tail.sentence_index - head.sentence_index,
            len(head.token_indices),
            len(tail.token_indices),
            head.ner_tag,
            tail.ner_tag,
            head_spacy.dep_,
            tail_spacy.dep_
        ]

        if self._use_pos_features:
            features += [
                head_spacy.pos_,
                tail_spacy.pos_,
            ]

        features += [
            m.ner_tag if m is not None else '' for m in context
        ]

        # mentions_between = [
        #     document.mentions[i]
        #     for i
        #     in range(min(head_mention_index, tail_mention_index) + 1,
        #              max(head_mention_index, tail_mention_index))
        # ]
        # mentions_between_count = {
        #     tag: 0 for tag in self._ner_tags
        # }
        # for m in mentions_between:
        #     mentions_between_count[m.ner_tag.lower()] += 1
        #
        # features += [
        #     mentions_between_count[tag] for tag in sorted(list(mentions_between_count.keys()))
        # ]

        if self._num_passes > 1:
            head_relations = last_pass.get_relations_by_mention(head_mention_index, only_head=True)
            head_incoming_relations = collections.Counter([r.tag for r in head_relations])
            # if len(head_relations) > 0:
            #     print(head_relations)

            tail_relations = last_pass.get_relations_by_mention(tail_mention_index, only_tail=True)
            tail_incoming_relations = collections.Counter([r.tag for r in tail_relations])
            # if len(tail_relations) > 0:
            #     print(tail_relations)

            # if sum(head_incoming_relations.values()) > 0 or sum(tail_incoming_relations.values()) > 0:
            #     print(f'head: {head_incoming_relations}, tail: {tail_incoming_relations}')

            features += [
                *[head_incoming_relations[t] for t in self._target_tags],
                *[tail_incoming_relations[t] for t in self._target_tags]
            ]

        if self._use_embedding_features:
            features += [
                *self.embed_tokens(head.get_tokens(document)),
                *self.embed_tokens(tail.get_tokens(document))
            ]

        return features

    def root_token_for_mention(self, mention: data.Mention, document: data.Document,
                               spacy_tokens: tokens.Doc) -> typing.Tuple[data.Token, tokens.Token]:
        sentence = document.sentences[mention.sentence_index]
        spacy_token: tokens.Token

        assert len(mention.token_indices) > 0

        mention_token_depths = {
            mention_token_index: self._get_token_depth(spacy_tokens[token_index_in_sentence])
            for mention_token_index, token_index_in_sentence in enumerate(mention.token_indices)
        }

        top_level_token_index: int = sorted(mention_token_depths, key=mention_token_depths.get)[0]

        token = sentence.tokens[mention.token_indices[top_level_token_index]]
        spacy_token = spacy_tokens[mention.token_indices[top_level_token_index]]

        return token, spacy_token

    @staticmethod
    def _get_token_depth(spacy_token: tokens.Token) -> int:
        depth = 0
        while spacy_token.head != spacy_token:
            depth += 1
            spacy_token = spacy_token.head
        return depth

    def embed_tokens(self, tokens: typing.List[data.Token]):
        words = [t.text for t in tokens]
        seq = np.array([self._embedder[w] for w in words if w in self._embedder])
        if seq.shape[0] == 0:
            return np.ones(self._embedding_size) * 1.0 / self._embedding_size ** 0.5

        seq = seq.sum(axis=0)
        return seq / ((seq ** 2).sum() + 1e-100) ** 0.5

    def _ner_tag_to_one_hot(self, ner_tag: str) -> np.ndarray:
        one_hot = np.zeros(len(self._ner_tags), )
        one_hot[self._ner_tags.index(ner_tag.lower())] = 1.0
        return one_hot

    def _relation_tag_to_one_hot(self, relation_tag: str) -> np.ndarray:
        target = np.zeros((len(self._target_tags) + 1,))
        if relation_tag.lower() in self._target_tags:
            target[self._target_tags.index(relation_tag)] = 1.
        else:
            target[len(self._target_tags)] = 1.
        return target

    def _relation_tag_to_scalar(self, relation_tag: str) -> float:
        if relation_tag == self._no_relation_tag:
            return len(self._target_tags)
        return self._target_tags.index(relation_tag.lower())
