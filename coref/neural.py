import collections
import typing

import spacy
from spacy import tokens

import data
from coref import util


class NeuralCoRefSolver:
    # loading an english SpaCy model
    nlp = spacy.load('en_coreference_web_trf')

    def __init__(self, co_referencable_tags: typing.List[str],
                 ner_tag_strategy: str = 'skip',
                 min_cluster_overlap: float = .33,
                 min_mention_overlap: float = .1,
                 verbose: bool = False):
        """
        :param co_referencable_tags: NER tags we want to resolve co-references for (ignore all others, e.g. Activities)
        :param ner_tag_strategy: strategy to use when co-reference between mentions of different ner-tags have been
               found: "skip" ignores the entire cluster, "keep" uses all mentions regardless of their ner tag,
               "frequency" only uses mentions that have the most frequent ner tag in the predicted cluster
        :param min_cluster_overlap: minimum percentage of cluster mentions predicted by neuralcoref
               that have to be resolved to mentions for a entity to be accepted
        :param min_mention_overlap: minimum percentage of overlap between a neuralcoref mention prediction and a
                                    mention present in the document to be counted as resolved
        """
        self._tags = [t.lower() for t in co_referencable_tags]
        self._min_cluster_overlap = min_cluster_overlap
        self._min_mention_overlap = min_mention_overlap
        self._ner_tag_strategy = ner_tag_strategy
        self._verbose = verbose

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(document.entities) == 0 for document in documents])

        for document in documents:
            coref_entities = self._get_co_reference_indices(document)
            for e in coref_entities:
                # try to resolve the list of mentions (each a list of token indices) to a entity
                entity = self._resolve_single_entity(e, document)
                if self._verbose:
                    print('=========================')
                if entity is None:
                    continue
                if document.contains_entity(entity):
                    continue
                document.entities.append(entity)
            util.resolve_remaining_mentions_to_entities(document)
        return documents

    def _resolve_single_entity(self,
                               cluster_token_indices: typing.List[typing.List[int]],
                               document: data.Document) -> typing.Optional[data.Entity]:
        if self._verbose:
            print('neuralcoref found the following cluster:')
            for mention_indices in cluster_token_indices:
                mention_text = " ".join([document.tokens[i].text for i in mention_indices])
                print(f'"{mention_text}", document level token indicies: {mention_indices}')
            print(f'Its tokens are contained in the following predicted mentions (of one of the types {self._tags}:')

        mention_indices = set()
        for mention_token_indices in cluster_token_indices:
            mention_index = self._get_mention_for_token_indices(mention_token_indices, document, self._tags,
                                                                threshold=self._min_mention_overlap,
                                                                verbose=self._verbose)
            if mention_index is not None:
                mention_indices.add(mention_index)

        mention_indices = list(mention_indices)
        if len(mention_indices) == 0:
            # did not find a single predicted mention for the cluster neuralcoref predicted
            return None

        overlap = len(mention_indices) / len(cluster_token_indices)

        if overlap < self._min_cluster_overlap:
            if self._verbose:
                print(f'Only resolved {len(mention_indices)} of {len(cluster_token_indices)} predicted mentions '
                      f'({overlap:.2%} overlap), DISCARDING entity!')
            return None

        if self._verbose:
            print(f'Resolved {len(mention_indices)} of {len(cluster_token_indices)} predicted mentions '
                  f'({overlap:.2%} overlap).')

        return util.resolve_ner_conflicts(document, mention_indices, self._ner_tag_strategy, verbose=self._verbose)

    @staticmethod
    def _get_mention_for_token_indices(token_indices: typing.List[int],
                                       document: data.Document,
                                       valid_tags: typing.List[str],
                                       threshold: float = 0.1,
                                       verbose: bool = False) -> typing.Optional[int]:
        """
        Resolves a list of token indices to a mention predicted by the pipeline

        :param threshold: float between 0 and 1, dictating how many of the given token_indices have
                          to match a mention's token indices for it to be returned

        :returns: index of the matched data.Mention if we could find one, None otherwise
        """
        candidates = collections.defaultdict(int)
        for token_index in token_indices:
            token = document.tokens[token_index]
            for mention_index, mention in enumerate(document.mentions):
                if mention.contains_token(token, document):
                    candidates[mention_index] += 1
                    break
        if len(candidates) == 0:
            # no mention found
            if verbose:
                print(f'No candidates found')
            return None

        mention_index = max(candidates, key=candidates.get)
        mention = document.mentions[mention_index]
        overlap = candidates[mention_index] / len(token_indices)

        is_above_threshold = overlap >= threshold
        if is_above_threshold:
            if verbose:
                print(f'{mention.pretty_print(document)} with an overlap of {overlap:.2%}')
            return mention_index

    def _get_co_reference_indices(self, document: data.Document) -> typing.List[typing.List[typing.List[int]]]:
        """
        return a three times nested list
        - 1: list of entities
          - 2: list of mentions of a given entity
            - 3: list of token indices of a given mention
        All token indices are document level!
        """

        doc = self.nlp(spacy.tokens.Doc(self.nlp.vocab, [token.text for token in document.tokens]))

        clusters: typing.List[spacy.tokens.span_group.SpanGroup]
        clusters = [cluster for cluster_id, cluster in doc.spans.items() if cluster_id.startswith('coref_clusters')]

        entities = []
        for cluster in clusters:
            entity = []
            for mention in cluster:
                entity.append(list(range(mention.start, mention.end)))
            entities.append(entity)
        return entities
