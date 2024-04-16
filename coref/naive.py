import typing

import data
from coref import util


class NaiveCoRefSolver:
    def __init__(self, resolved_tags: typing.List[str], min_mention_overlap: float = .33):
        self._tags = resolved_tags
        self._mention_overlap_threshold = min_mention_overlap

    def resolve_co_references(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) == 0 for d in documents])

        for document in documents:
            all_matches: typing.Dict[int, typing.Dict[int, float]] = {}
            for mention_index, mention in enumerate(document.mentions):
                if mention.ner_tag not in self._tags:
                    continue
                all_matches[mention_index] = self._match_mention(mention, document)

            # resolve mentions starting with best matches
            # noinspection PyTypeChecker
            all_match_items: typing.List[typing.Tuple[int, typing.Dict[int, float]]] = list(all_matches.items())
            all_match_items = [i for i in all_match_items if len(i[1].values()) > 0]
            all_match_items.sort(key=lambda item: max(item[1].values()), reverse=True)

            entities: typing.Dict[int, data.Entity] = {}

            for mention_index, matches in all_match_items:
                top_match_index = max(matches, key=lambda k: matches[k])
                top_match_overlap = matches[top_match_index]

                if mention_index in entities:
                    # already resolved
                    continue

                if top_match_overlap < self._mention_overlap_threshold:
                    continue

                if top_match_index not in entities:
                    entities[top_match_index] = data.Entity([top_match_index])
                entities[top_match_index].mention_indices.append(mention_index)
                entities[mention_index] = entities[top_match_index]

            document.entities.extend(entities.values())
            util.resolve_remaining_mentions_to_entities(document)

        return documents

    @staticmethod
    def _match_mention(mention: data.Mention, document: data.Document) -> typing.Dict[int, float]:
        matches: typing.Dict[int, float] = {}
        for other_index, other in enumerate(document.mentions):
            if other == mention:
                continue

            if other.ner_tag != mention.ner_tag:
                continue

            mention_token_texts = NaiveCoRefSolver._text_from_mention(mention, document)
            other_token_texts = NaiveCoRefSolver._text_from_mention(other, document)
            overlap_text = NaiveCoRefSolver._longest_overlap_of_lists(mention_token_texts, other_token_texts)

            left_overlap = len(overlap_text) / len(mention_token_texts)
            right_overlap = len(overlap_text) / len(other_token_texts)
            overlap = (left_overlap + right_overlap) / 2

            matches[other_index] = overlap
        return matches

    @staticmethod
    def _text_from_mention(mention: data.Mention, document: data.Document) -> typing.List[str]:
        return [t.text for t in mention.get_tokens(document)]

    @staticmethod
    def _longest_overlap_of_lists(left: typing.List, right: typing.List) -> typing.List:
        """
        Adapted from https://www.w3resource.com/python-exercises/list-advanced/python-list-advanced-exercise-11.php
        """
        m, n = len(left), len(right)
        jh = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if left[i - 1] == right[j - 1]:
                    jh[i][j] = 1 + jh[i - 1][j - 1]
                else:
                    jh[i][j] = max(jh[i - 1][j], jh[i][j - 1])

        result = []
        i, j = m, n
        while i > 0 and j > 0:
            if left[i - 1] == right[j - 1]:
                result.append(left[i - 1])
                i -= 1
                j -= 1
            elif jh[i - 1][j] > jh[i][j - 1]:
                i -= 1
            else:
                j -= 1

        return result[::-1]
