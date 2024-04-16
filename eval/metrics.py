import dataclasses
import typing

import data


@dataclasses.dataclass
class Scores:
    p: float
    r: float
    f1: float

    @staticmethod
    def from_stats(stats: 'Stats') -> 'Scores':
        return Scores(
            p=stats.precision,
            r=stats.recall,
            f1=stats.f1
        )

    def __add__(self, other):
        if type(other) != Scores:
            raise TypeError(f'Can not add Scores and {type(other)}')
        return Scores(
            p=self.p + other.p,
            r=self.r + other.r,
            f1=self.f1 + other.f1
        )

    def __truediv__(self, other):
        return Scores(
            p=self.p / other,
            r=self.r / other,
            f1=self.f1 / other
        )


@dataclasses.dataclass
class Stats:
    num_pred: float
    num_gold: float
    num_ok: float

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall
        if precision + recall == 0.0:
            return 0
        return 2 * precision * recall / (precision + recall)

    @property
    def precision(self) -> float:
        if self.num_pred == 0 and self.num_gold == 0:
            return 1.0
        elif self.num_pred == 0 and self.num_gold != 0:
            return 0.0
        else:
            return self.num_ok / self.num_pred

    @property
    def recall(self) -> float:
        if self.num_gold == 0 and self.num_pred == 0:
            return 1.0
        elif self.num_gold == 0 and self.num_pred != 0:
            return 0.0
        else:
            return self.num_ok / self.num_gold

    def __add__(self, other):
        if type(other) != Stats:
            raise TypeError(f'Can not add Stats and {type(other)}')
        return Stats(
            num_pred=self.num_pred + other.num_pred,
            num_gold=self.num_gold + other.num_gold,
            num_ok=self.num_ok + other.num_ok
        )


def relation_f1_stats(*, predicted_documents: typing.List[data.Document],
                      ground_truth_documents: typing.List[data.Document],
                      verbose: bool = False) -> typing.Dict[str, Stats]:
    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='relations', verbose=verbose)


def mentions_f1_stats(*, predicted_documents: typing.List[data.Document],
                      ground_truth_documents: typing.List[data.Document],
                      verbose: bool = False) -> typing.Dict[str, Stats]:
    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='mentions', verbose=verbose)


def entity_f1_stats(*, predicted_documents: typing.List[data.Document],
                    ground_truth_documents: typing.List[data.Document],
                    only_tags: typing.List[str],
                    min_num_mentions: int = 1,
                    verbose: bool = False) -> typing.Dict[str, Stats]:
    predicted_documents = [d.copy() for d in predicted_documents]
    for d in predicted_documents:
        d.entities = [e for e in d.entities
                      if len(e.mention_indices) >= min_num_mentions and e.get_tag(d) in only_tags]

    ground_truth_documents = [d.copy() for d in ground_truth_documents]
    for d in ground_truth_documents:
        d.entities = [e for e in d.entities
                      if len(e.mention_indices) >= min_num_mentions and e.get_tag(d) in only_tags]

    return _f1_stats(predicted_documents=predicted_documents,
                     ground_truth_documents=ground_truth_documents,
                     attribute='entities', verbose=verbose)


def _add_to_stats_by_tag(stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]],
                         get_tag: typing.Callable[[typing.Any], str], object_list: typing.Iterable, stat: str):
    assert stat in ['gold', 'pred', 'ok']
    for e in object_list:
        tag = get_tag(e)
        if tag not in stats_by_tag:
            stats_by_tag[tag] = (0, 0, 0)
        prev_stats = stats_by_tag[tag]
        if stat == 'gold':
            stats_by_tag[tag] = (prev_stats[0] + 1, prev_stats[1], prev_stats[2])
        elif stat == 'pred':
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1] + 1, prev_stats[2])
        else:
            stats_by_tag[tag] = (prev_stats[0], prev_stats[1], prev_stats[2] + 1)
    return stats_by_tag


def _get_ner_tag_for_tuple(element_type: str, element: typing.Tuple, document: data.Document) -> str:
    assert element_type in ['mentions', 'relations', 'entities']
    assert type(element) == tuple
    if element_type == 'entities':
        return list(element[0])[0][0]
    return element[0]


def _f1_stats(*, predicted_documents: typing.List[data.Document],
              ground_truth_documents: typing.List[data.Document],
              attribute: str, verbose: bool = False) -> typing.Dict[str, Stats]:
    assert attribute in ['mentions', 'relations', 'entities']
    assert len(predicted_documents) == len(ground_truth_documents)

    stats_by_tag: typing.Dict[str, typing.Tuple[float, float, float]] = {}

    for p, t in zip(predicted_documents, ground_truth_documents):
        true_attribute = getattr(t, attribute)
        pred_attribute = getattr(p, attribute)

        true_as_set = set([e.to_tuple(t) for e in true_attribute])
        assert len(true_as_set) == len(
            true_attribute), f'{len(true_as_set)}, {len(true_attribute)}, {true_as_set}, {true_attribute}'

        pred_as_set = set([e.to_tuple(p) for e in pred_attribute])

        _add_to_stats_by_tag(stats_by_tag, lambda e: _get_ner_tag_for_tuple(attribute, e, t), true_as_set, 'gold')
        _add_to_stats_by_tag(stats_by_tag, lambda e: _get_ner_tag_for_tuple(attribute, e, p), pred_as_set, 'pred')

        ok_preds = true_as_set.intersection(pred_as_set)
        non_ok = [e.pretty_print(p) for e in pred_attribute if e.to_tuple(p) not in true_as_set
                  # if _get_ner_tag_for_tuple(attribute, e.to_tuple(p), p).lower() == 'actor'
                  ]
        if verbose and len(non_ok) > 0:
            print('=' * 150)
            print(p.text)
            print('pred')
            print(', '.join([a.pretty_print(p) for a in pred_attribute]))
            print([e for e in pred_as_set])
            print('-' * 100)
            print('true')
            print(', '.join([a.pretty_print(t) for a in true_attribute]))
            print([e for e in true_as_set])
            print('-' * 100)
            print('ok')
            print(ok_preds)
            print('non ok')
            print(non_ok)

        _add_to_stats_by_tag(stats_by_tag, lambda e: _get_ner_tag_for_tuple(attribute, e, p), ok_preds, 'ok')

    return {
        tag: Stats(num_pred=p, num_gold=g, num_ok=o) for tag, (g, p, o) in stats_by_tag.items()
    }
