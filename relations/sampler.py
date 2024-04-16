import functools
import itertools
import math
import random
import timeit
import typing

import numpy as np

import data


def negative_sample_np(document: data.Document, num_positive: int, negative_rate: float,
                       verbose: bool = False) -> typing.List[typing.Tuple[int, int]]:
    """
    scales better with large negative sampling rates, but slower for smaller ones (<10k)
    :param document:
    :param num_positive:
    :param negative_rate:
    :param verbose:
    :return:
    """
    num_negative_samples = math.ceil(negative_rate * num_positive)

    forward_candidates = np.array(list(itertools.combinations(range(len(document.mentions)), 2)))
    backward_candidates = forward_candidates.copy()
    backward_candidates[:, [0, 1]] = backward_candidates[:, [1, 0]]
    candidates = np.concatenate([forward_candidates, backward_candidates])

    negative_samples_mask_fn = functools.partial(_is_negative_sample, document=document)
    negative_samples_mask = candidates.copy()
    np.apply_along_axis(negative_samples_mask_fn, 1, candidates)

    candidates = candidates[negative_samples_mask]

    num_candidates = candidates.shape[0]
    negative_sample_indices = np.random.choice(num_candidates, num_negative_samples, replace=True)

    candidates = candidates[negative_sample_indices, :].tolist()
    return candidates


def _is_negative_sample(row: np.ndarray, document: data.Document) -> np.ndarray:
    head_mention_index: int = row[0]
    tail_mention_index: int = row[1]
    head_index = document.entity_index_for_mention_index(head_mention_index)
    tail_index = document.entity_index_for_mention_index(tail_mention_index)
    relation_exists = document.relation_exists_between(head_index, tail_index)

    if relation_exists:
        return np.array([False, False])
    return np.array([True, True])


def negative_sample(document: data.Document, num_positive: int, negative_rate: float,
                    verbose: bool = False) -> typing.List[typing.Tuple[int, int]]:
    num_negative_samples = math.ceil(negative_rate * num_positive)
    negative_samples = []

    candidates = list(itertools.combinations(range(len(document.mentions)), 2))
    candidates += [(t, h) for h, t in candidates]

    for head_mention_index, tail_mention_index in candidates:
        if len(negative_samples) >= num_negative_samples:
            break

        head_index = document.entity_index_for_mention_index(head_mention_index)
        tail_index = document.entity_index_for_mention_index(tail_mention_index)
        if document.relation_exists_between(head_index, tail_index):
            continue

        negative_samples.append((head_mention_index, tail_mention_index))

    if len(negative_samples) < num_negative_samples:
        if verbose:
            print(f'Could only build {len(negative_samples)}/{num_negative_samples} '
                  f'negative samples, as there were not enough candidates in {document.name}, '
                  f'reusing some.')
        missing_num_samples = num_negative_samples - len(negative_samples)
        while missing_num_samples > 0:
            negative_samples += negative_samples[:missing_num_samples]
            missing_num_samples = num_negative_samples - len(negative_samples)

        random.shuffle(negative_samples)

    return negative_samples


if __name__ == '__main__':
    documents_ = data.loader.read_documents_from_json(f'../jsonl/fold_0/train.json')

    print('numpy')
    print(timeit.timeit(lambda: [negative_sample_np(doc_, len(doc_.relations), 10000.0) for doc_ in documents_],
                        number=1))
    print('plain python')
    print(timeit.timeit(lambda: [negative_sample(doc_, len(doc_.relations), 10000.0) for doc_ in documents_],
                        number=1))
