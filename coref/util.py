import collections
import typing

import data


def resolve_ner_conflicts(document: data.Document,
                          mention_indices: typing.List[int],
                          ner_tag_strategy: str,
                          verbose: bool = False) -> typing.Optional[data.Entity]:
    ner_tags = [document.mentions[m_id].ner_tag for m_id in mention_indices]
    if len(set(ner_tags)) == 1:
        return data.Entity(mention_indices)

    if ner_tag_strategy == 'keep':
        # keep all mentions regardless of ner tags
        if verbose:
            print(f'Resolved entity would have mixed NER tags at mention level. '
                  f'KEEPING cluster as per ner tag strategy.')
        return data.Entity(mention_indices)
    elif ner_tag_strategy == 'skip':
        # skip entire cluster
        if verbose:
            print(f'Resolved entity would have mixed NER tags at mention level. '
                  f'DISCARDING cluster as per ner tag strategy.')
        return None
    elif ner_tag_strategy == 'frequency':
        # use only mentions that belong to the most frequent ner tag
        if verbose:
            print(f'Resolved entity would have mixed NER tags at mention level. '
                  f'KEEPING mentions of most frequent ner tag in cluster as per ner tag strategy.')
        counter = collections.Counter(ner_tags)
        most_frequent_tag = max(counter.keys(), key=lambda k: counter[k])
        mention_indices = [i for i in mention_indices if document.mentions[i].ner_tag == most_frequent_tag]
        return data.Entity(mention_indices)
    else:
        raise ValueError(f'Unknown ner strategy "{ner_tag_strategy}"')


def resolve_remaining_mentions_to_entities(document: data.Document) -> None:
    """
    Resolves each mention that is not yet part of an entity
    to a new entity that only contains this mention.
    """
    part_of_entity = set()
    for e in document.entities:
        for i in e.mention_indices:
            part_of_entity.add(i)
    for mention_index, mention in enumerate(document.mentions):
        if mention_index in part_of_entity:
            continue
        document.entities.append(data.Entity([mention_index]))
