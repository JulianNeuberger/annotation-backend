import typing

import data


class RuleBasedRelationEstimator:
    def __init__(self, rules: typing.List['RelationExtractionRule']):
        self._rules = rules

    def predict(self, documents: typing.List[data.Document]) -> typing.List[data.Document]:
        assert all([len(d.entities) > 0 for d in documents])
        assert all([len(d.relations) == 0 for d in documents])

        for document in documents:
            for rule in self._rules:
                rs = rule.get_relations(document)
                for r in rs:
                    assert r.head_entity_index != r.tail_entity_index
                document.relations.extend(rs)

        return documents


class RelationExtractionRule:
    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        raise NotImplementedError()

    @staticmethod
    def get_next_index_of_mention_with_tag(document: data.Document, start_index: int, tags: typing.List[str],
                                           search_backwards: bool = False) -> typing.Optional[int]:
        search_indices = range(start_index, len(document.mentions))
        if search_backwards:
            search_indices = range(start_index, -1, -1)

        mentions = document.mentions.copy()
        for i in search_indices:
            mention = mentions[i]
            if mention.ner_tag in tags:
                return i
        return None


class SequenceFlowsRule(RelationExtractionRule):
    def __init__(self, triggering_elements: typing.List[str], target_tag: str, verbose: bool = False):
        self._elements = triggering_elements
        self._tag = target_tag
        self._verbose = verbose

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, current_mention in enumerate(document.mentions):
            cur_mention_behavioural = current_mention.ner_tag in self._elements
            if not cur_mention_behavioural:
                continue

            next_behavioral_element_index = self.get_next_index_of_mention_with_tag(document, i + 1, self._elements)

            if next_behavioral_element_index is None:
                continue

            tail_mention = document.mentions[next_behavioral_element_index]

            head_entity = document.entity_index_for_mention(current_mention)
            tail_entity = document.entity_index_for_mention(tail_mention)

            flow_relation = data.Relation(
                head_entity_index=head_entity,
                tail_entity_index=tail_entity,
                tag=self._tag, evidence=list({current_mention.sentence_index, tail_mention.sentence_index})
            )

            if document.contains_relation(flow_relation):
                if self._verbose:
                    print(f'[{self.__class__.__name__}] '
                          f'{flow_relation.pretty_print(document)} already in relations, not adding.')
                continue

            relations.append(flow_relation)

        return relations


class SameGatewayRule(RelationExtractionRule):
    def __init__(self, triggering_elements: typing.List[str], target_tag: str):
        self._elements = triggering_elements
        self._tag = target_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag not in self._elements:
                continue

            next_gateway_index = self.get_next_index_of_mention_with_tag(document, i + 1, [mention.ner_tag])
            if next_gateway_index is None:
                continue

            next_gateway = document.mentions[next_gateway_index]

            distance = next_gateway.sentence_index - mention.sentence_index
            if distance > 1:
                continue

            head_entity = document.entity_index_for_mention(mention)
            tail_entity = document.entity_index_for_mention(next_gateway)

            relations.append(data.Relation(
                head_entity_index=head_entity,
                tail_entity_index=tail_entity,
                tag=self._tag,
                evidence=list({mention.sentence_index, next_gateway.sentence_index})
            ))

        return relations


class GatewayActivityRule(RelationExtractionRule):
    def __init__(self, gateway_tags: typing.List[str], activity_tag: str, same_gateway_tag: str, flow_tag: str):
        self._gateways = gateway_tags
        self._activity = activity_tag
        self._same_gateway = same_gateway_tag
        self._tag = flow_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag not in self._gateways:
                continue
            if self.is_mention_part_of_relation_with_tag(document, i, self._same_gateway):
                continue

            next_activity_index = self.get_next_index_of_mention_with_tag(document, i + 1, [self._activity])
            if next_activity_index is None:
                continue

            tail_mention = document.mentions[next_activity_index]

            head_entity = document.entity_index_for_mention(mention)
            tail_entity = document.entity_index_for_mention(tail_mention)

            flow_relation = data.Relation(
                head_entity_index=head_entity,
                tail_entity_index=tail_entity,
                tag=self._tag,
                evidence=list({mention.sentence_index, tail_mention.sentence_index})
            )

            if document.contains_relation(flow_relation):
                print(f'[{self.__class__.__name__}] {flow_relation} already in relations, not adding.')
                continue

            relations.append(flow_relation)
        return relations

    @staticmethod
    def is_mention_part_of_relation_with_tag(document: data.Document, mention_index: int, tag: str) -> bool:
        for relation in document.relations:
            if relation.tag != tag:
                continue
            if relation.tail_entity_index == mention_index:
                return True
            if relation.head_entity_index == mention_index:
                return True

        return False


class ActorPerformerRecipientRule(RelationExtractionRule):
    def __init__(self, actor_tag: str, activity_tag: str, performer_tag: str, recipient_tag: str):
        self._actor = actor_tag
        self._activity = activity_tag
        self._performer = performer_tag
        self._recipient = recipient_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []

        for i, mention in enumerate(document.mentions):
            if not mention.ner_tag == self._activity:
                continue

            performer_index = self.get_next_index_of_mention_with_tag(document, i, [self._actor],
                                                                      search_backwards=True)
            if performer_index is not None:
                if document.mentions[performer_index].sentence_index == mention.sentence_index:
                    tail_mention = document.mentions[performer_index]
                    head_entity = document.entity_index_for_mention(mention)
                    tail_entity = document.entity_index_for_mention(tail_mention)

                    relations.append(data.Relation(
                        head_entity_index=head_entity,
                        tail_entity_index=tail_entity,
                        tag=self._performer,
                        evidence=list({mention.sentence_index, tail_mention.sentence_index})
                    ))

            recipient_index = self.get_next_index_of_mention_with_tag(document, i, [self._actor])
            if recipient_index is not None:
                if document.mentions[recipient_index].sentence_index == mention.sentence_index:
                    tail_mention = document.mentions[recipient_index]
                    head_entity = document.entity_index_for_mention(mention)
                    tail_entity = document.entity_index_for_mention(tail_mention)

                    relations.append(data.Relation(
                        head_entity_index=head_entity,
                        tail_entity_index=tail_entity,
                        tag=self._recipient,
                        evidence=list({mention.sentence_index, tail_mention.sentence_index})
                    ))

        return relations


class FurtherSpecificationRule(RelationExtractionRule):
    def __init__(self, further_specification_element_tag: str, activity_tag: str,
                 further_specification_relation_tag: str):
        self._further_spec = further_specification_element_tag
        self._activity = activity_tag
        self._tag = further_specification_relation_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []

        for i, mention in enumerate(document.mentions):
            if mention.ner_tag != self._further_spec:
                continue

            left_index = self.get_next_index_of_mention_with_tag(document, i, [self._activity], search_backwards=True)
            right_index = self.get_next_index_of_mention_with_tag(document, i, [self._activity])

            if left_index is None and right_index is None:
                continue

            chosen_index: int
            if left_index is None:
                chosen_index = right_index
            elif right_index is None:
                chosen_index = left_index
            else:
                # both not None, choose nearer

                # closest token of left candidate is the right-most
                left_candidate = document.mentions[left_index]
                left_sentence = document.sentences[left_candidate.sentence_index]
                left_token = left_sentence.tokens[left_candidate.token_indices[-1]]
                left_end = left_token.index_in_document

                # closest token of right candidate is the left-most
                right_candidate = document.mentions[right_index]
                right_sentence = document.sentences[right_candidate.sentence_index]
                right_token = right_sentence.tokens[right_candidate.token_indices[0]]
                right_start = right_token.index_in_document

                source_start = document.sentences[mention.sentence_index].tokens[
                    min(mention.token_indices)].index_in_document
                source_end = document.sentences[mention.sentence_index].tokens[
                    max(mention.token_indices)].index_in_document

                distance_to_left = source_start - left_end
                distance_to_right = right_start - source_end
                if distance_to_left <= distance_to_right:
                    chosen_index = left_index
                else:
                    chosen_index = right_index

            head_mention = document.mentions[chosen_index]
            head_entity = document.entity_index_for_mention(head_mention)
            tail_entity = document.entity_index_for_mention(mention)

            relations.append(data.Relation(
                head_entity_index=head_entity,
                tail_entity_index=tail_entity,
                tag=self._tag,
                evidence=list({mention.sentence_index, head_mention.sentence_index})
            ))

        return relations


class UsesRelationRule(RelationExtractionRule):
    def __init__(self, activity_data_tag: str, activity_tag: str, uses_relation_tag: str):
        self._activity_data = activity_data_tag
        self._activity = activity_tag
        self._tag = uses_relation_tag

    def get_relations(self, document: data.Document) -> typing.List[data.Relation]:
        relations = []
        for i, mention in enumerate(document.mentions):
            if mention.ner_tag != self._activity_data:
                continue

            activity_index = self.get_next_index_of_mention_with_tag(document, i - 1,
                                                                     [self._activity], search_backwards=True)

            if activity_index is None or document.mentions[activity_index].sentence_index != mention.sentence_index:
                activity_index = self.get_next_index_of_mention_with_tag(document, i + 1, [self._activity])

            if activity_index is None or document.mentions[activity_index].sentence_index != mention.sentence_index:
                continue

            head_mention = document.mentions[activity_index]
            head_entity = document.entity_index_for_mention(head_mention)
            tail_entity = document.entity_index_for_mention(mention)

            r = data.Relation(
                head_entity_index=head_entity,
                tail_entity_index=tail_entity,
                tag=self._tag,
                evidence=list({mention.sentence_index, head_mention.sentence_index})
            )
            relations.append(r)

        return relations
