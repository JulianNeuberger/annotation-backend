import dataclasses

import typing


@dataclasses.dataclass
class Document:
    id: str
    category: str
    text: str
    name: str
    tokens: typing.List["Token"] = dataclasses.field(default_factory=list)
    mentions: typing.List["Mention"] = dataclasses.field(default_factory=list)
    entities: typing.List["Entity"] = dataclasses.field(default_factory=list)
    relations: typing.List["Relation"] = dataclasses.field(default_factory=list)

    @property
    def sentences(self) -> typing.List[typing.List["Token"]]:
        ret = []
        last_sentence_id: typing.Optional[int] = None
        for token in self.tokens:
            if token.sentence_index != last_sentence_id:
                last_sentence_id = token.sentence_index
                ret.append([])
            ret[-1].append(token)
        return ret

    def relation_exists_between(
        self, head_mention_index: int, tail_mention_index: int
    ) -> bool:
        for r in self.relations:
            if (
                r.head_mention_index == head_mention_index
                and r.tail_mention_index == tail_mention_index
            ):
                return True
        return False

    def get_relations_by_mention(
        self, mention_index: int, only_head=False, only_tail=False
    ) -> typing.List["Relation"]:
        if only_tail and only_head:
            raise ValueError(
                "The mention can not be only head and tail at the same time!"
            )

        ret = []
        for relation in self.relations:
            is_head = relation.head_mention_index == mention_index
            if only_head and is_head:
                ret.append(relation)
                continue

            is_tail = relation.tail_mention_index == mention_index
            if only_tail and is_tail:
                ret.append(relation)
                continue

            if is_tail or is_head:
                ret.append(relation)
        return ret

    def contains_relation(self, relation: "Relation") -> bool:
        return relation.to_tuple(self) in [e.to_tuple(self) for e in self.relations]

    def contains_entity(self, entity: "Entity") -> bool:
        return entity.to_tuple(self) in [e.to_tuple(self) for e in self.entities]

    def entity_index_for_mention_index(self, mention_index: int) -> int:
        for i, e in enumerate(self.entities):
            if mention_index in e.mention_indices:
                return i
        print(mention_index)
        print(self.entities)
        mention = self.mentions[mention_index]
        raise ValueError(
            f"Document contains no entity using mention {mention}, "
            f"which should not happen, but can happen, "
            f"if entities are not properly resolved"
        )

    def mention_index(self, mention: "Mention") -> int:
        mentions_as_tuples = [m.to_tuple(self) for m in self.mentions]
        mention_index = mentions_as_tuples.index(mention.to_tuple(self))
        return mention_index

    def entity_index_for_mention(self, mention: "Mention") -> int:
        mention_index = self.mention_index(mention)
        return self.entity_index_for_mention_index(mention_index)

    def get_mentions_for_token(self, token: "Token") -> typing.List["Mention"]:
        matched = []
        for mention in self.mentions:
            index_in_document = token.index_in_document
            if index_in_document in mention.token_document_indices:
                matched.append(mention)
        return matched

    def sentence_index_for_token_index(self, token_index: int) -> int:
        assert 0 <= token_index < len(self.tokens)
        return self.tokens[token_index].sentence_index

    def copy(
        self,
        clear_mentions: bool = False,
        clear_relations: bool = False,
        clear_entities: bool = False,
    ) -> "Document":
        return Document(
            name=self.name,
            text=self.text,
            id=self.id,
            category=self.category,
            tokens=[t.copy() for t in self.tokens],
            mentions=[] if clear_mentions else [m.copy() for m in self.mentions],
            relations=[] if clear_relations else [r.copy() for r in self.relations],
            entities=[] if clear_entities else [e.copy() for e in self.entities],
        )

    def to_json_serializable(self):
        return {
            "text": self.text,
            "name": self.name,
            "id": self.id,
            "category": self.category,
            "tokens": [s.to_json_serializable() for s in self.tokens],
            "mentions": [m.to_json_serializable() for m in self.mentions],
            "entities": [e.to_json_serializable() for e in self.entities],
            "relations": [r.to_json_serializable() for r in self.relations],
        }


@dataclasses.dataclass
class Mention:
    ner_tag: str
    token_document_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def get_sentence_index(self, doc: Document) -> int:
        unique_sentence_indices = set(
            [doc.tokens[i].sentence_index for i in self.token_document_indices]
        )
        assert len(unique_sentence_indices) == 1
        return doc.tokens[self.token_document_indices[0]].sentence_index

    def get_tokens(self, document: Document) -> typing.List["Token"]:
        return [document.tokens[i] for i in self.token_document_indices]

    def to_tuple(self, *args) -> typing.Tuple:
        return (self.ner_tag.lower(),) + tuple(self.token_document_indices)

    def text(self, document: Document):
        return " ".join([t.text for t in self.get_tokens(document)])

    def contains_token(self, token: "Token", document: "Document") -> bool:
        for own_token_idx in self.token_document_indices:
            own_token = document.tokens[own_token_idx]
            if own_token.index_in_document == token.index_in_document:
                return True
        return False

    def pretty_print(self, document: Document):
        return f"{self.text(document)} ({self.ner_tag}, {min(self.token_document_indices)}-{max(self.token_document_indices)})"

    def copy(self) -> "Mention":
        return Mention(
            ner_tag=self.ner_tag,
            token_document_indices=[i for i in self.token_document_indices],
        )

    def to_json_serializable(self):
        return {
            "type": self.ner_tag,
            "tokenDocumentIndices": self.token_document_indices,
        }


@dataclasses.dataclass
class Entity:
    mention_indices: typing.List[int] = dataclasses.field(default_factory=list)

    def to_tuple(self, document: Document) -> typing.Tuple:
        mentions = [document.mentions[i] for i in self.mention_indices]

        return (frozenset([m.to_tuple(document) for m in mentions]),)

    def copy(self) -> "Entity":
        return Entity(mention_indices=[i for i in self.mention_indices])

    def get_tag(self, document: Document) -> str:
        tags = list(set([document.mentions[m].ner_tag for m in self.mention_indices]))
        assert len(tags) == 1
        return tags[0]

    def pretty_print(self, document: Document):
        return f"Entity [{[document.mentions[m].pretty_print(document) for m in self.mention_indices]}]"

    def to_json_serializable(self):
        return {"mentionIndices": self.mention_indices}


@dataclasses.dataclass
class Relation:
    head_mention_index: int
    tail_mention_index: int
    tag: str

    def copy(self) -> "Relation":
        return Relation(
            head_mention_index=self.head_mention_index,
            tail_mention_index=self.tail_mention_index,
            tag=self.tag,
        )

    def to_tuple(self, document: Document) -> typing.Tuple:
        return (
            self.tag.lower(),
            document.entities[self.head_mention_index].to_tuple(document),
            document.entities[self.tail_mention_index].to_tuple(document),
        )

    def pretty_print(self, document: Document):
        head_mention = document.mentions[self.head_mention_index]
        tail_mention = document.mentions[self.tail_mention_index]

        return (
            f"[{head_mention.pretty_print(document)}]"
            f"--[{self.tag}]-->"
            f"[{tail_mention.pretty_print(document)}]"
        )

    def to_json_serializable(self):
        return {
            "headMentionIndex": self.head_mention_index,
            "tailMentionIndex": self.tail_mention_index,
            "type": self.tag,
        }


@dataclasses.dataclass
class Token:
    text: str
    index_in_document: int
    pos_tag: str
    sentence_index: int

    def index_in_sentence(self, doc: "Document") -> int:
        index_in_sentence = 0
        for token_index, token in enumerate(doc.tokens):
            if token == self:
                return index_in_sentence
            if token.sentence_index == self.sentence_index:
                index_in_sentence += 1
        raise IndexError(
            f"Could not find token in sentence with id {self.sentence_index}."
        )

    def copy(self) -> "Token":
        return Token(
            text=self.text,
            index_in_document=self.index_in_document,
            pos_tag=self.pos_tag,
            sentence_index=self.sentence_index,
        )

    def to_json_serializable(self):
        return {
            "text": self.text,
            "indexInDocument": self.index_in_document,
            "posTag": self.pos_tag,
            "sentenceIndex": self.sentence_index,
        }
