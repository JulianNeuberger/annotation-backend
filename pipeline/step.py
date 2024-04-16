import abc
import dataclasses
import math
import pathlib
import time
import typing

import coref
import data
import mentions
import relations
from eval import metrics


@dataclasses.dataclass
class PipelineStepResult:
    predictions: typing.List[data.Document]
    stats: typing.Dict[str, metrics.Stats]


class PipelineStep(abc.ABC):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if not type(other) == PipelineStep:
            return False
        return self._name == other._name

    def run(self, *,
            train_documents: typing.Optional[typing.List[data.Document]] = None,
            test_documents: typing.Optional[typing.List[data.Document]] = None,
            ground_truth_documents: typing.Optional[typing.List[data.Document]] = None,
            training_only: bool = False): 
        if train_documents is not None:
            train_data = [d.copy() for d in train_documents]
        else: 
            train_data = None

        if test_documents is not None:
            test_data = [d.copy() for d in test_documents]
        else: 
            test_data = None

        result = self._run(train_documents=train_data, test_documents=test_data, training_only=training_only)
        return result
        # if not training_only: # No evaluation while training only
        #     print('Running evaluation...')
        #     start = time.time_ns()
        #     stats = self._eval(ground_truth=ground_truth_documents, predictions=result)
        #     print(f'Evaluation done after {(time.time_ns() - start) / 1e6:.1f}ms!')
        #     return PipelineStepResult(result, stats)
    
    # Added for more control
    def _train(self, *, train_documents: typing.List[data.Document]):
        raise NotImplementedError()
    
    # Added for more control
    def _predict(self, *, test_documents: typing.List[data.Document]):
        raise NotImplementedError()

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        raise NotImplementedError()

    def _run(self, *,
             train_documents: typing.Optional[typing.List[data.Document]] = None,
             test_documents: typing.Optional[typing.List[data.Document]] = None, 
             training_only: bool = False) -> typing.Optional[typing.List[data.Document]]:
        raise NotImplementedError()


class CatBoostRelationExtractionStep(PipelineStep):
    def __init__(self, *,
                 name: str,
                 num_trees: int = 1000,
                 negative_sampling_rate: float = 40,
                 context_size: int = 2,
                 depth: int = 8,
                 num_passes: int = 1,
                 learning_rate: float = None,
                 use_pos_features: bool = False,
                 use_embedding_features: bool = False,
                 verbose: bool = False,
                 class_weighting: float = 0.0,
                 seed: int = 42):
        super().__init__(name)
        self.estimator = None
        self._num_trees = num_trees
        self._num_passes = num_passes
        self._negative_sampling = negative_sampling_rate
        self._context_size = context_size
        self._verbose = verbose
        self._seed = seed
        self._depth = depth
        self._use_pos_features = use_pos_features
        self._use_embedding_features = use_embedding_features
        self._learning_rate = learning_rate
        self._class_weighting = class_weighting

    def _eval(self, *, predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(predicted_documents=predictions,
                                         ground_truth_documents=ground_truth,
                                         verbose=self._verbose)

    def _run(self, *, train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document], training_only: bool = False) -> typing.Optional[typing.List[data.Document]]:
        #self._train(train_documents=train_documents)
        if training_only:
            self._train(train_documents=train_documents)
            print('Training finished')
        else:
            return self._predict(test_documents=test_documents)

    def _train(self, *, train_documents: typing.List[data.Document]):
        ner_tags = ['Activity', 'Actor', 'Activity Data', 'Condition Specification',
                    'Further Specification', 'AND Gateway', 'XOR Gateway']
        relation_tags = ['Flow', 'Uses', 'Actor Performer', 'Actor Recipient', 'Further Specification', 'Same Gateway']
        class_weights = {t.lower(): 0 for t in relation_tags}
        if self._class_weighting != 0.0:
            for d in train_documents:
                for r in d.relations:
                    class_weights[r.tag.lower()] += 1
            num_samples = sum(class_weights.values())
            num_classes = len(relation_tags)
            class_weights = {k: num_samples / (num_classes * v) for k, v in class_weights.items()}
            class_weights = {k: math.pow(v, 1 / self._class_weighting) for k, v in class_weights.items()}
        else:
            class_weights = {k: 1.0 for k, v in class_weights.items()}
        print(f'Using class weights {class_weights}')
        if self.estimator is None:
            self.estimator = relations.CatBoostRelationEstimator(negative_sampling_rate=self._negative_sampling,
                                                            num_trees=self._num_trees,
                                                            use_pos_features=self._use_pos_features,
                                                            use_embedding_features=self._use_embedding_features,
                                                            num_passes=self._num_passes,
                                                            context_size=self._context_size,
                                                            relation_tags=relation_tags,
                                                            ner_tags=ner_tags,
                                                            name=self._name,
                                                            seed=self._seed,
                                                            depth=self._depth,
                                                            learning_rate=self._learning_rate,
                                                            class_weights=class_weights,
                                                            verbose=True)
        self.estimator.train(train_documents)
    
    def _predict(self, test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        if self.estimator is None: 
            config = {
                "negative_sampling_rate": self._negative_sampling,
                "num_trees": self._num_trees,
                "use_pos_features": self._use_pos_features,
                "use_embedding_features": self._use_embedding_features,
                "num_passes": self._num_passes,
                "context_size": self._context_size,
                "relation_tags": ['Flow', 'Uses', 'Actor Performer', 'Actor Recipient', 'Further Specification', 'Same Gateway'],
                "ner_tags": ['Activity', 'Actor', 'Activity Data', 'Condition Specification',
                    'Further Specification', 'AND Gateway', 'XOR Gateway'],
                "name": self._name,
                "seed": self._seed,
                "depth": self._depth,
                "learning_rate": self._learning_rate,
                "verbose": True
            }
            self.estimator = relations.CatBoostRelationEstimator.load_model(config)
        test_documents = [d.copy(clear_relations=True) for d in test_documents]
        return self.estimator.predict(test_documents)

class NeuralRelationExtraction(PipelineStep):
    def __init__(self, name: str, negative_sampling_rate: float,
                 verbose: bool = False, seed: int = 42):
        super().__init__(name)
        self._negative_sampling = negative_sampling_rate
        self._verbose = verbose
        self._seed = seed

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth,
                                         verbose=self._verbose)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document], training_only: bool = False) -> typing.List[data.Document]:
        ner_tags = ['Activity', 'Actor', 'Activity Data', 'Condition Specification',
                    'Further Specification', 'AND Gateway', 'XOR Gateway']
        relation_tags = ['Flow', 'Uses', 'Actor Performer', 'Actor Recipient', 'Further Specification', 'Same Gateway']
        estimator = relations.NeuralRelationEstimator(
            checkpoint='allenai/longformer-base-4096',
            entity_tags=ner_tags,
            relation_tags=relation_tags
        )
        estimator.train(train_documents)
        test_documents = [d.copy(clear_relations=True) for d in test_documents]
        return estimator.predict(test_documents)


class RuleBasedRelationExtraction(PipelineStep):
    def _eval(self, *, predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.relation_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth)

    def _run(self, *, train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document], training_only: bool = False) -> typing.List[data.Document]:
        activity = 'Activity'
        actor = 'Actor'
        activity_data = 'Activity Data'
        condition = 'Condition Specification'
        further_spec = 'Further Specification'
        and_gateway = 'AND Gateway'
        xor_gateway = 'XOR Gateway'

        flow = 'Flow'
        uses = 'Uses'
        performer = 'Actor Performer'
        recipient = 'Actor Recipient'
        further_spec_relation = 'Further Specification'
        same_gateway = 'Same Gateway'

        extractor = relations.RuleBasedRelationEstimator([
            relations.rules.SameGatewayRule(triggering_elements=[xor_gateway, and_gateway], target_tag=same_gateway),
            relations.rules.GatewayActivityRule(gateway_tags=[and_gateway, xor_gateway], activity_tag=activity,
                                                same_gateway_tag=same_gateway, flow_tag=flow),
            relations.rules.SequenceFlowsRule(triggering_elements=[activity, xor_gateway, and_gateway, condition],
                                              target_tag=flow),
            relations.rules.ActorPerformerRecipientRule(actor_tag=actor, activity_tag=activity,
                                                        performer_tag=performer, recipient_tag=recipient),
            relations.rules.FurtherSpecificationRule(further_specification_element_tag=further_spec,
                                                     further_specification_relation_tag=further_spec_relation,
                                                     activity_tag=activity),
            relations.rules.UsesRelationRule(activity_data_tag=activity_data, activity_tag=activity,
                                             uses_relation_tag=uses)
        ])
        test_documents = [d.copy(clear_relations=True) for d in test_documents]
        return extractor.predict(test_documents)


class CrfMentionEstimatorStep(PipelineStep):
    def __init__(self, name: str):
        super().__init__(name)
        self.estimator = None

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.mentions_f1_stats(predicted_documents=predictions, ground_truth_documents=ground_truth)
    
    def _train(self, *, train_documents: typing.List[data.Document]):
        if self.estimator is None:
            self.estimator = mentions.ConditionalRandomFieldsEstimator(pathlib.Path(f'api/models/crf/{self._name}'))
        self.estimator.train(train_documents)

    def _predict(self, test_documents: typing.List[data.Document]) -> typing.List[data.Document]:
        if self.estimator is None:
            print('Here!')
            self.estimator = mentions.ConditionalRandomFieldsEstimator.load_model(self._name)
        mention_extraction_input = [d.copy(clear_mentions=True) for d in test_documents]
        return self.estimator.predict(mention_extraction_input)

    def _run(self, *,
             train_documents: typing.Optional[typing.List[data.Document]] = None,
             test_documents: typing.Optional[typing.List[data.Document]] = None, 
             training_only: bool = False) -> typing.Optional[typing.List[data.Document]]:
        #self._train(train_documents= train_documents)
        if training_only:
            self._train(train_documents= train_documents)
            print("Training finished.")
        else:
            return self._predict(test_documents=test_documents)
    

class NeuralCoReferenceResolutionStep(PipelineStep):
    def __init__(self, name: str, resolved_tags: typing.List[str],
                 ner_strategy: str, mention_overlap: float, cluster_overlap: float):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._ner_strategy = ner_strategy
        self._mention_overlap = mention_overlap
        self._cluster_overlap = cluster_overlap

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document]) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(predicted_documents=predictions,
                                       only_tags=self._resolved_tags,
                                       min_num_mentions=2,
                                       ground_truth_documents=ground_truth)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document], training_only: bool = False) -> typing.List[data.Document]:
        test_documents = [d.copy(clear_entities=True) for d in test_documents]
        solver = coref.NeuralCoRefSolver(self._resolved_tags,
                                         ner_tag_strategy=self._ner_strategy,
                                         min_mention_overlap=self._mention_overlap,
                                         min_cluster_overlap=self._cluster_overlap)
        return solver.resolve_co_references(test_documents)


class NaiveCoReferenceResolutionStep(PipelineStep):
    def __init__(self, name: str, resolved_tags: typing.List[str], mention_overlap: float):
        super().__init__(name)
        self._resolved_tags = resolved_tags
        self._mention_overlap = mention_overlap

    def _eval(self, *,
              predictions: typing.List[data.Document],
              ground_truth: typing.List[data.Document], training_only: bool = False) -> typing.Dict[str, metrics.Stats]:
        return metrics.entity_f1_stats(predicted_documents=predictions,
                                       only_tags=self._resolved_tags,
                                       min_num_mentions=2,
                                       ground_truth_documents=ground_truth)

    def _run(self, *,
             train_documents: typing.List[data.Document],
             test_documents: typing.List[data.Document], training_only: bool = False) -> typing.List[data.Document]:
        test_documents = [d.copy(clear_entities=True) for d in test_documents]
        solver = coref.NaiveCoRefSolver(self._resolved_tags, min_mention_overlap=self._mention_overlap)
        return solver.resolve_co_references(test_documents)
