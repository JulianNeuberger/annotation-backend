import dataclasses
import time
import typing

import data
from pipeline.step import PipelineStep, PipelineStepResult
from pipeline.step import CatBoostRelationExtractionStep, RuleBasedRelationExtraction, NeuralRelationExtraction
from pipeline.step import CrfMentionEstimatorStep
from pipeline.step import NeuralCoReferenceResolutionStep, NaiveCoReferenceResolutionStep


@dataclasses.dataclass
class PipelineResult:
    step_results: typing.Dict[PipelineStep, PipelineStepResult]


class Pipeline:
    def __init__(self, steps: typing.List[PipelineStep], name: str):
        self._name = name
        self._steps = steps

    @property
    def name(self):
        return self._name

    @property
    def step_names(self):
        return [s.name for s in self._steps]

    @property
    def steps(self):
        return [s for s in self._steps]

    def run(self, *,
            train_documents: typing.List[data.Document],
            test_documents: typing.List[data.Document],
            ground_truth_documents: typing.List[data.Document]) -> PipelineResult:
        print(f'Running {self.description()}')
        pipeline_result = PipelineResult({})

        train_documents = [d.copy() for d in train_documents]
        test_documents = [d.copy() for d in test_documents]

        for s in self._steps:
            result = s.run(train_documents=train_documents,
                           test_documents=test_documents,
                           ground_truth_documents=ground_truth_documents)
            pipeline_result.step_results[s] = result
            test_documents = [d.copy() for d in result.predictions]

        return pipeline_result

    def description(self):
        return f'pipeline with {len(self.steps)} steps: {", ".join(self.step_names)}'
