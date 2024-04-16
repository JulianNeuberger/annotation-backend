import dataclasses
import json
import os
import time
import typing

import pandas as pd
import tqdm

import data
import pipeline
from eval import metrics

FoldStats = typing.List[typing.Dict[str, metrics.Stats]]


@dataclasses.dataclass
class PrintableScores:
    scores_by_tag: typing.Dict[str, metrics.Scores]
    overall_scores: metrics.Scores

    def __add__(self, other):
        scores_by_tag = {}
        for k in list(self.scores_by_tag.keys()) + list(other.scores_by_tag.keys()):
            if k not in self.scores_by_tag:
                scores_by_tag[k] = other.scores_by_tag[k]
            elif k not in other.scores_by_tag:
                scores_by_tag[k] = self.scores_by_tag[k]
            else:
                scores_by_tag[k] = self.scores_by_tag[k] + other.scores_by_tag[k]

        return PrintableScores(
            scores_by_tag=scores_by_tag,
            overall_scores=self.overall_scores + other.overall_scores
        )

    def __truediv__(self, other):
        return PrintableScores(
            scores_by_tag={
                k: (v / other) for k, v in self.scores_by_tag.items()
            },
            overall_scores=self.overall_scores / other
        )


def cross_validate_pipeline(p: pipeline.Pipeline, *,
                            train_folds: typing.List[typing.List[data.Document]],
                            test_folds: typing.List[typing.List[data.Document]],
                            save_results: bool = False,
                            dump_predictions_dir: str = None):
    assert len(train_folds) == len(test_folds)
    pipeline_results = []
    for n_fold, (train_fold, test_fold) in tqdm.tqdm(enumerate(zip(train_folds, test_folds)),
                                                     total=len(train_folds), desc='cross validation fold'):
        start = time.time_ns()
        ground_truth = [d.copy() for d in test_fold]
        print(f'copy of {len(test_fold)} documents took {(time.time_ns() - start) / 1e6:.4f}ms')

        pipeline_result = p.run(train_documents=train_fold,
                                test_documents=test_fold,
                                ground_truth_documents=ground_truth)
        pipeline_results.append(pipeline_result)
    res = accumulate_pipeline_results(pipeline_results)
    if dump_predictions_dir is not None:
        for i, pipeline_result in enumerate(pipeline_results):
            json_data = [p.to_json_serializable() for p in pipeline_result.step_results[p.steps[-1]].predictions]
            os.makedirs(dump_predictions_dir, exist_ok=True)
            with open(os.path.join(dump_predictions_dir, f'fold-{i}.json'), 'w', encoding='utf8') as f:
                json.dump(json_data, f)

    if save_results:
        df_persistence = 'experiments.pkl'
        if os.path.isfile(df_persistence):
            df: pd.DataFrame = pd.read_pickle(df_persistence)
        else:
            df = pd.DataFrame(columns=['experiment_name', 'tag', 'p', 'r', 'f1']).set_index(['experiment_name', 'tag'])
        final_result = res[p.steps[-1]]

        new_rows = []
        for tag, value in final_result.scores_by_tag.items():
            new_rows.append({
                'experiment_name': p.name,
                'tag': tag,
                'p': value.p,
                'r': value.r,
                'f1': value.f1
            })
        new_rows.append({
            'experiment_name': p.name,
            'tag': 'overall',
            'p': final_result.overall_scores.p,
            'r': final_result.overall_scores.r,
            'f1': final_result.overall_scores.f1
        })

        new_rows_df = pd.DataFrame.from_records(new_rows).set_index(['experiment_name', 'tag'])

        df = new_rows_df.combine_first(df)

        pd.to_pickle(df, df_persistence)

    print_pipeline_results(p, res)
    return res


def f1_stats_from_pipeline_result(result: pipeline.PipelineResult,
                                  average_mode: str) -> typing.Dict[pipeline.PipelineStep, PrintableScores]:
    res: typing.Dict[pipeline.PipelineStep, PrintableScores] = {}
    for pipeline_step, step_results in result.step_results.items():
        scores_by_ner = {
            k: metrics.Scores.from_stats(v) for k, v in step_results.stats.items()
        }
        if average_mode == 'micro':
            combined_stats = sum(step_results.stats.values(), metrics.Stats(0, 0, 0))
            overall_scores = metrics.Scores.from_stats(combined_stats)
        elif average_mode == 'macro':
            macro_scores = sum(scores_by_ner.values(), metrics.Scores(0, 0, 0)) / len(scores_by_ner)
            overall_scores = macro_scores
        else:
            raise ValueError(f'Unknown averaging mode {average_mode}.')
        res[pipeline_step] = PrintableScores(scores_by_tag=scores_by_ner, overall_scores=overall_scores)

    return res


def accumulate_pipeline_results(pipeline_results: typing.List[pipeline.PipelineResult],
                                averaging_mode: str = 'micro') -> typing.Dict[pipeline.PipelineStep, PrintableScores]:
    scores = []
    for pipeline_result in pipeline_results:
        scores.append(f1_stats_from_pipeline_result(pipeline_result, averaging_mode))

    num_results = len(scores)
    steps = pipeline_results[0].step_results.keys()

    return {
        step: sum([s[step] for s in scores], PrintableScores({}, metrics.Scores(0, 0, 0))) / num_results
        for step in steps
    }


def print_pipeline_results(p: pipeline.Pipeline,
                           res: typing.Dict[pipeline.PipelineStep, PrintableScores]):
    print(f'=== {p.name} {"=" * (47 - len(p.name))}')
    for step, scores in res.items():
        print(f'--- {step.name} {"-" * (47 - len(step.name))}')
        print_scores(scores.scores_by_tag, scores.overall_scores)


def scenario_4_5_6():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    start = time.time()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-rule-based', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
            pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        dump_predictions_dir='predictions/rule-based',
        save_results=True
    )

    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-cat-boost', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=2000, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        dump_predictions_dir='predictions/ours',
        save_results=True
    )
    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)


def ablation_studies():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    start = time.time()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='naive-coref-rule-based', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
                                                    resolved_tags=['Actor', 'Activity Data'],
                                                    mention_overlap=.8),
            pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )

    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='naive-coref-cat-boost', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
                                                    resolved_tags=['Actor', 'Activity Data'],
                                                    mention_overlap=.8),
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=2000, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )
    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='rule-based-isolated', steps=[
            pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )

    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='cat-boost-isolated', steps=[
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=2000, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )
    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='co-ref-only-rule-based', steps=[
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
            pipeline.RuleBasedRelationExtraction(name='rule-based relation extraction')
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )

    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')

    print()
    print('---')
    print(flush=True)

    start = time.time()
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='co-ref-only-cat-boost', steps=[
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=2000, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )
    print(f'Pipeline took {(time.time() - start) / 60.:.2f} minutes')


def accumulate(left: typing.Dict[str, metrics.Stats],
               right: typing.Dict[str, metrics.Stats]) -> typing.Dict[str, metrics.Stats]:
    key_set = set(left.keys()).union(set(right.keys()))
    return {
        ner_tag: left.get(ner_tag, metrics.Stats(1, 1, 1)) + right.get(ner_tag, metrics.Stats(1, 1, 1))
        for ner_tag in key_set
    }


def print_scores(scores_by_ner: typing.Dict[str, metrics.Scores],
                 overall_score: metrics.Scores,
                 order: typing.List[str] = None):
    len_ner_tags = max([len(t) for t in scores_by_ner.keys()])

    if order is None:
        order = list(scores_by_ner.keys())

    print(f'{" " * (len_ner_tags - 2)}Tag |   P     |   R     |   F1    ')
    print(f'{"=" * (len_ner_tags + 2)}+=========+=========+========')

    for ner_tag in order:
        if ner_tag not in scores_by_ner:
            continue
        score = scores_by_ner[ner_tag]
        print(f' {ner_tag: >{len_ner_tags}} '
              f'| {score.p: >7.2%} '
              f'| {score.r: >7.2%} '
              f'| {score.f1: >7.2%}')
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')

    print(f' {"Overall": >{len_ner_tags}} '
          f'| {overall_score.p: >7.2%} '
          f'| {overall_score.r: >7.2%} '
          f'| {overall_score.f1: >7.2%}')
    print(f'{"-" * (len_ner_tags + 2)}+---------+---------+---------')


def catboost_debug():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-cat-boost', steps=[
            # pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            # pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
            #                                         resolved_tags=['Actor', 'Activity Data'],
            #                                         mention_overlap=.8)
            # pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
            #                                          resolved_tags=['Actor', 'Activity Data'],
            #                                          cluster_overlap=.5,
            #                                          mention_overlap=.5,
            #                                          ner_strategy='frequency'),
            pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=2)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=False,
        dump_predictions_dir='predictions/test/'
    )


def neural_rel_debug():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    print('Running pipeline with neural entity resolution, and cat-boost relation extraction')
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-cat-boost', steps=[
            pipeline.NeuralRelationExtraction(name='neural relation extraction', negative_sampling_rate=40.0)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=False
    )


def coref_debug():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    print('neural entity resolution on perfect mentions')
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-cat-boost', steps=[
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.8,
                                                     ner_strategy='frequency')
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=False
    )

    print('naive entity resolution on perfect mentions')
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='complete-cat-boost', steps=[
            pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
                                                    resolved_tags=['Actor', 'Activity Data'],
                                                    mention_overlap=.8)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=False
    )


def scenario_1():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='mention-extraction-only', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=False
    )


def scenario_2_3():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='naive-coref-perfect-mentions', steps=[
            pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
                                                    resolved_tags=['Actor', 'Activity Data'],
                                                    mention_overlap=.8)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='naive-coref', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NaiveCoReferenceResolutionStep(name='naive coreference resolution',
                                                    resolved_tags=['Actor', 'Activity Data'],
                                                    mention_overlap=.8)
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )

    print()
    print('---')
    print(flush=True)

    cross_validate_pipeline(
        p=pipeline.Pipeline(name='neural-coref-perfect-mentions', steps=[
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )
    cross_validate_pipeline(
        p=pipeline.Pipeline(name='neural-coref', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction'),
            pipeline.NeuralCoReferenceResolutionStep(name='neural coreference resolution',
                                                     resolved_tags=['Actor', 'Activity Data'],
                                                     cluster_overlap=.5,
                                                     mention_overlap=.5,
                                                     ner_strategy='frequency'),
        ]),
        train_folds=train_folds,
        test_folds=test_folds,
        save_results=True
    )


def main():
    # ablation_studies()
    # catboost_debug()
    # coref_debug()
    # neural_rel_debug()

     scenario_1()
   #scenario_2_3()
    # scenario_4_5_6()


if __name__ == '__main__':
    main()
