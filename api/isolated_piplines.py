import json
import random
import typing

from matplotlib import pyplot as plt
import data
from main import accumulate_pipeline_results, print_pipeline_results
from pipeline import Pipeline, PipelineResult
import pipeline


class IsolatedPipline(Pipeline):
    def run(self, *, 
            train_documents: typing.Optional[typing.List[data.Document]] = None,
            test_documents: typing.Optional[typing.List[data.Document]] = None, 
            ground_truth_documents: typing.Optional[typing.List[data.Document]] = None, 
            training_only: bool = True): # Adding a flag to only train the model
        
        print(f'Running {self.description()}')
        if train_documents is not None:
            train_documents = [d.copy() for d in train_documents]
        if test_documents is not None:
            test_documents = [d.copy() for d in test_documents]

        if training_only:
            for s in self._steps:
                s.run(train_documents=train_documents,
                            training_only= training_only)
        
        else:
            pipeline_result = PipelineResult({})
            for s in self._steps:
                result = s.run(test_documents=test_documents,
                            ground_truth_documents=ground_truth_documents)
                pipeline_result.step_results[s] = result
                test_documents = [d.copy() for d in result]
            return pipeline_result    

        print(f"Finished {self.description()}")


def train_ner_pipline():
    training_sets = create_trainsets(provide_training_data())
    print(type(training_sets))

    #for i, training_set in enumerate(training_sets, start=1):

    ner_pd = IsolatedPipline(name=f'mention-extraction model', steps=[
            pipeline.CrfMentionEstimatorStep(name=f'crf mention extraction average model')])
    ner_pd.run(train_documents=training_sets, training_only=True)


def predict_ner_pipline():
    testing_set = provide_test_data()
    ground_truth = [d.copy() for d in testing_set]

    metrics_data = []
    num_docs = []

    for i, value in enumerate(testing_set, start=1):
        ner_pd = IsolatedPipline(name=f'mention-extraction-only model {i}', steps=[
                pipeline.CrfMentionEstimatorStep(name=f'crf mention extraction with model {i} and {4*i} docs')])
        pipeline_result = ner_pd.run(test_documents=testing_set, ground_truth_documents=ground_truth, training_only=False)
        # res = accumulate_pipeline_results([pipeline_result])
        #print(pipeline_result)
        # num_docs_for_model = 4 * i
        # overall_scores = res[ner_pd.steps[-1]].overall_scores
        # metrics_data.append(overall_scores)
        # num_docs.append(num_docs_for_model)

    #f1_scores = [score.f1 * 100 for score in metrics_data]
    #save_f1_data('fold 4', f1_scores, num_docs)
    
    return metrics_data, num_docs
    

def train_re_pipline():
    training_sets = create_trainsets(provide_training_data())
    
    #for i, training_set in enumerate(training_sets, start=1):
    catboost_pd = IsolatedPipline(name='catboost', steps=[pipeline.CatBoostRelationExtractionStep(name=f'cat-boost re average model', use_pos_features=False,
                                                    context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)])
    catboost_pd.run(train_documents=training_sets, training_only=True)


def predict_re_pipline():
    testing_set = provide_test_data()
    ground_truth = [d.copy() for d in testing_set]

    metrics_data = []
    num_docs = []

    for i, value in enumerate(testing_set, start=1):
        catboost_pd = IsolatedPipline(name='catboost', steps=[pipeline.CatBoostRelationExtractionStep(name=f'cat-boost re with model {i} and {4*i} docs', use_pos_features=False,
                                                        context_size=2, num_trees=100, negative_sampling_rate=40.0,
                                                        depth=8, class_weighting=0, num_passes=1)])
        pipeline_result = catboost_pd.run(test_documents=testing_set, ground_truth_documents=ground_truth, training_only=False)
        # res = accumulate_pipeline_results([pipeline_result])
        #print(pipeline_result)
        # num_docs_for_model = 4 * i
        # overall_scores = res[catboost_pd.steps[-1]].overall_scores
        # metrics_data.append(overall_scores)
        # num_docs.append(num_docs_for_model)
    
    # f1_scores = [score.f1 * 100 for score in metrics_data]
    # save_f1_data('fold 4', f1_scores, num_docs)

    return metrics_data, num_docs
 

# Using the existing data for testing purposes
def provide_training_data():
    train_data = data.loader.read_documents_from_json(f'./jsonl/fold_{4}/train.json')
    
    return train_data


def create_trainsets(train_documents: typing.List[data.Document]):
    assert len(train_documents) == 36

    training_sets = []
    for i in range(4, 37, 4): 
        training_sets.append(train_documents[:i])

    assert len(training_sets) == 9

    return training_sets[1]


def provide_test_data():
    test_data = data.loader.read_documents_from_json(f'./jsonl/fold_{4}/test.json')
    
    return test_data


def plot_results(metrics_data, num_docs):
    precision_scores = [score.p * 100 for score in metrics_data]
    recall_scores = [score.r * 100 for score in metrics_data]
    f1_scores = [score.f1 * 100 for score in metrics_data]

    print(num_docs)

    plt.figure(figsize=(10,6))
    plt.plot(num_docs, precision_scores, label='Precision', marker='o')
    plt.plot(num_docs, recall_scores, label='Recall', marker='s')
    plt.plot(num_docs, f1_scores, label='F1 Score', marker='^')

    plt.xlabel('Number of Documents')
    plt.ylabel('Score (%)')
    plt.title('CatBoost Model Performance (100 Num_Trees)')
    plt.legend()
    plt.grid(True)
    plt.show()


def save_f1_data(run_id, f1_scores, num_docs, filename='f1_scores.json'):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[run_id] = {
        'f1_scores': f1_scores,
        'num_docs': num_docs,
    }

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)


def plot_f1_scores():
    with open('f1_scores.json', 'r') as file:
        all_runs = json.load(file)

    plt.figure(figsize=(10, 6))
    for run_id, data in all_runs.items():
        plt.plot(data['num_docs'], data['f1_scores'], label=f'{run_id}', marker='o')

    plt.xlabel('Number of Documents')
    plt.ylabel('F1 Score (%)')
    plt.title('F1 Scores Across Different Runs (CatBoost)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    train_ner_pipline()

    #metrics_data, num_docs = predict_ner_pipline()
    
    train_re_pipline()

    #metrics_data, num_docs = predict_re_pipline()

    #plot_results(metrics_data, num_docs)

    #plot_f1_scores()

if __name__ == '__main__':
    main()