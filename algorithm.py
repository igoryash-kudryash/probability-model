from typing import List, Dict
from random import choices
import csv
import matplotlib.pyplot as plt


class ProbsAlgo:
    def __init__(self, path_to_data: str, probs: List[float], n_iter: int) -> None:
        self.true_labels = self.read_file(path_to_data=path_to_data)
        assert sum(probs) == 1
        self.probs = probs
        self.n_iter = n_iter

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    @staticmethod
    def read_file(path_to_data: str) -> List[int]:
        with open(path_to_data, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            labels = [int(row[0]) for row in reader]
        return labels

    def make_predictions(self) -> List[List[int]]:
        predictions = []
        for i in range(self.n_iter):
            pred = []
            for j in range(len(self.true_labels)):
                pred.append(choices([0, 1, 2], self.probs)[0])
            predictions.append(pred)

        assert len(predictions) == self.n_iter
        for pred in predictions:
            assert len(pred) == len(self.true_labels)
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        res = [int(i == j) for i, j in zip(true_labels, predictions)]
        return sum(res) / len(res)

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        tp = [int(i == j == class_number) for i, j in zip(true_labels, predictions)]
        tp_fp = [int(i == class_number) for i in predictions]
        if sum(tp) == 0:
            return 0  # Zero predicted labels of this class
        return sum(tp) / sum(tp_fp)

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        tp = [int(i == j == class_number) for i, j in zip(true_labels, predictions)]
        tp_fn = [int(i == class_number) for i in true_labels]
        if sum(tp_fn) == 0:
            return 1  # Zero true labels of this class
        return sum(tp) / sum(tp_fn)

    def get_final_metrics(self) -> Dict[str, List[float]]:
        final_metrics = dict()
        accuracy = []
        precision0, precision1, precision2 = [], [], []
        recall0, recall1, recall2 = [], [], []
        for i in range(self.n_iter):
            accuracy.append(self.accuracy(self.true_labels, self.preds[i]))

            precision0.append(self.precision(self.true_labels, self.preds[i], 0))
            precision1.append(self.precision(self.true_labels, self.preds[i], 1))
            precision2.append(self.precision(self.true_labels, self.preds[i], 2))

            recall0.append(self.recall(self.true_labels, self.preds[i], 0))
            recall1.append(self.recall(self.true_labels, self.preds[i], 1))
            recall2.append(self.recall(self.true_labels, self.preds[i], 2))

        return dict({
            'accuracy': accuracy,
            'precision0': precision0,
            'precision1': precision1,
            'precision2': precision2,
            'recall0': recall0,
            'recall1': recall1,
            'recall2': recall2
        })

    @staticmethod
    def prefix_sum(lst):
        pr_sum = [lst[0]]
        for i in range(1, len(lst)):
            pr_sum.append(pr_sum[i - 1] + lst[i])
        for i in range(len(pr_sum)):
            pr_sum[i] /= (i + 1)
        return pr_sum

    def plot_and_save_result(self, output_path: str):
        fig, ax = plt.subplots(7, 1, figsize=(8, 18))
        fig.tight_layout()
        plt.rcParams['axes.grid'] = True

        ax[0].plot(range(1, self.n_iter + 1), self.metrics['accuracy'])
        ax[0].set_title('Accuracy')
        ax[0].grid()
        ax[0].set_ylim([0, 1])

        ax[1].plot(range(1, self.n_iter + 1), self.metrics['precision0'])
        ax[1].set_title('Precision for class 0')
        ax[1].grid()
        ax[1].set_ylim([0, 1])

        ax[2].plot(range(1, self.n_iter + 1), self.metrics['precision1'])
        ax[2].set_title('Precision for class 1')
        ax[2].grid()
        ax[2].set_ylim([0, 1])

        ax[3].plot(range(1, self.n_iter + 1), self.metrics['precision2'])
        ax[3].set_title('Precision for class 2')
        ax[3].grid()
        ax[3].set_ylim([0, 1])

        ax[4].plot(range(1, self.n_iter + 1), self.metrics['recall0'])
        ax[4].set_title('Recall for class 0')
        ax[4].grid()
        ax[4].set_ylim([0, 1])

        ax[5].plot(range(1, self.n_iter + 1), self.metrics['recall1'])
        ax[5].set_title('Recall for class 1')
        ax[5].grid()
        ax[5].set_ylim([0, 1])

        ax[6].plot(range(1, self.n_iter + 1), self.metrics['recall2'])
        ax[6].set_title('Recall for class 2')
        ax[6].grid()
        ax[6].set_ylim([0, 1])

        fig.savefig(output_path)
