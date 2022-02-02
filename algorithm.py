from typing import List, Dict
from random import choices
import matplotlib.pyplot as plt
from collections import defaultdict
import os


class ProbsAlgo:
    def __init__(self, path_to_data: str, probs: List[float], n_iter: int) -> None:
        self.true_labels = self.read_file(path_to_data=path_to_data)
        assert sum(probs) == 1, 'Probs do not sum to one!'
        self.probs = probs
        self.n_iter = n_iter

        self.preds = self.make_predictions()
        self.metrics = self.get_final_metrics()

    @staticmethod
    def read_file(path_to_data: str) -> List[int]:
        """

        :param path_to_data: path to the data containing true labels
        :return: true labels
        """
        if not os.path.isfile(path_to_data):
            raise FileNotFoundError
        with open(path_to_data, newline='') as csvfile:
            labels = [int(i) for i in csvfile]
        return labels

    def make_predictions(self) -> List[List[int]]:
        """

        :return: predictions of the model
        """
        predictions = []
        for i in range(self.n_iter):
            pred = []
            for j in range(len(self.true_labels)):
                pred.append(choices([0, 1, 2], self.probs)[0])
            predictions.append(pred)

        assert len(predictions) == self.n_iter, 'Wrong size!'
        for pred in predictions:
            assert len(pred) == len(self.true_labels), 'Wrong size!'
        return predictions

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> float:
        """

        :param true_labels: list of true labels
        :param predictions: list of predicted labels
        :return: accuracy
        """
        assert len(true_labels) == len(predictions), 'Sizes of true labels and predictions do not match'
        res = (i == j for i, j in zip(true_labels, predictions))
        return sum(res) / len(true_labels)

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        """

        :param true_labels: list of true labels
        :param predictions: list of predicted labels
        :param class_number: number of class
        :return: precision for class_number
        """
        assert len(true_labels) == len(predictions), 'Sizes of true labels and predictions do not match'
        tp = (i == j == class_number for i, j in zip(true_labels, predictions))
        tp_fp = (i == class_number for i in predictions)
        s = sum(tp)

        # assert добавить в recall
        # почитать про генераторы
        # документация функции в """ - - - """
        # написать генератор вместо списка, который возвращает флоат
        # в get_final_metrics список list(self(...)))

        # не создавать в ретерне новый словарь, а применить функцию к готовым значениями
        # в get_final_metrics

        # иду по списку - запоминаю сумму в предыдщую сумму

        # сделать проверку, что путь до даты - адекватная вещь через raise Exception

        if s == 0:
            return 0  # Zero predicted labels of this class
        else:
            return s / sum(tp_fp)

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> float:
        """

        :param true_labels: list of true labels
        :param predictions: list of predicted labels
        :param class_number: number of class
        :return: recall for class_number
        """
        assert len(true_labels) == len(predictions), 'Sizes of true labels and predictions do not match'
        tp = (i == j == class_number for i, j in zip(true_labels, predictions))
        tp_fn = (i == class_number for i in true_labels)
        s = sum(tp_fn)
        if s == 0:
            return 1  # Zero true labels of this class
        else:
            return sum(tp) / s

    @staticmethod
    def prefix_sum(lst: List[float]) -> List[float]:
        """

        :param lst: list of values
        :return: cumulative average
        """
        pr_sum = [lst[0]]
        for i in range(1, len(lst)):
            pr_sum.append(pr_sum[i - 1] + lst[i])
        for i in range(len(pr_sum)):
            pr_sum[i] /= (i + 1)
        return pr_sum

    def get_final_metrics(self) -> Dict[str, List[float]]:
        """

        :return: dict of final metrics
        """
        final_metrics = defaultdict(list)

        for i in range(self.n_iter):
            final_metrics['accuracy'].append(self.accuracy(self.true_labels, self.preds[i]))

            final_metrics['precision0'].append(self.precision(self.true_labels, self.preds[i], 0))
            final_metrics['precision1'].append(self.precision(self.true_labels, self.preds[i], 1))
            final_metrics['precision2'].append(self.precision(self.true_labels, self.preds[i], 2))

            final_metrics['recall0'].append(self.recall(self.true_labels, self.preds[i], 0))
            final_metrics['recall1'].append(self.recall(self.true_labels, self.preds[i], 1))
            final_metrics['recall2'].append(self.recall(self.true_labels, self.preds[i], 2))

        return dict({
            'accuracy': self.prefix_sum(final_metrics['accuracy']),
            'precision0': self.prefix_sum(final_metrics['precision0']),
            'precision1': self.prefix_sum(final_metrics['precision1']),
            'precision2': self.prefix_sum(final_metrics['precision2']),
            'recall0': self.prefix_sum(final_metrics['recall0']),
            'recall1': self.prefix_sum(final_metrics['recall1']),
            'recall2': self.prefix_sum(final_metrics['recall2'])
        })

    def plot_and_save_result(self, output_path: str) -> None:
        """

        :param output_path: path to the output image
        :return: None
        """
        fig, ax = plt.subplots(7, 1, figsize=(8, 18))
        fig.tight_layout()
        plt.rcParams['axes.grid'] = True

        ax[0].plot(range(1, self.n_iter + 1), self.metrics['accuracy'])
        ax[0].set_title('Accuracy')
        ax[0].grid()
        ax[0].set_ylim([-0.05, 1.05])

        ax[1].plot(range(1, self.n_iter + 1), self.metrics['precision0'])
        ax[1].set_title('Precision for class 0')
        ax[1].grid()
        ax[1].set_ylim([-0.05, 1.05])

        ax[2].plot(range(1, self.n_iter + 1), self.metrics['precision1'])
        ax[2].set_title('Precision for class 1')
        ax[2].grid()
        ax[2].set_ylim([-0.05, 1.05])

        ax[3].plot(range(1, self.n_iter + 1), self.metrics['precision2'])
        ax[3].set_title('Precision for class 2')
        ax[3].grid()
        ax[3].set_ylim([-0.05, 1.05])

        ax[4].plot(range(1, self.n_iter + 1), self.metrics['recall0'])
        ax[4].set_title('Recall for class 0')
        ax[4].grid()
        ax[4].set_ylim([-0.05, 1.05])

        ax[5].plot(range(1, self.n_iter + 1), self.metrics['recall1'])
        ax[5].set_title('Recall for class 1')
        ax[5].grid()
        ax[5].set_ylim([-0.05, 1.05])

        ax[6].plot(range(1, self.n_iter + 1), self.metrics['recall2'])
        ax[6].set_title('Recall for class 2')
        ax[6].grid()
        ax[6].set_ylim([-0.05, 1.05])

        fig.savefig(output_path)
