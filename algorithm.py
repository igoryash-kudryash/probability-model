from typing import List, Dict


class ProbsAlgo:
    def __init__(self, path_to_data: str, probs: List[float], n_iter: int) -> None:
        self.true_labels = self.read_file(path_to_data)
        self.probs = probs
        self.n_iter = n_iter

    def read_file(self, path_to_data: str) -> List[int]:
        pass

    def make_predictions(self) -> List[List[int]]:
        pass

    @staticmethod
    def accuracy(true_labels: List[int], predictions: List[int]) -> List[float]:
        pass

    @staticmethod
    def precision(true_labels: List[int], predictions: List[int], class_number: int) -> List[float]:
        pass

    @staticmethod
    def recall(true_labels: List[int], predictions: List[int], class_number: int) -> List[float]:
        pass

    def get_final_metrics(self) -> Dict[str: List[float]]:
        pass

    def plot_save_result(self, output_path: str):
        pass