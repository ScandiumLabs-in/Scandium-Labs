from matbench.bench import MatbenchBenchmark
import numpy as np


class MatbenchEvaluator:
    def __init__(self):
        self.mb = MatbenchBenchmark(autoload=False)
        self.results = {}

    def run(self, tasks=None):
        self.mb.load()
        for task in self.mb.tasks:
            if tasks and task.name not in tasks:
                continue
            train_inputs, train_outputs = task.train
            test_inputs, test_outputs = task.test

            predictions = self._predict(task, train_inputs, train_outputs, test_inputs)
            if predictions is not None:
                task.record(predictions)
                score = task.score
                self.results[task.name] = score
        return self.results

    def _predict(self, task, train_inputs, train_outputs, test_inputs):
        try:
            train_outputs = np.array(train_outputs)
            mean_val = np.mean(train_outputs)
            predictions = np.full(len(test_inputs), mean_val)
            return predictions
        except Exception:
            return None
