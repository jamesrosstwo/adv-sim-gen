from abc import ABC


class DrivingMetric(ABC):
    def evaluate(self, policy, dataset):
        pass