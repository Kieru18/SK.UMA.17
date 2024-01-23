from abc import ABC, abstractmethod


class RegressionModel(ABC):
    @abstractmethod
    def fit(self, X, y) -> None:
        pass

    @abstractmethod
    def predict(self, X) -> float:
        pass
