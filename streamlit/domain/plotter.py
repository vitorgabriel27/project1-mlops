from abc import ABC, abstractmethod
from services.dataframes import BaseDataFrame

class Plotter(ABC):
    dataframe: type[BaseDataFrame]

    def __init__(self, filters: dict = {}):
        self.df = self.dataframe.filter(filters)

    def render(self):
        self.plot()

    @abstractmethod
    def plot(self):
        pass