from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseDiscoveryMethod(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        ...
