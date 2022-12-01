import typing

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer


class StandardScaler(object):
    """Impliments standard (mean/std) scaling."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor):
        self.mean = x.mean().item()
        self.std = x.std().item()

        # Having a std == 0 (when all values are the same), breaks training. Just use 1.0 in this case
        if (self.std == 0):
            self.std = 1.0

    def transform(self, x: typing.Union[torch.Tensor, np.ndarray]):

        # Ensure we are in the right floating point format
        if (isinstance(x, torch.Tensor)):
            result = x.to(dtype=torch.float32, copy=True)
        elif (isinstance(x, np.ndarray)):
            result = x.astype(float)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

        result -= self.mean
        result /= self.std
        return result

    def inverse_transform(self, x: torch.Tensor):

        # Ensure we are in the right floating point format
        if (isinstance(x, torch.Tensor)):
            result = x.to(dtype=torch.float32, copy=True)
        elif (isinstance(x, np.ndarray)):
            result = x.astype(float)

        result *= self.std
        result += self.mean
        return result

    def fit_transform(self, x: torch.Tensor):
        self.fit(x)
        return self.transform(x)


class GaussRankScaler(object):
    """
    So-called "Gauss Rank" scaling.
    Forces a transformation, uses bins to perform
        inverse mapping.

    Uses sklearn QuantileTransformer to work.
    """

    def __init__(self):
        self.transformer = QuantileTransformer(output_distribution='normal')

    def fit(self, x):
        x = x.reshape(-1, 1)
        self.transformer.fit(x)

    def transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.transform(x)
        return result.reshape(-1)

    def inverse_transform(self, x):
        x = x.reshape(-1, 1)
        result = self.transformer.inverse_transform(x)
        return result.reshape(-1)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


class NullScaler(object):

    def __init__(self):
        pass

    def fit(self, x):
        pass

    def transform(self, x):
        return x

    def inverse_transform(self, x):
        return x

    def fit_transform(self, x):
        return self.transform(x)
