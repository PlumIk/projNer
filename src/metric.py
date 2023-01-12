from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from torchmetrics.functional import symmetric_mean_absolute_percentage_error
import torch


@dataclass
class MetricBundle:
    RMSE: float
    SMAPE: float
    R2: float

    @classmethod
    def create(cls, actual: np.ndarray, predicted: np.ndarray) -> 'MetricBundle':
        rmse = mean_squared_error(squared=True, y_true=actual, y_pred=predicted)
        r2 = r2_score(y_true=actual, y_pred=predicted)
        smape = symmetric_mean_absolute_percentage_error(preds=torch.as_tensor(data=predicted),
                                                         target=torch.as_tensor(data=actual)
                                                         )
        return MetricBundle(RMSE=rmse, R2=r2, SMAPE=float(smape))

    def __str__(self):
        string = (f"RMSE: {self.RMSE}\\\n"
                  f"r2_score: {self.R2}\\\n"
                  f"smape: {self.SMAPE}")
        return string