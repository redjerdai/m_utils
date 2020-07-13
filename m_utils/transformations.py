#
import numpy


class PctTransformer:
    def __init__(self):
        self.f_row = None

    def fit(self, data):
        pass

    def transform(self, data):
        self.f_row = data[[0], :]
        data_lag = numpy.roll(data, shift=1, axis=1)
        data_lag = data_lag[1:, :]
        data_pct = (data_lag / data[:-1, :]) - 1
        return data_pct

    def inverse_transform(self, data_pct):
        data_cp = data_pct + 1
        data_cp = numpy.concatenate((self.f_row, data_cp), axis=0)
        data_cp = data_cp.cumprod(axis=0)
        return data_cp

