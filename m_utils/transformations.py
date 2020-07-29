#
import numpy
from sklearn.preprocessing import StandardScaler


# trash and should be removed
class PctTransformer:
    def __init__(self):
        self.f_row = None

    def fit(self, data):
        pass

    # are you ok, man? what's this?
    def transform(self, data):
        self.f_row = data[[0], :]
        data_lag = numpy.roll(data, shift=1, axis=1)  # <- really?
        data_lag = data_lag[1:, :]
        data_pct = (data_lag / data[:-1, :]) - 1   # it that a joke, huh?
        return data_pct

    def inverse_transform(self, data_pct):
        data_cp = data_pct + 1
        data_cp = numpy.concatenate((self.f_row, data_cp), axis=0)
        data_cp = data_cp.cumprod(axis=0)
        return data_cp


class LogPctTransformer:
    def __init__(self):
        self.first_row = None
        self.check_row = None
        self.shape = None
        pass

    def fit(self, data):
        self.first_row = data[[0], :]
        self.check_row = data[[0, 1], :]
        self.shape = data.shape
        pass

    def transform(self, data):
        data_log_pct = numpy.log(data[1:, :]) - numpy.log(numpy.roll(data, shift=1, axis=0)[1:, :])
        return data_log_pct

    def inverse_transform(self, data):
        if self.shape[0] == (data.shape[0] + 1) and self.shape[1] == data.shape[1]:
            if (self.check_row == data[[0], :]).all():
                result = self._inverse_transform(data, self.first_row)
            else:
                first_row = numpy.ones(shape=(1, data.shape[1]))
                result = self._inverse_transform(data, first_row)
        else:
            first_row = numpy.ones(shape=(1, data.shape[1]))
            result = self._inverse_transform(data, first_row)
        return result

    def _inverse_transform(self, data, first_row):
        current_row = first_row
        rows_stack = [current_row]
        for j in numpy.arange(data.shape[0]):
            current_row = numpy.exp((data[j, :] + numpy.log(rows_stack[-1])))
            rows_stack.append(current_row)
        result = numpy.concatenate(rows_stack, axis=1)
        return result


class Whitener:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, data):
        self.scaler.fit(data)

    def transform(self, data):
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class HypeTan:
    def __init__(self):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return numpy.tanh(data)

    def inverse_transform(self, data):
        return numpy.arctanh(data)

