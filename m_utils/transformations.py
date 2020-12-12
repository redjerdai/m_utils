#
import numpy
import pandas
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
        self.full_set = None
        # self.check_row = None
        self.last_row = None
        self.shape = None
        pass

    def fit(self, data):
        self.full_set = data.copy()
        # self.check_row = numpy.log(data[[1], :]) - numpy.log(data[[0], :])
        self.last_row = data[[data.shape[0] - 1], :]
        self.shape = data.shape
        pass

    def transform(self, data):
        data_log_pct = numpy.log(data) - numpy.log(numpy.roll(data, shift=1, axis=0))
        data_log_pct[0, :] = numpy.nan
        return data_log_pct

    """
    def inverse_transform(self, data):
        if self.shape[0] == data.shape[0] and self.shape[1] == data.shape[1]:
            
            if pandas.isna(data).all(axis=1)[0]:
                # suppose it is train
                result = self._inverse_transform(data, self.full_set)
            else:
                # suppose it is test
                result = self._inverse_transform(data, self.last_row)
        else:
            # suppose it is test
            result = self._inverse_transform(data, self.last_row)
        return result

    def _inverse_transform(self, data, first_row):
        current_row = first_row
        rows_stack = [current_row]
        for j in numpy.arange(data.shape[0]):
            if j != 0:
                current_row = numpy.exp((data[j, :] + numpy.log(rows_stack[-1])))
                rows_stack.append(current_row)
        result = numpy.concatenate(rows_stack, axis=0)
        return result
    """

    def inverse_transform(self, data):

        if self.shape[0] == data.shape[0]:

            rows_stack = []
            for j in range(data.shape[0]):
                if pandas.isna(data[j, :]).any():
                    rows_stack.append(numpy.array([numpy.nan] * data.shape[1]).reshape(1, -1))
                else:
                    current_row = self.full_set[j - 1, :] * numpy.exp(data[j, :]).reshape(1, -1)
                    rows_stack.append(current_row)

            result = numpy.concatenate(rows_stack, axis=0)
            return result

        else:

            current_row = self.full_set[-1, :]
            rows_stack = [current_row]
            for j in numpy.arange(data.shape[0]):
                if j != 0:
                    current_row = numpy.exp((data[j, :] + numpy.log(rows_stack[-1])))
                    rows_stack.append(current_row)
            result = numpy.concatenate(rows_stack, axis=0)
            return result


class __LogPctTransformer:
    def __init__(self):
        self.first_row = None
        # self.check_row = None
        self.last_row = None
        self.shape = None
        pass

    def fit(self, data):
        self.first_row = data[[0], :]
        # self.check_row = numpy.log(data[[1], :]) - numpy.log(data[[0], :])
        self.last_row = data[[data.shape[0] - 1], :]
        self.shape = data.shape
        pass

    def transform(self, data):
        data_log_pct = numpy.log(data) - numpy.log(numpy.roll(data, shift=1, axis=0))
        data_log_pct[0, :] = numpy.nan
        return data_log_pct

    def inverse_transform(self, data):
        if self.shape[0] == data.shape[0] and self.shape[1] == data.shape[1]:
            """
            if (self.check_row == data[[1], :]).all():
                result = self._inverse_transform(data, self.first_row)
            else:
                first_row = numpy.ones(shape=(1, data.shape[1]))
                result = self._inverse_transform(data, first_row)
            """

            if pandas.isna(data).all(axis=1)[0]:
                # suppose it is train
                result = self._inverse_transform(data, self.first_row)
            else:
                # suppose it is test
                result = self._inverse_transform(data, self.last_row)
        else:
            """
            first_row = numpy.ones(shape=(1, data.shape[1]))
            result = self._inverse_transform(data, first_row)
            """
            # suppose it is test
            result = self._inverse_transform(data, self.last_row)
        return result

    def _inverse_transform(self, data, first_row):
        current_row = first_row
        rows_stack = [current_row]
        for j in numpy.arange(data.shape[0]):
            if j != 0:
                current_row = numpy.exp((data[j, :] + numpy.log(rows_stack[-1])))
                rows_stack.append(current_row)
        result = numpy.concatenate(rows_stack, axis=0)
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
    # Note
    # for big numbers 'transform' yields 1.0 which could not be 'inverse_transform'_ed because of lack of precision

    def __init__(self):
        pass

    def fit(self, data):
        pass

    def transform(self, data):
        return numpy.tanh(data)

    def inverse_transform(self, data):
        return numpy.arctanh(data)


# RENAME: transformers --> transformators !! there shall be no term confusions
class TransformStack:

    def __init__(self, masked_listed, coded):
        self.masked_listed = masked_listed
        self.coded = coded
        """"""
        self.transformers = list(self.masked.keys())
        self.masks = [self.masked[key] for key in self.transformers]
        self.n = len(self.transformers)

    def say_my_name(self):

        return self.coded

    def fit(self, array):

        for j in range(self.n):
            self.transformers[j].fit(array[:, self.masks[j]])

    def forward(self, array):

        array_ = array.copy()

        for j in range(self.n):
            # array_ = self.transformers[j].transform(array_[:, self.masks[j]])
            try:
                tmp = self.transformers[j].forward(array_[:, self.masks[j]])
                array_[:, self.masks[j]] = tmp
            except Exception as e:
                print(tmp.shape)
                print(array_[:, self.masks[j]].shape)
                raise e

        return array_

    def backward(self, array):

        array_ = array.copy()

        for j in range(self.n):
            # array_ = self.transformers[-j - 1].inverse_transform(array_[:, self.masks[-j - 1]])
            array_ = self.transformers[-j - 1].backward(array_[:, self.masks[-j - 1]])

        return array_
