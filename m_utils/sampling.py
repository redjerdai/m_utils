#
import numpy
import pandas


idx = pandas.IndexSlice


# sampling function
def ts_sampler(data, n_folds, test_rate):
    if data.index.nlevels == 1:
        a = data.index.values

        thresh = int(a.shape[0] * (1 - test_rate))

        data_test = data.loc[a[(thresh - 1)]:, :]

        prt = (thresh - 1) / n_folds
        parts = [(int(j * prt), int((j + 1) * prt)) for j in range(n_folds)]
        parted = [a[pt[0]:pt[1]] for pt in parts]

        folded = [data.loc[fled, :] for fled in parted]

    if data.index.nlevels == 2:
        a = numpy.unique(data.index.levels[1].values)

        thresh = int(a.shape[0] * (1 - test_rate))

        data_test = data.loc[idx[:, a[(thresh - 1)]:], :]

        prt = (thresh - 1) / n_folds
        parts = [(int(j * prt), int((j + 1) * prt)) for j in range(n_folds)]
        parted = [a[pt[0]:pt[1]] for pt in parts]

        folded = [data.loc[idx[:, fled], :] for fled in parted]

    if data.index.nlevels == 3:
        a = numpy.unique(data.index.levels[1].values)

        thresh = int(a.shape[0] * (1 - test_rate))

        data_test = data.loc[idx[:, a[(thresh - 1)]:, :], :]

        prt = (thresh - 1) / n_folds
        parts = [(int(j * prt), int((j + 1) * prt)) for j in range(n_folds)]
        parted = [a[pt[0]:pt[1]] for pt in parts]

        folded = [data.loc[idx[:, fled, :], :] for fled in parted]

    return folded, data_test
