#
import numpy
import pandas


def _lag_it(frame, n_lags):
    frame_ = frame.copy()
    if frame_.index.nlevels == 1:
        frame_ = frame_.shift(periods=n_lags, axis=0)
    elif frame_.index.nlevels == 2:
        for ix in frame_.index.levels[0]:
            frame_.loc[[ix], :] = frame_.loc[[ix], :].shift(periods=n_lags, axis=0)
    else:
        raise NotImplemented()
    return frame_


def lag_it(frame, n_lags, exactly=True, keep_basic=True, suffix='_LAG'):
    if exactly:
        if keep_basic:
            new_columns = [x + suffix + '0' for x in frame.columns.values] + [x + suffix + str(n_lags) for x in
                                                                              frame.columns.values]
            frame = pandas.concat((frame, _lag_it(frame=frame, n_lags=n_lags)), axis=1)
            frame.columns = new_columns
        else:
            new_columns = [x + suffix + str(n_lags) for x in frame.columns.values]
            frame = _lag_it(frame=frame, n_lags=n_lags)
            frame.columns = new_columns
    else:
        if keep_basic:
            new_columns = [x + suffix + '0' for x in frame.columns.values]
            frames = [frame]
        else:
            new_columns = []
            frames = []
        for j in numpy.arange(start=1, stop=(n_lags + 1)):
            new_columns = new_columns + [x + suffix + str(j) for x in frame.columns.values]
            frames.append(_lag_it(frame=frame, n_lags=j))
        frame = pandas.concat(frames, axis=1)
        frame.columns = new_columns
    return frame


def _percent_it(frame, n_lags):
    frame_ = frame.copy()
    if frame_.index.nlevels == 1:
        frame_ = frame_.pct_change(periods=n_lags, axis=0, fill_method=None)
    elif frame_.index.nlevels == 2:
        for ix in frame_.index.levels[0]:
            frame_.loc[[ix], :] = frame_.loc[[ix], :].pct_change(periods=n_lags, axis=0, fill_method=None)
    else:
        raise NotImplemented()
    return frame_


def percent_it(frame, horizon, exactly=True):
    if exactly:
        new_columns = [x + '_PCT' + str(horizon) for x in frame.columns.values]
        frame = _percent_it(frame=frame, n_lags=horizon)
        frame.columns = new_columns
    else:
        new_columns = []
        frames = []
        for j in numpy.arange(start=1, stop=(horizon + 1)):
            new_columns = new_columns + [x + '_PCT' + str(j) for x in frame.columns.values]
            frames.append(_percent_it(frame=frame, n_lags=j))
        frame = pandas.concat(frames, axis=1)
        frame.columns = new_columns
    return frame


def _fill_it(frame, date_start, date_end, freq, tz):
    result = pandas.DataFrame(index=pandas.date_range(start=date_start, end=date_end, freq=freq, tz=tz),
                              data=frame)
    return result


def fill_it(frame, freq, zero_index_name, first_index_name):
    data = []
    for ix0 in frame.index.levels[0]:
        filled = _fill_it(frame=frame.loc[ix0, :], date_start=frame.index.levels[1].min(),
                          date_end=frame.index.levels[1].max(), freq=freq, tz=frame.index.levels[1][0].tz)
        filled = filled.reset_index()
        filled[zero_index_name] = ix0
        filled = filled.rename(columns={'index': first_index_name})
        data.append(filled)
    data = pandas.concat(data, axis=0)
    data = data.set_index(keys=[zero_index_name, first_index_name])
    return data
