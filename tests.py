#
import numpy
import pandas


#
from m_utils.transformations import Whitener, HypeTan, LogPctTransformer


#
"""
Transformations
"""


#
N = 10_000


# NOK Tests


#
data_nok = pandas.DataFrame(data={'A': numpy.random.normal(size=(N,)),
                                  'B': numpy.array(numpy.arange(N)),
                                  'C': numpy.array(numpy.random.choice([0, 1]), dtype=str)})

#
data_nok = data_nok.values[:, :-1]
data_nok = numpy.array(data_nok, dtype=numpy.float64)

#
"""
trf_nok_w = Whitener()
trf_nok_w.fit(data_nok)
data_nok_w = trf_nok_w.transform(data_nok)
data_nok__w = trf_nok_w.inverse_transform(data_nok_w)

trf_nok_h = HypeTan()
trf_nok_h.fit(data_nok)
data_nok_h = trf_nok_h.transform(data_nok)
data_nok__h = trf_nok_h.inverse_transform(data_nok_h)
"""
trf_nok_p = LogPctTransformer()
trf_nok_p.fit(data_nok)
data_nok_p = trf_nok_p.transform(data_nok)
data_nok__p = trf_nok_p.inverse_transform(data_nok_p)

# OK Tests


#
data_ok = pandas.DataFrame(data={'A': 100 + numpy.random.normal(size=(N,)),
                                 'B': 100 + numpy.random.gamma(shape=1.0, scale=2.0, size=(N,)),
                                 'C': 100 + numpy.random.beta(a=0.5, b=0.5, size=(N,))})
data_ok['D'] = (1 + numpy.random.normal(size=(N,))).cumsum()

data_ok = data_ok.values
data_ok = numpy.array(data_ok, dtype=numpy.float64)

trf_ok_w = Whitener()
trf_ok_w.fit(data_ok)
data_ok_w = trf_ok_w.transform(data_ok)
data_ok__w = trf_ok_w.inverse_transform(data_ok_w)

trf_ok_h = HypeTan()
trf_ok_h.fit(data_ok)
data_ok_h = trf_ok_h.transform(data_ok)
data_ok__h = trf_ok_h.inverse_transform(data_ok_h)

trf_ok_p = LogPctTransformer()
trf_ok_p.fit(data_ok)
data_ok_p = trf_ok_p.transform(data_ok)
data_ok__p = trf_ok_p.inverse_transform(data_ok_p)

trf_ok_w_cc = Whitener()
trf_ok_h_cc = HypeTan()

trf_ok_w_cc.fit(data_ok)
data_ok_cc = trf_ok_w_cc.transform(data_ok)
trf_ok_h_cc.fit(data_ok_cc)
data_ok__cc = trf_ok_h_cc.transform(data_ok_cc)
data_ok___cc = trf_ok_h_cc.inverse_transform(data_ok__cc)
data_ok____cc = trf_ok_w_cc.inverse_transform(data_ok___cc)
