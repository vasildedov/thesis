from datasetsforecast.m3 import M3

df = M3.load(directory='data', group='Yearly')
df1 = M3.load(directory='data', group='Monthly')
df2 = M3.load(directory='data', group='Quarterly')
df3 = M3.load(directory='data', group='Other')

from datasetsforecast.m4 import M4, M4Info
df4 = M4.load(directory='data', group='Yearly')
df5 = M4.load(directory='data', group='Quarterly')
df6_train, hi, df6_test = M4.load(directory='data', group='Monthly')
df7_train, hi, df7_test = M4.load(directory='data', group='Weekly')
df8_train, hi, df8_test = M4.load(directory='data', group='Daily')
df9_train, hi, df9_test = M4.load(directory='data', group='Hourly')

from datasetsforecast.long_horizon import LongHorizon
df = LongHorizon.load(directory='data', group='ETTh2')
