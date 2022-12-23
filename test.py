import numpy as np

from crawler import downloadTicker
from enricher import enrich
from hyperparameters import Hyperparameters
from logger import configure
from plotter import maybeBlockPlots
from postprocessor import decodeWindowedPredictions, AggregationMethod, aggregateDecodedPredictions, \
    decodeWindowedLabels
from preprocessor import preprocess, backwardsRollingWindowInc, forwardRollingWindowInc, forwardRollingWindowExc, \
    backwardsRollingWindowExc, filterBackwardsRollingWindow, filterForwardRollingWindow, filterDoubleRollingWindow
from prophet import Prophet
from transformer import transform
from utils_fs import pathExists

a = list(range(10))
print('aaaaaaaaaaaaaaaaaaaaaaaaa: ', a, f'len: {len(a)}')
print('backwardsRollingWindowInc: ', backwardsRollingWindowInc(a, 3), f'len: {len(backwardsRollingWindowInc(a, 3))}')
print('forwardRollingWindowInc  : ', forwardRollingWindowInc(a, 3), f'len: {len(forwardRollingWindowInc(a, 3))}')
print('backwardsRollingWindowExc: ', backwardsRollingWindowExc(a, 3), f'len: {len(backwardsRollingWindowExc(a, 3))}')
print('forwardRollingWindowExc  : ', forwardRollingWindowExc(a, 3), f'len: {len(forwardRollingWindowExc(a, 3))}')
print()
x = filterBackwardsRollingWindow(backwardsRollingWindowInc(a, 3), 3)
print('backwardsRollingWindowInc-filter: good:', x[0], ' -- bad:', x[1])
x = filterForwardRollingWindow(forwardRollingWindowExc(a, 3), 3)
print('forwardRollingWindowExc-filter  : good:', x[0], ' -- bad:', x[1])
x = filterForwardRollingWindow(forwardRollingWindowInc(a, 3), 3)
print('forwardRollingWindowInc-filter  : good:', x[0], ' -- bad:', x[1])
x = filterBackwardsRollingWindow(backwardsRollingWindowExc(a, 3), 3, True)
print('backwardsRollingWindowExc-filter: good:', x[0], ' -- bad:', x[1])
print()
print()
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 3), 3, forwardRollingWindowExc(a, 3), 3)
print(f'doubleFilter(3,3):\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 3), 3, forwardRollingWindowExc(a, 2), 2)
print(f'doubleFilter(3,2):\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 3), 3, forwardRollingWindowExc(a, 2), 2, True)
print(f'doubleFilter(3,2,True):\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 2), 2, forwardRollingWindowExc(a, 3), 3)
print(
    f'doubleFilter(2,3):\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\tbad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 4), 4, forwardRollingWindowExc(a, 2), 2)
print(f'doubleFilter(4,2)[:6]:\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
a = np.array(list(range(10)))
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a, 3), 3, forwardRollingWindowExc(a, 3), 3)
print(f'np array doubleFilter(3,3):\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
a = list(range(30))
train_idx = 22
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a[:train_idx], 3), 3,
                              forwardRollingWindowExc(a[:train_idx], 2), 2)
print(f'doubleFilter(3,2)[:{train_idx}]:\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t' \
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
x = filterDoubleRollingWindow(backwardsRollingWindowInc(a[train_idx:], 4), 4,
                              forwardRollingWindowExc(a[train_idx:], 2), 2)
print(f'doubleFilter(3,2)[{train_idx}:]:\n\tgood:\n\t\tfea:{x[0][0]}\n\t\tlbl:{x[0][1]}\n\t'
      f'bad:\n\t\tpas:{x[1][0]}\n\t\tfut:{x[1][1]}')
print()
x = decodeWindowedLabels(np.array([[3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]), 3)
print(f'Decoded labels: {x}')
print()
predictions = [[4.1, 5.1], [5.2, 6.2], [6.3, 7.3], [7.4, 8.4], [8.5, 9.5], [9.6, 10.6], [10.7, 11.7]]
x = decodeWindowedPredictions(np.asarray(predictions), 4, 2)
print(f'Predictions: {predictions}\nParsed predictions: {x}')
x = aggregateDecodedPredictions(x, AggregationMethod.FIRST)
print(f'Aggregated predictions: {x}')
print()

configure(name='stock-pred-v3-test', verbose=True)

ticker = 'goog'
start_date = '01/01/2010'
end_date = '01/12/2022'
recreate_dataset = False
force_retrain = False
prophet_to_load_path = 'prophets/lstm_model-0-6d3dfbc24dab5d725134630202b6c6e9eb5fa809dd2287a9878915e324c0d3da.json'
max_epochs = None

dataset_filepath = downloadTicker(ticker, start_date, end_date, force=recreate_dataset)
dataset_filepath = enrich(dataset_filepath, force=recreate_dataset)
dataset_filepath = transform(dataset_filepath, force=recreate_dataset)

Hyperparameters.getDefault().setFilenameAndLoadAmountFeatures(dataset_filepath)
if max_epochs is not None:
    Hyperparameters.getDefault().network.max_epochs = max_epochs
configs_filepath = Hyperparameters.getDefault().saveJson()
loaded_configs = Hyperparameters.loadJson(configs_filepath)

processed_data = preprocess(dataset_filepath)
if force_retrain or not pathExists(prophet_to_load_path):
    prophet = Prophet.build(loaded_configs)
    history = prophet.train(processed_data)
    prophet_to_load_path = prophet.save()
prophet = Prophet.load(prophet_to_load_path)
predictions = prophet.prophesize(processed_data)

maybeBlockPlots()

# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
#  - sobreescrever metricas com callback / metrics melhores
