from typing import Optional, Union

import logger
from data_pipeline import runPipeline
from hyperparameters import Hyperparameters
from nas import ProphetNAS, GAAlgorithm
from postprocessor import AggregationMethod
from prophet import Prophet
from search_space import getSearchSpaceById
from utils_date import getDiffInYearsFromStrDate
from utils_misc import isAppleSilicon


def dataMiningPreset() -> list:
    alg = GAAlgorithm.NSGA2.setObjs(3)
    agg = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    return runNas('goog', '01/01/2018', '30/11/2022', 'MidSearchSpace', 30, 30, 100, ga_alg=alg, agg_method=agg)


def fastPreset() -> list:
    alg = GAAlgorithm.NSGA3.setObjs(5)
    agg = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    parallel = 2
    return runNas('goog', '01/01/2018', '30/11/2022', 'Fast', 4, 4, 8, ga_alg=alg, agg_method=agg,
                  nas_parallelism=parallel, lstm_parallelism=parallel)


def runNas(ticker: str, start_date: str, end_date: str, ss_id: Union[str, int], pop_sz: int, children_sz: int,
           max_eval: int, train_ratio: Optional[bool] = None, validation_ratio: Optional[bool] = None,
           rm_duplicates: bool = True, ga_alg: Optional[GAAlgorithm] = None,
           agg_method: AggregationMethod = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE,
           fib_sz: Optional[int] = 20, k_pca: bool = True, nas_parallelism: Optional[int] = None,
           lstm_parallelism: Optional[int] = None, nas_verbose: bool = True, lstm_verbose: bool = False,
           silence: bool = True) -> list:
    if silence:
        from pymoo.config import Config
        Config.warnings['not_compiled'] = False
        import tensorflow as tf
        import logging
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(3)
        tf.get_logger().setLevel(logging.ERROR)
    logger.configure(name='stock-pred-v3-nas', verbose=nas_verbose or lstm_verbose)
    if train_ratio is None or validation_ratio is None:
        year_diff = getDiffInYearsFromStrDate(end_date, start_date)
        if year_diff < 5:
            if train_ratio is None:
                train_ratio = .7
            if validation_ratio is None:
                train_ratio = .3
        else:
            if train_ratio is None:
                train_ratio = .5
            if validation_ratio is None:
                train_ratio = .2
    if nas_parallelism is None or lstm_parallelism is None:  # 1 means no parallelism, 0 means all cores
        mac_mx = isAppleSilicon()
        if lstm_parallelism is None:
            lstm_parallelism = 1
        if mac_mx:
            max_cores = 5
            if nas_parallelism is None:
                nas_parallelism = max_cores - lstm_parallelism
        else:
            if nas_parallelism is None:
                nas_parallelism = 0

    old_lstm_parallelism = Prophet.PARALLELISM
    Prophet.PARALLELISM = lstm_parallelism
    ProphetNAS.VERBOSE_CALLBACKS = lstm_verbose
    if ga_alg is not None:
        ProphetNAS.ALGORITHM = ga_alg

    dataset_configs = Hyperparameters(name='pymoo_nas_dataset', train_ratio=train_ratio, fibonacci_seq_size=fib_sz,
                                      use_kernel_pca=k_pca, validation_ratio=validation_ratio)
    processed_data, dataset_filepath = runPipeline(ticker, start_date, end_date, dataset_configs, get_path=True,
                                                   encode=False)
    search_space = getSearchSpaceById(ss_id, dataset_filepath, 'pymoo_{gen}_{id}')
    enriched_search_space = Hyperparameters.enrichSearchSpace(search_space)
    nas_opt = ProphetNAS(enriched_search_space, processed_data, pop_sz, children_sz, rm_duplicates, agg_method)
    sol = nas_opt.optimize(max_eval, parallelism=nas_parallelism)
    nas_opt.save()
    Prophet.PARALLELISM = old_lstm_parallelism
    return sol


def main():
    pass


if __name__ == '__main__':
    main()
