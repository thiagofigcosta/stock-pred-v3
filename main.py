import logger
from data_pipeline import runPipeline
from hyperparameters import Hyperparameters
from nas import ProphetNAS, getDummySearchSpace, getSearchSpace, getMidSearchSpace
from postprocessor import AggregationMethod
from prophet import Prophet


def run_nas():
    from pymoo.config import Config
    Config.warnings['not_compiled'] = False
    import tensorflow as tf
    import logging
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)
    tf.get_logger().setLevel(logging.ERROR)

    ticker = 'goog'
    start_date = '01/01/2018'  # 2002
    end_date = '30/11/2022'
    train_ratio = .7  # .5
    validation_ratio = .3
    fibonacci_seq_size = 20
    use_kernel_pca = True
    agg_method = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    max_evaluations = 100
    population_size = 30
    offspring_size = 30
    parallelism = 4  # 1 means no parallelism, 0 means all cores
    Prophet.PARALLELISM = 1
    eliminate_duplicates = True
    verbose = True

    logger.configure(name='stock-pred-v3-nas', verbose=verbose)

    dataset_configs = Hyperparameters(name='pymoo_nas_dataset', train_ratio=train_ratio,
                                      validation_ratio=validation_ratio,
                                      fibonacci_seq_size=fibonacci_seq_size, use_kernel_pca=use_kernel_pca)
    processed_data, dataset_filepath = runPipeline(ticker, start_date, end_date, dataset_configs, get_path=True,
                                                   encode=False)

    search_space = getMidSearchSpace(dataset_filepath)
    enriched_search_space = Hyperparameters.enrichSearchSpace(search_space)
    nas_opt = ProphetNAS(enriched_search_space, processed_data, population_size, offspring_size, eliminate_duplicates,
                         agg_method)
    nas_opt.optimize(max_evaluations, parallelism=parallelism)
    nas_opt.save()


if __name__ == '__main__':
    run_nas()
