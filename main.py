import argparse
import sys
from typing import Optional, Union

import logger
from data_pipeline import runPipeline
from hyperparameters import Hyperparameters
from logger import info, logDict, fatal
from nas import ProphetNAS, GAAlgorithm
from postprocessor import AggregationMethod
from prophet import Prophet
from search_space import getSearchSpaceById
from utils_date import getDiffInYearsFromStrDate
from utils_misc import isAppleSilicon


def dataMiningPreset() -> list:
    alg = GAAlgorithm.NSGA2.setObjs(3)
    agg = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    return runNas('goog', '01/01/2018', '30/11/2022', 'MidSearchSpace', 100, 30, ga_alg=alg, agg_method=agg)


def fastPreset(respect_nsga3: bool = True) -> list:
    alg = GAAlgorithm.NSGA3.setObjs(5)
    ref_dir_configs = (-1, 1)
    pop_sz = 126 if respect_nsga3 else 8
    max_gens = 10
    agg = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    nas_parallelism, lstm_parallelism = -2, 2
    return runNas('goog', '01/01/2018', '30/11/2022', 'FastOne', pop_sz * max_gens, pop_sz, ga_alg=alg, agg_method=agg,
                  nas_parallelism=nas_parallelism, lstm_parallelism=lstm_parallelism, ref_dir_configs=ref_dir_configs)


def bigPreset(respect_nsga3: bool = True) -> list:
    alg = GAAlgorithm.NSGA3.setObjs(5)
    ref_dir_configs = (-1, 1)
    pop_sz = 126 if respect_nsga3 else 8
    max_gens = 20
    agg = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE
    nas_parallelism, lstm_parallelism = -2, 2
    return runNas('goog', '01/01/2018', '30/11/2022', 'SearchSpace', pop_sz * max_gens, pop_sz, ga_alg=alg,
                  agg_method=agg, nas_parallelism=nas_parallelism, lstm_parallelism=lstm_parallelism,
                  ref_dir_configs=ref_dir_configs)


def runNas(ticker: str, start_date: str, end_date: str, ss_id: Union[str, int], max_eval: int, pop_sz: int,
           children_sz: Optional[int] = None, train_ratio: Optional[bool] = None,
           validation_ratio: Optional[bool] = None, rm_duplicates: bool = True, ga_alg: Optional[GAAlgorithm] = None,
           ref_dir_configs: Optional[tuple[float, float]] = None,
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
    level = logger.Level.VERBOSE if nas_verbose or lstm_verbose else logger.Level.INFO
    logger.configure(name='stock-pred-v3-nas', level=level)
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
    nas_opt = ProphetNAS(enriched_search_space, processed_data, pop_sz, children_sz, rm_duplicates, agg_method,
                         ref_dir_configs)
    sol = nas_opt.optimize(max_eval, parallelism=nas_parallelism)
    nas_opt.save()
    Prophet.PARALLELISM = old_lstm_parallelism
    return sol


def getArgumentParser(prog: str) -> argparse.ArgumentParser:
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter # includes the default by default
    parser = argparse.ArgumentParser(prog, description='This is stock-pred-v3, the prophet of the stock market')

    # modes
    subparsers = parser.add_subparsers(dest='mode', help='The operating mode')
    nas_parser = subparsers.add_parser('nas', help='Makes a neural architecture search!')
    lstm_parser = subparsers.add_parser('lstm', help='Runs over a specific set of hyperparameters')

    # arguments for all modes
    for mode, sub_p in subparsers.choices.items():
        sub_p.add_argument('--stock', required=True, type=str, nargs='?',
                           help='Provide the stock tickers to run, e.g. GOOG', metavar='stocks')
        sub_p.add_argument('--start', required=True, type=str, help='The start date to crawl the data from',
                           metavar='start')
        sub_p.add_argument('--end', required=True, type=str, help='The end date to crawl the data till', metavar='end')

        sub_p.add_argument('--dry_run', action='store_true', default=False,
                           help='Shows the arguments provided without debating or running, :)')
        sub_p.add_argument('--agg_method', type=str, default='VOTING_EXP_F_WEIGHTED_AVERAGE',
                           help='The method of uniting redundant previsions', metavar='agg')
        sub_p.add_argument('--lstm_p', type=int, help='Parallelism for LSTM Networks', metavar='lstm_cores')

    # nas mode args
    nas_parser.add_argument('--ss_id', required=True, type=str, help='The search space for the NAS',
                            metavar='search_space')
    nas_parser.add_argument('--pop_sz', required=True, type=int, help='The populations size for th NAS',
                            metavar='pop_size')
    nas_parser.add_argument('--children_sz', type=int, help='The amount of offspring for the NAS',
                            metavar='children_sz')
    nas_parser.add_argument('--max_eval', required=True, type=int, help='The maximum amount of evaluations for the NAS',
                            metavar='max_eval')
    nas_parser.add_argument('--nas_alg', required=True, type=str, help='The NAS algorithm', metavar='nas_alg')
    nas_parser.add_argument('--nas_obj', required=True, type=int, help='The amount of objectives for the NAS',
                            metavar='nas_obj')
    nas_parser.add_argument('--nas_p', type=int, help='Parallelism for NAS', metavar='nas_cores')
    nas_parser.add_argument('--nas_ref_dir_conf', type=str, help='Ref dir parameters', metavar='nas_ref_dir_conf')
    # lstm mode args
    return parser, subparsers


def main(argv):
    parser, subparsers = getArgumentParser(argv.pop(0))
    args = parser.parse_args(argv)
    if args.mode is None:
        parser.print_help(sys.stdout)
    else:
        mode = args.mode.lower()
        if mode not in subparsers.choices.keys():
            fatal(f'Invalid mode: {mode}')

    if args.dry_run:
        info('Received the following args')
        logDict(args.__dict__, 'Arguments')
        return

    if mode == 'nas':
        stocks = args.stock.split(',')
        ga_alg = None
        ref_dir_configs = None
        if 'ns' in args.nas_alg.lower():
            if '3' in args.nas_alg or 'iii' in args.nas_alg.lower():
                ga_alg = GAAlgorithm.NSGA3.setObjs(args.nas_obj)
                if args.nas_ref_dir_conf is not None:
                    ref_params = args.nas_ref_dir_conf.replace('[', '').replace(']', ''). \
                        replace('(', '').replace(')', '').split(',')
                    ref_dir_configs = (float(ref_params[0]), float(ref_params[1]))
            elif '2' in args.nas_alg or 'ii' in args.nas_alg.lower():
                ga_alg = GAAlgorithm.NSGA2.setObjs(args.nas_obj)
        elif 'ga' in args.nas_alg.lower():
            ga_alg = GAAlgorithm.GA.setObjs(args.nas_obj)
        if ga_alg is None:
            raise ValueError(f'Unknown algorithm {args.nas_alg}')
        agg_method = None
        agg_name = args.agg_method.replace(' ', '_').lower()
        for met in AggregationMethod.getAll():
            met_name = met.toStrNoSpace().lower()
            if met_name == agg_name:
                agg_method = met
        if agg_method is None:
            raise ValueError(f'Unknown agg_method {args.nas_alg}')
        for stock in stocks:
            runNas(stock, args.start, args.end, args.ss_id, args.max_eval, pop_sz=args.pop_sz,
                   children_sz=args.children_sz,
                   ga_alg=ga_alg, ref_dir_configs=ref_dir_configs, agg_method=agg_method, nas_parallelism=args.nas_p,
                   lstm_parallelism=args.lstm_p)
    elif mode == 'lstm':
        raise NotImplementedError()
    else:
        fatal(f'Invalid mode: {mode}')


if __name__ == '__main__':
    try:
        main(sys.argv)
    except Exception as e:
        logger.exception(e, raise_it=True)
