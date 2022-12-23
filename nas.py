import warnings
from enum import Enum, auto
from typing import Optional, Union

import numpy as np
import pathos.pools as pp
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.core.mixed import MixedVariableSampling, MixedVariableMating, MixedVariableDuplicateElimination
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.core.result import Result
from pymoo.core.variable import Integer, Real, Binary, Choice
from pymoo.factory import get_reference_directions
from pymoo.optimize import minimize
from pymoo.visualization.heatmap import Heatmap
from pymoo.visualization.pcp import PCP
from pymoo.visualization.petal import Petal
from pymoo.visualization.radar import Radar
from pymoo.visualization.radviz import Radviz
from pymoo.visualization.scatter import Scatter
from pymoo.visualization.star_coordinate import StarCoordinate

from hyperparameters import Hyperparameters, HYPERPARAMETERS_DIR
from logger import info, warn, exception, error
from plotter import maybeSetFigureManager, showOrSavePymooPlot, getCMap
from postprocessor import AggregationMethod
from preprocessor import ProcessedDataset
from prophet import Prophet
from prophet_enums import Optimizer, ActivationFunc
from search_space import SearchSpace
from utils_date import getNowStr
from utils_fs import createFolder, pathJoin
from utils_misc import getEnumRange, getCpuCount, mergeDicts, getRunId
from utils_persistance import saveJson

EVALUATE_ONE_AT_THE_TIME = False

SSpaceType = SearchSpace.Type
ProblemClass = ElementwiseProblem if EVALUATE_ONE_AT_THE_TIME else Problem

NAS_DIR = 'nas'


def getActivationFuncsList() -> list[ActivationFunc]:
    activations = [
        ActivationFunc.LEAKY_RELU,
        ActivationFunc.SIGMOID,
        ActivationFunc.TANH,
        ActivationFunc.EXPONENTIAL,
        ActivationFunc.HARD_SIGMOID,
        ActivationFunc.SELU,
        ActivationFunc.ELU,
    ]
    return activations


def getRecurrentActivationFuncsList() -> list[ActivationFunc]:
    activations = [
        ActivationFunc.RELU,
        ActivationFunc.SIGMOID,
        ActivationFunc.TANH,
        ActivationFunc.HARD_SIGMOID,
        ActivationFunc.SELU,
        ActivationFunc.ELU,
    ]
    return activations


def getSearchSpace(dataset_filename: Optional[str] = None, name: Optional[str] = None,
                   preprocess_on_nas: bool = False) -> SearchSpace:
    ss = SearchSpace()
    if dataset_filename is not None:
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=dataset_filename)
    if name is not None:
        ss.add(name='name', data_type=SSpaceType.CONSTANT, const=name)
    ss.add(name='backward_samples', data_type=SSpaceType.INT, min_value=5, max_value=60)
    ss.add(name='forward_samples', data_type=SSpaceType.INT, min_value=7, max_value=7)  # TODO 7 - 14
    ss.add(name='max_epochs', data_type=SSpaceType.INT, min_value=500, max_value=5000)
    ss.add(name='stateful', data_type=SSpaceType.BOOLEAN)
    ss.add(name='batch_size', data_type=SSpaceType.INT, min_value=0, max_value=128)
    ss.add(name='dense_instead_lstm_on_out', data_type=SSpaceType.BOOLEAN)
    ss.add(name='patience_epochs_stop', data_type=SSpaceType.INT, min_value=100, max_value=5000)
    ss.add(name='patience_epochs_reduce', data_type=SSpaceType.INT, min_value=0, max_value=1000)
    ss.add(name='reduce_factor', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.2)
    ss.add(name='optimizer', data_type=SSpaceType.INT, **getEnumRange(Optimizer))
    ss.add(name='shuffle', data_type=SSpaceType.BOOLEAN)
    ss.add(name='n_hidden_lstm_layers', data_type=SSpaceType.INT, min_value=0, max_value=3)
    ss.add(name='layer_sizes', data_type=SSpaceType.INT, min_value=10, max_value=80)
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE, choices=getActivationFuncsList())
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE, choices=getRecurrentActivationFuncsList())
    ss.add(name='dropout', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='rec_dropout', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='kernel_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='bias_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='recurrent_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='activity_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='kernel_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='bias_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='recurrent_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='activity_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='use_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='unit_forget_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='go_backwards', data_type=SSpaceType.BOOLEAN)
    # only work if we load files during the nas callback
    if preprocess_on_nas:
        ss.add(name='normalize', data_type=SSpaceType.BOOLEAN)
        ss.add(name='normalize_prediction_feat', data_type=SSpaceType.BOOLEAN)
    return ss


def getDummySearchSpace(dataset_filename: Optional[str] = None, name: Optional[str] = None,
                        preprocess_on_nas: bool = False) -> SearchSpace:
    ss = SearchSpace()
    if dataset_filename is not None:
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=dataset_filename)
    if name is not None:
        ss.add(name='name', data_type=SSpaceType.CONSTANT, const=name)
    ss.add(name='backward_samples', data_type=SSpaceType.INT, min_value=5, max_value=14)
    ss.add(name='forward_samples', data_type=SSpaceType.INT, min_value=7, max_value=7)
    ss.add(name='max_epochs', data_type=SSpaceType.INT, min_value=5, max_value=10)
    ss.add(name='stateful', data_type=SSpaceType.BOOLEAN)
    ss.add(name='batch_size', data_type=SSpaceType.INT, min_value=0, max_value=12)
    ss.add(name='dense_instead_lstm_on_out', data_type=SSpaceType.BOOLEAN)
    ss.add(name='patience_epochs_stop', data_type=SSpaceType.CONSTANT, const=0)
    ss.add(name='patience_epochs_reduce', data_type=SSpaceType.CONSTANT, const=0)
    ss.add(name='reduce_factor', data_type=SSpaceType.CONSTANT, const=0)
    ss.add(name='optimizer', data_type=SSpaceType.INT, **getEnumRange(Optimizer))
    ss.add(name='shuffle', data_type=SSpaceType.CONSTANT, const=True)
    ss.add(name='n_hidden_lstm_layers', data_type=SSpaceType.INT, min_value=0, max_value=1)
    ss.add(name='layer_sizes', data_type=SSpaceType.INT, min_value=5, max_value=10)
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE, choices=getActivationFuncsList())
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE, choices=getRecurrentActivationFuncsList())
    ss.add(name='dropout', data_type=SSpaceType.CONSTANT, const=0)
    ss.add(name='rec_dropout', data_type=SSpaceType.CONSTANT, const=0)
    ss.add(name='use_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='unit_forget_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='go_backwards', data_type=SSpaceType.BOOLEAN)
    # only work if we load files during the nas callback
    if preprocess_on_nas:
        ss.add(name='normalize', data_type=SSpaceType.BOOLEAN)
        ss.add(name='normalize_prediction_feat', data_type=SSpaceType.BOOLEAN)
    return ss


class GAAlgorithm(Enum):
    GA = auto()
    NSGA2 = auto()
    NSGA3 = auto()

    def getObjs(self) -> int:
        return self._n_objs

    def setObjs(self, n_objs: int) -> Enum:
        raise_it = False
        if n_objs < 1:
            raise_it = True
        if n_objs > 1 and self == GAAlgorithm.GA:
            raise_it = True
        if n_objs > 3 and self == GAAlgorithm.NSGA2:
            raise_it = True
        if n_objs > 10 and self == GAAlgorithm.NSGA3:
            raise_it = True
        if raise_it:
            raise ValueError(f'{self} cannot handle {n_objs} objectives')
        self._n_objs = n_objs
        return self

    def __str__(self, short: bool = False) -> str:
        if self == GAAlgorithm.GA:
            if short:
                return "GA"
            else:
                return "Genetic Algorithm"
        elif self == GAAlgorithm.NSGA2:
            if short:
                return "NSGA II"
            else:
                return "Non-dominated Sorting Genetic Algorithm II"
        elif self == GAAlgorithm.NSGA3:
            if short:
                return "NSGA III"
            else:
                return "Non-dominated Sorting Genetic Algorithm III"
        else:
            raise AttributeError('Unknown algorithm')

    @staticmethod
    def getAll() -> list[Enum]:
        return list(map(lambda c: c, GAAlgorithm))


class NotificationCallback(Callback):

    def __init__(self, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose

    def notify(self, algorithm):
        if self.verbose:
            info(f'Finished generation {algorithm.n_gen}, best result so fa: {algorithm.pop.get("F").min()}!')


class ProphetNAS(ProblemClass):
    WORST_VALUE = 2147483647
    BEST_VALUE = -2147483647
    VERBOSE_CALLBACKS = False

    ALGORITHM = GAAlgorithm.NSGA3.setObjs(5)

    def __init__(self, search_space: SearchSpace, processed_data: ProcessedDataset, pop_size: int = 50,
                 children_per_gen: int = 50, eliminate_duplicates: bool = False,
                 agg_method: Union[str, AggregationMethod] = AggregationMethod.VOTING_EXP_F_WEIGHTED_AVERAGE, **kwargs):
        info(f'Instantiating the {ProphetNAS.ALGORITHM}\'s problem...')
        variables = {}
        for dimension in search_space:
            if dimension.data_type == SearchSpace.Type.INT:
                var = Integer(bounds=(dimension.min_value, dimension.max_value))
                variables[dimension.name] = var
            elif dimension.data_type == SearchSpace.Type.FLOAT:
                var = Real(bounds=(dimension.min_value, dimension.max_value))
                variables[dimension.name] = var
            elif dimension.data_type == SearchSpace.Type.BOOLEAN:
                var = Binary()
                variables[dimension.name] = var
            elif dimension.data_type == SearchSpace.Type.CHOICE:
                var = Choice(options=list(dimension.choices))
                variables[dimension.name] = var
            elif dimension.data_type == SearchSpace.Type.CONSTANT:
                pass
            else:
                raise ValueError(f'Unknown SearchSpace.Type')

        alg_kwargs = dict(
            pop_size=pop_size,
            n_offsprings=children_per_gen,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=(MixedVariableDuplicateElimination() if eliminate_duplicates else None)),
            eliminate_duplicates=(MixedVariableDuplicateElimination() if eliminate_duplicates else None),
        )
        ref_dirs = None
        n_obj = ProphetNAS.ALGORITHM.getObjs()
        if ProphetNAS.ALGORITHM == GAAlgorithm.GA:
            alg = GA
            alg_args = []
            alg_kwargs['survival'] = RankAndCrowdingSurvival()
        elif ProphetNAS.ALGORITHM == GAAlgorithm.NSGA2:
            alg = NSGA2
            alg_args = []
        elif ProphetNAS.ALGORITHM == GAAlgorithm.NSGA3:
            alg = NSGA3
            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
            alg_args = [ref_dirs]
        else:
            raise ValueError(f'Unknown {ProphetNAS.ALGORITHM} ')

        agg_method = AggregationMethod.strNoSpace(agg_method)

        self.search_space = search_space
        self.processed_data = processed_data
        self.n_obj = n_obj
        self.parallelism = None
        self.gen = None
        self.history = None
        self.solution = None
        self.ref_dirs = ref_dirs
        self.agg_method = agg_method
        self.pop_size = pop_size
        self.children_per_gen = children_per_gen
        self.eliminate_duplicates = eliminate_duplicates
        self.verbose = False

        self.algorithm = alg(*alg_args, **alg_kwargs)
        super().__init__(vars=variables, n_obj=self.n_obj, **kwargs)
        info(f'Instantiated {ProphetNAS.ALGORITHM} problem!')

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs):
        generation_metrics = [[] for _ in range(self.n_obj)]
        if type(x) is not dict and len(x.shape) == 2 and x.shape[0] == 1:
            x = x[0]
        if self.parallelism == 1 or type(x) is dict:
            for i, individual in enumerate(x):
                metrics = ProphetNAS._trainCallback((i, individual), self.gen, self.search_space, self.processed_data,
                                                    self.agg_method, train_mode=0)  # train_mode=0 -> train and test
                for m, metric in enumerate(metrics):
                    if m < self.n_obj:
                        generation_metrics[m].append(metric)
        else:
            a = [self.gen] * len(x)
            b = [self.search_space] * len(x)
            c = [self.processed_data] * len(x)
            d = [self.agg_method] * len(x)
            e = [1] * len(x)  # train_mode=0 -> train
            with pp.ThreadPool(self.parallelism, maxtasksperchild=None) as pool:
                # cannot plot because matplot is not thread safe
                # outputs = pool.map(ProphetNAS._trainCallback, enumerate(x), a, b, c, d, e)
                success = pool.imap(ProphetNAS._trainCallback, enumerate(x), a, b, c, d, e)  # imap is non blocking
                error = 0
                for ind in success:
                    if not ind:
                        error += 1
                if error > 0:
                    warn(f'Got {error} errors while evaluating...')

            # testing and plotting
            for i, individual in enumerate(x):
                metrics = ProphetNAS._trainCallback((i, individual), self.gen, self.search_space, self.processed_data,
                                                    self.agg_method, train_mode=2)  # train_mode=2 -> test
                for m, metric in enumerate(metrics):
                    if m < self.n_obj:
                        generation_metrics[m].append(metric)

        self.gen += 1
        if self.n_obj == 1:
            if type(x) is dict:
                out["F"] = generation_metrics[0][0]
            else:
                out["F"] = generation_metrics[0]
        else:
            if type(x) is dict:
                out["F"] = np.array(generation_metrics).reshape(-1).tolist()
            else:
                out["F"] = np.column_stack(generation_metrics)

    def optimize(self, max_eval: int = 1000, parallelism: int = 1, verbose: bool = True,
                 store_metrics: bool = True, do_plot: bool = True) -> list:
        info(f'Starting Neural Architecture Search with {ProphetNAS.ALGORITHM}...')
        if parallelism == 0:
            parallelism = getCpuCount()
        elif parallelism < 0:
            parallelism = 1
        if parallelism != 1:
            info(f'Parallelism on NAS: {parallelism}')
            maybeSetFigureManager()
        self.parallelism = parallelism
        self.gen = 0
        self.verbose = verbose
        res = minimize(
            self,
            self.algorithm,
            termination=('n_evals', max_eval),
            save_history=store_metrics,
            callback=NotificationCallback(self.verbose),
            verbose=self.verbose
        )
        # parse history
        history = []
        if store_metrics:
            for gen in res.history:
                individuals = []
                results = []
                for ind in gen.pop:
                    individuals.append(ind.X)
                    results.append(ind.F.tolist())
                gen_vanilla = {'individual': individuals, 'result': results}
                history.append(gen_vanilla)
        self.history = history

        solution = []
        # parse solutions
        for i, ind in enumerate(res.opt):
            solution.append({'individual': ind.X, 'result': ind.F.tolist(),
                             'id': ProphetNAS.getIndId(-i, individual=ind)})  # negative index for solution
        self.solution = solution
        if do_plot:
            self.plot(res)
        info(f'Finished Neural Architecture Search with {ProphetNAS.ALGORITHM}!')
        return solution

    def plot(self, res: Result):
        warnings.filterwarnings("ignore")
        plot_kwargs = dict(tight_layout=True, figsize=(18, 16),
                           cmap=getCMap("tab10"))
        if self.ref_dirs is not None:
            F = self.pareto_front(self.ref_dirs)
        else:
            F = self.pareto_front()
        labels = list(res.opt[0].X.keys())
        X = np.array([[sol.X[name] for name in labels] for sol in res.opt])
        bounds = []
        for name in labels:
            if type(self.vars[name]) is Binary:
                bounds.append((0, 1))
            elif type(self.vars[name]) is Choice:
                bounds.append((0, len(self.vars[name].options) - 1))
            else:
                bounds.append(self.vars[name].bounds)
        bounds = np.array(bounds).T

        title = 'Parallel Coordinate Plot (PCP)'
        label = 'parallel_coordinate_pot'
        plot = PCP(title=title, labels=labels, bounds=bounds, **plot_kwargs)
        plot.set_axis_style(color="grey", alpha=1)
        plot.add(X)
        showOrSavePymooPlot(plot, label)

        if F is not None:
            title = f"{'Pairwise ' if F.shape[1] > 3 else ''}Pareto Front"
            label = f"{'pairwise_' if F.shape[1] > 3 else ''}pareto_front'"
            plot = Scatter(title=title, **plot_kwargs)
            plot.add(F, plot_type="line", color="black", alpha=0.7)
            plot.add(res.F, facecolor="none", edgecolor="red")
            showOrSavePymooPlot(plot, label)

            title = 'Optimization Heatmap'
            label = 'opt_heatmap'
            plot = Heatmap(title=title, y_labels=None, labels=labels, bounds=bounds,
                           **mergeDicts(dict(cmap=getCMap("Oranges_r")), plot_kwargs))
            plot.add(F)
            showOrSavePymooPlot(plot, label)

            title = 'Optimization Petal Plot'
            label = 'petal'
            plot = Petal(title=title, bounds=bounds, labels=labels,
                         **mergeDicts(dict(cmap=getCMap("tab20")), plot_kwargs))
            plot.add(F)
            showOrSavePymooPlot(plot, label)

            title = 'Optimization Radar Plot'
            label = 'radar'
            plot = Radar(title=title, bounds=bounds, labels=labels, **plot_kwargs)
            plot.add(F)
            showOrSavePymooPlot(plot, label)

            title = 'Optimization Radviz Plot'
            label = 'radviz'
            plot = Radviz(title=title, bounds=bounds, labels=labels, **plot_kwargs)
            plot.add(F)
            showOrSavePymooPlot(plot, label)

            title = 'Optimization Star Coordinate Plot'
            label = 'starcoordinate'
            plot = StarCoordinate(title=title, bounds=bounds, labels=labels, **plot_kwargs)
            plot.add(F)
            showOrSavePymooPlot(plot, label)

        warnings.resetwarnings()

    def save(self):
        metadata = {
            'algorithm': str(ProphetNAS.ALGORITHM),
            'n_obj': self.n_obj,
            'parallelism': self.parallelism,
            'gen': self.gen,
            'agg_method': self.agg_method,
            'pop_size': self.pop_size,
            'children_per_gen': self.children_per_gen,
            'eliminate_duplicates': self.eliminate_duplicates,
            'saved_at': getNowStr(),
        }
        createFolder(ProphetNAS.getFilepath(''))
        saveJson(self.search_space.toDict(), ProphetNAS.getFilepath('search_space.json'))
        saveJson(self.history, ProphetNAS.getFilepath('history.json'))
        saveJson(self.solution, ProphetNAS.getFilepath('solution.json'))
        saveJson(metadata, ProphetNAS.getFilepath('metadata.json'))

    @staticmethod
    def getIndId(i, gen: int = None, individual=None) -> str:
        if gen is None:
            gen = individual.data['n_gen']
        return f'gen: {gen} - ind: {i}'

    @staticmethod
    def _trainCallback(i_and_individual: tuple, gen: int, search_space: SearchSpace,
                       processed_data: ProcessedDataset, agg_method: str, train_mode: int) -> Optional[tuple]:
        i, individual = i_and_individual
        mse = None
        f1 = None
        r2 = None
        cs = None
        acc = None
        raise_exceptions = False
        try:
            ind_id = ProphetNAS.getIndId(i, gen=gen)
            hyperparameters = Hyperparameters.parseDna(individual, search_space)
            if hyperparameters.name is not None:
                hyperparameters.name += f'-gen={gen}-id={ind_id}'
                hyperparameters.refreshUuids()

            if train_mode <= 1:  # train and test or just train
                try:
                    prophet = Prophet.build(hyperparameters, basename=hyperparameters.name,
                                            do_log=ProphetNAS.VERBOSE_CALLBACKS,
                                            do_verbose=ProphetNAS.VERBOSE_CALLBACKS,
                                            ignore_save_error=True, path_subdir=f'{getRunId()}')
                except ValueError:
                    s_dir = 'ERROR'
                    createFolder(pathJoin(HYPERPARAMETERS_DIR, s_dir))
                    filep = hyperparameters.saveJson(subdir=s_dir)
                    error(f'Error building LSTM network, hyperparameters saved at {filep}')
                    raise Exception()

                prophet.train(processed_data.encode(hyperparameters, copy=True), do_log=ProphetNAS.VERBOSE_CALLBACKS)
                if train_mode == 1:  # just train
                    prophet.save(ignore_error=True, do_log=ProphetNAS.VERBOSE_CALLBACKS)
                    Prophet.destroy(prophet)
                    return
            else:
                prophet_path = Prophet.genProphetBasename(hyperparameters, basename=hyperparameters.name)
                prophet = Prophet.load(prophet_path, do_log=ProphetNAS.VERBOSE_CALLBACKS,
                                       do_verbose=ProphetNAS.VERBOSE_CALLBACKS, path_subdir=f'{getRunId()}')
            predictions = prophet.prophesize(processed_data.encode(hyperparameters, copy=True),
                                             do_log=ProphetNAS.VERBOSE_CALLBACKS)
            prophet.save(ignore_error=True, do_log=ProphetNAS.VERBOSE_CALLBACKS)
            manual_metrics = predictions.get('manual_metrics', {})
            if 'validation' in manual_metrics:
                metrics = manual_metrics.get('validation', {})
            else:
                metrics = manual_metrics.get('train', {})
            metrics = metrics.get(agg_method, {})
            regression = metrics.get('regression', {})
            binary = metrics.get('binary', {})
            mse = regression.get('mse', None)
            f1 = binary.get('f1', None)
            r2 = regression.get('r2', None)
            cs = regression.get('cos_sim', None)
            acc = binary.get('acc', None)
            Prophet.destroy(prophet)
        except Exception as e:
            exception(e, raise_it=raise_exceptions)
        mse = ProphetNAS.parseMetric(mse, True, 'MSE')
        f1 = ProphetNAS.parseMetric(f1, False, 'F1 Score')
        r2 = ProphetNAS.parseMetric(r2, False, 'R²')
        cs = ProphetNAS.parseMetric(cs, False, 'Cosine Similarity')
        acc = ProphetNAS.parseMetric(acc, False, 'Accuracy')

        return mse, f1, r2, cs, acc

    @staticmethod
    def parseMetric(value: Union[int, float, np.float32], minimization_metric: bool,
                    name: Optional[str] = None) -> Union[int, float, np.float32]:
        if value is None or value != value:
            value = ProphetNAS.WORST_VALUE
            warn(f'Error on {name} metric ({value})!')
        if not minimization_metric:
            value *= -1  # since this is a minimization problem
        return value

    @staticmethod
    def getFilepath(basename: str) -> str:
        return pathJoin(NAS_DIR, f'run_id-{getRunId():06d}', basename)


createFolder(NAS_DIR)
