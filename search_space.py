from enum import Enum, auto
from typing import Union, Optional

from logger import warn
from prophet_enums import Optimizer, ActivationFunc
from utils_fs import getBasename
from utils_misc import anythingToBool, getNumericTypes, getEnumRange
from utils_random import randChoice, randInt

_ALLOWED_TYPES = Union[int, float, bool, str, Enum]


class SearchSpace(object):
    class Dimension(object):
        pass  # Just to hint its return type in the true class


class SearchSpace(object):
    LOOSE_SPACE = True

    class Type(Enum):
        INT = auto()
        FLOAT = auto()
        BOOLEAN = auto()
        CHOICE = auto()
        CONSTANT = auto()

        def __str__(self) -> str:
            return self.name

    class Dimension(object):
        def __init__(self, data_type: Enum,
                     min_value: Optional[_ALLOWED_TYPES] = None, max_value: Optional[_ALLOWED_TYPES] = None,
                     choices: Optional[Union[list[_ALLOWED_TYPES], set[_ALLOWED_TYPES]]] = None,
                     const: Optional[_ALLOWED_TYPES] = None, name: Optional[str] = ''):
            has_range_val = min_value is not None and max_value is not None
            has_choices = choices is not None and len(choices) > 0
            has_const = const is not None
            if has_range_val:
                if isinstance(min_value, Enum):
                    min_value = min_value.value
                if isinstance(max_value, Enum):
                    max_value = max_value.value
                if type(min_value) is not bool and (min_value > max_value):
                    raise AttributeError('Incorrect limits, swap min and max')
                if data_type not in (SearchSpace.Type.INT, SearchSpace.Type.FLOAT, SearchSpace.Type.BOOLEAN):
                    raise AttributeError(f'{data_type} does not support ranges')
            if has_choices:
                choices = list(choices)
                for i, choice in enumerate(choices):
                    if isinstance(choice, Enum):
                        choices[i] = choice.value
                    elif type(choice) not in (str, int, float, bool):
                        raise ValueError(f'Invalid type for choice {type(choice)}')
                if data_type != SearchSpace.Type.CHOICE:
                    raise AttributeError(f'{data_type} does not support ranges')
            if has_const:
                if isinstance(const, Enum):
                    const = const.value
                if data_type != SearchSpace.Type.CONSTANT:
                    raise AttributeError(f'{data_type} does not support const')
            if not has_range_val and not has_choices and not has_const:
                if data_type == SearchSpace.Type.BOOLEAN:
                    min_value = False
                    max_value = True
                else:
                    raise AttributeError('No values were set')

            name = name if name is not None and name != "" else f"Var-{id(self)}"
            self.name = name
            self.data_type = data_type
            self.min_value = min_value
            self.max_value = max_value
            self.choices = choices
            self.const = const
            self.fixConstraints()

        def fixConstraints(self) -> None:
            if self.data_type == SearchSpace.Type.INT:
                self.min_value = int(self.min_value)
                self.max_value = int(self.max_value)
            elif self.data_type == SearchSpace.Type.FLOAT:
                self.min_value = float(self.min_value)
                self.max_value = float(self.max_value)
            elif self.data_type == SearchSpace.Type.BOOLEAN:
                self.min_value = anythingToBool(self.min_value)
                self.max_value = anythingToBool(self.max_value)
                if self.min_value == self.max_value:
                    self.const = self.min_value
                    self.min_value = self.max_value = 0
                    self.data_type = SearchSpace.Type.CONSTANT
            elif self.data_type == SearchSpace.Type.CHOICE:
                self.choices = set(self.choices)
                for choice in self.choices:
                    if type(choice) is not str and type(choice) not in getNumericTypes():
                        raise ValueError(f'Unsupported typed for choice: {type(choice)}')
            elif self.data_type == SearchSpace.Type.CONSTANT:
                pass
            else:
                raise ValueError(f'Unknown SearchSpace.Type: {self.data_type}')

        def fixValue(self, value: _ALLOWED_TYPES) -> _ALLOWED_TYPES:
            if isinstance(value, Enum):
                value = value.value

            if self.data_type == SearchSpace.Type.INT:
                value = int(value)
                if value < self.min_value:
                    value = self.min_value
                elif value > self.max_value:
                    value = self.max_value
            elif self.data_type == SearchSpace.Type.FLOAT:
                value = float(value)
                if value < self.min_value:
                    value = self.min_value
                elif value > self.max_value:
                    value = self.max_value
            elif self.data_type == SearchSpace.Type.BOOLEAN:
                value = anythingToBool(value)
                min_max_set = {self.min_value, self.max_value}
                if value not in min_max_set:
                    if not SearchSpace.LOOSE_SPACE:
                        raise ValueError(f'Value `{value}` should be in {min_max_set}!')
                    old_val = value
                    value = randChoice(min_max_set)
                    warn(f'Value: {old_val} not in {min_max_set}, assigned new value: {value}')
            elif self.data_type == SearchSpace.Type.CHOICE:
                if value not in self.choices:
                    if not SearchSpace.LOOSE_SPACE:
                        raise ValueError(f'Value `{value}` should be in {self.choices}!')
                    old_val = value
                    value = randChoice(self.choices)
                    warn(f'Value: {old_val} not in {self.choices}, assigned new value: {value}')
            elif self.data_type == SearchSpace.Type.CONSTANT:
                return self.const
            else:
                raise ValueError(f'Unknown SearchSpace.Type: {self.data_type}')
            return value

        def copy(self) -> SearchSpace.Dimension:
            that = SearchSpace.Dimension(data_type=self.data_type, min_value=self.min_value, max_value=self.max_value,
                                         choices=self.choices, const=self.const, name=self.name)
            return that

        def __str__(self) -> str:
            out = f'{self.name}: '
            if self.data_type == SearchSpace.Type.CHOICE:
                out += f'[ {self.data_type} | choices: {self.choices} ]'
            elif self.data_type == SearchSpace.Type.CONSTANT:
                out += f'[ {self.data_type} | const: {self.const} ]'
            elif self.data_type == SearchSpace.Type.BOOLEAN:
                out += f'[ {self.data_type} ]'
            else:
                out += f'[ {self.data_type} | min: {self.min_value} max: {self.max_value} ]'
            return out

        def toDict(self) -> dict:
            out = {
                '__type__': 'Dimension',
                'data_type': f'{SearchSpace.Type.CHOICE}',
            }
            if self.data_type == SearchSpace.Type.CHOICE:
                out['choices'] = f'{self.choices}'
            elif self.data_type == SearchSpace.Type.CONSTANT:
                out['const'] = f'{self.const}'
            elif self.data_type == SearchSpace.Type.BOOLEAN:
                pass
            else:
                out['min'] = f'{self.min_value}'
                out['max'] = f'{self.max_value}'
            return out

    def __init__(self, name: Optional[str] = None, genome_map: Optional[dict] = None):
        self._search_space = []
        self._search_space_map = {}
        self.name = name
        if genome_map is None:
            genome_map = {}
        self.genome_map = genome_map
        self.const_mask = None

    def __len__(self) -> int:
        return len(self._search_space)

    def __getitem__(self, i: Union[int, str]) -> Optional[Dimension]:
        if type(i) is int:
            that: SearchSpace.Dimension = self._search_space[i].copy()
            return that
        elif type(i) is str:
            if len(self._search_space) == len(self._search_space_map):
                that: SearchSpace.Dimension = self._search_space[self._search_space_map[i]].copy()
                return that
            else:
                for el in self._search_space:
                    if el.name == i:
                        return el.copy()
        return None

    def __iter__(self) -> object:
        return SearchSpaceIterator(self)

    def __str__(self) -> str:
        str_out = 'Search Space: { \n'
        for dim in self._search_space:
            str_out += f'\t{dim},\n'
        str_out += '}'
        return str_out

    def __contains__(self, i: Union[int, str]) -> bool:
        if type(i) is int:
            return 0 <= i < len(self._search_space)
        elif type(i) is str:
            return i in self._search_space_map
        return False

    def add(self, data_type: Optional[Enum] = None, min_value: Optional[_ALLOWED_TYPES] = None,
            max_value: Optional[_ALLOWED_TYPES] = None,
            choices: Optional[Union[list[_ALLOWED_TYPES], set[_ALLOWED_TYPES]]] = None,
            const: Optional[_ALLOWED_TYPES] = None,
            name: Optional[str] = '', dimension: Optional[Dimension] = None) -> int:
        if dimension is not None:
            return self.add(data_type=dimension.data_type, min_value=dimension.min_value, max_value=dimension.max_value,
                            choices=dimension.choices, const=dimension.const, name=dimension.name)
        else:
            idx = len(self._search_space)
            if name is None or name == '':
                name = f'Dimension-{idx}'
            attempts = 0
            while name in self._search_space_map:
                if attempts == 0:
                    name += f'-'
                name += f'{randInt(9)}'
                attempts += 1
                if attempts > 10:
                    break
            self._search_space.append(SearchSpace.Dimension(data_type=data_type, min_value=min_value, const=const,
                                                            max_value=max_value, name=name, choices=choices))
            self._search_space_map[name] = idx
            return idx

    def get(self, i: Union[int, str]) -> Optional[Dimension]:
        return self._search_space[i]

    def getDimensionNames(self) -> set[str]:
        return set(self._search_space_map.keys())

    def copy(self) -> SearchSpace:
        that = SearchSpace(name=self.name, genome_map=self.genome_map.copy())
        for dimension in self._search_space:
            that._search_space.append(dimension.copy())
        return that

    def getConstantsMask(self) -> list[int]:
        mask = []
        mask_val = 0
        for i, dim in enumerate(self._search_space):
            if dim.data_type == SearchSpace.Type.CONSTANT:
                mask_val += 1
            mask.append(mask_val)
        self.const_mask = mask
        return mask

    def applyConstantsMask(self, i: int, mask: Optional[list[int]] = None, reverse: bool = False) -> int:
        if mask is None:
            mask = self.const_mask
        if mask is None:
            mask = self.getConstantsMask()
        mask_val = mask[i] if not reverse else -mask[i]
        return i - mask_val

    def toDict(self) -> dict:
        out = {
            '__type__': 'SearchSpace',
            'name': self.name,
            'dimensions': [],
        }
        for dim in self._search_space:
            out['dimensions'].append(dim.toDict())
        return out


class SearchSpaceIterator:
    def __init__(self, search_space: SearchSpace):
        self._search_space = search_space
        self._index = 0

    def __next__(self):
        if self._index < len(self._search_space):
            result = self._search_space.get(self._index)
            self._index += 1
            return result
        raise StopIteration


SSpaceType = SearchSpace.Type


def getActivationFuncsList(ss_id: int) -> list[ActivationFunc]:
    all_you_can_have = [
        ActivationFunc.LEAKY_RELU,
        ActivationFunc.SIGMOID,
        ActivationFunc.TANH,
        ActivationFunc.EXPONENTIAL,
        ActivationFunc.HARD_SIGMOID,
        ActivationFunc.SELU,
        ActivationFunc.ELU,
    ]
    if ss_id == getSearchSpaceIdByName('searchspace'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('dummy'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('mid'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('fast'):
        out = [
            ActivationFunc.LEAKY_RELU,
            ActivationFunc.SIGMOID,
            ActivationFunc.TANH,
            ActivationFunc.HARD_SIGMOID,
        ]
    else:
        raise ValueError(f'Invalid SS id {ss_id}')
    for el in out:
        if el not in all_you_can_have:
            raise ValueError(f'Invalid recurrent activation id {el}')
    return out


def getRecurrentActivationFuncsList(ss_id: int) -> list[ActivationFunc]:
    all_you_can_have = [
        ActivationFunc.RELU,
        ActivationFunc.SIGMOID,
        ActivationFunc.TANH,
        ActivationFunc.HARD_SIGMOID,
        ActivationFunc.SELU,
        ActivationFunc.ELU,
    ]
    if ss_id == getSearchSpaceIdByName('searchspace'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('dummy'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('mid'):
        out = all_you_can_have
    elif ss_id == getSearchSpaceIdByName('fast'):
        out = [
            ActivationFunc.RELU,
            ActivationFunc.SIGMOID,
            ActivationFunc.TANH,
            ActivationFunc.HARD_SIGMOID,
        ]
    else:
        raise ValueError(f'Invalid SS id {ss_id}')
    for el in out:
        if el not in all_you_can_have:
            raise ValueError(f'Invalid recurrent activation id {el}')
    return out


_SEARCH_SPACES_MAP = {
    'SearchSpace': 0,
    'SearchSpace-0': 0,
    #
    'Dummy': 1,
    'DummySearchSpace': 1,
    'SearchSpace-1': 1,
    #
    'Mid': 2,
    'MidSearchSpace': 2,
    'SearchSpace-2': 2,
    #
    'FastOne': 3,
    'Fast': 3,
    'SearchSpace-3': 3,
}
_SEARCH_SPACES_MAP = {k.lower(): v for k, v in _SEARCH_SPACES_MAP.items()}


def getSearchSpace(dataset_filename: Optional[str] = None, name: Optional[str] = None,
                   preprocess_on_nas: bool = False) -> SearchSpace:
    ss = SearchSpace()
    if dataset_filename is not None:
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=getBasename(dataset_filename))
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
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getActivationFuncsList(getSearchSpaceIdByName('searchspace')))
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getRecurrentActivationFuncsList(getSearchSpaceIdByName('searchspace')))
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


def getMidSearchSpace(dataset_filename: Optional[str] = None, name: Optional[str] = None,
                      preprocess_on_nas: bool = False) -> SearchSpace:
    ss = SearchSpace()
    if dataset_filename is not None:
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=getBasename(dataset_filename))
    if name is not None:
        ss.add(name='name', data_type=SSpaceType.CONSTANT, const=name)
    ss.add(name='backward_samples', data_type=SSpaceType.INT, min_value=5, max_value=60)
    ss.add(name='forward_samples', data_type=SSpaceType.INT, min_value=7, max_value=7)  # TODO 7 - 14
    ss.add(name='max_epochs', data_type=SSpaceType.INT, min_value=200, max_value=2000)
    ss.add(name='stateful', data_type=SSpaceType.BOOLEAN)
    ss.add(name='batch_size', data_type=SSpaceType.INT, min_value=0, max_value=128)
    ss.add(name='dense_instead_lstm_on_out', data_type=SSpaceType.BOOLEAN)
    ss.add(name='patience_epochs_stop', data_type=SSpaceType.INT, min_value=50, max_value=2000)
    ss.add(name='patience_epochs_reduce', data_type=SSpaceType.INT, min_value=0, max_value=500)
    ss.add(name='reduce_factor', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.2)
    ss.add(name='optimizer', data_type=SSpaceType.INT, **getEnumRange(Optimizer))
    ss.add(name='shuffle', data_type=SSpaceType.BOOLEAN)
    ss.add(name='n_hidden_lstm_layers', data_type=SSpaceType.INT, min_value=0, max_value=2)
    ss.add(name='layer_sizes', data_type=SSpaceType.INT, min_value=5, max_value=33)
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getActivationFuncsList(getSearchSpaceIdByName('mid')))
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getRecurrentActivationFuncsList(getSearchSpaceIdByName('mid')))
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
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=getBasename(dataset_filename))
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
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getActivationFuncsList(getSearchSpaceIdByName('dummy')))
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getRecurrentActivationFuncsList(getSearchSpaceIdByName('dummy')))
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


def getFastSearchSpace(dataset_filename: Optional[str] = None, name: Optional[str] = None,
                       preprocess_on_nas: bool = False) -> SearchSpace:
    ss = SearchSpace()
    if dataset_filename is not None:
        ss.add(name='dataset_filename', data_type=SSpaceType.CONSTANT, const=getBasename(dataset_filename))
    if name is not None:
        ss.add(name='name', data_type=SSpaceType.CONSTANT, const=name)
    ss.add(name='backward_samples', data_type=SSpaceType.INT, min_value=5, max_value=60)
    ss.add(name='forward_samples', data_type=SSpaceType.INT, min_value=7, max_value=7)
    ss.add(name='max_epochs', data_type=SSpaceType.INT, min_value=100, max_value=1200)
    ss.add(name='stateful', data_type=SSpaceType.BOOLEAN)
    ss.add(name='batch_size', data_type=SSpaceType.INT, min_value=0, max_value=128)
    ss.add(name='dense_instead_lstm_on_out', data_type=SSpaceType.BOOLEAN)
    ss.add(name='patience_epochs_stop', data_type=SSpaceType.INT, min_value=50, max_value=180)
    ss.add(name='patience_epochs_reduce', data_type=SSpaceType.INT, min_value=0, max_value=90)
    ss.add(name='reduce_factor', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.2)
    ss.add(name='optimizer', data_type=SSpaceType.INT, **getEnumRange(Optimizer))
    ss.add(name='shuffle', data_type=SSpaceType.BOOLEAN)
    ss.add(name='n_hidden_lstm_layers', data_type=SSpaceType.INT, min_value=1, max_value=2)
    ss.add(name='layer_sizes', data_type=SSpaceType.INT, min_value=10, max_value=200)
    ss.add(name='activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getActivationFuncsList(getSearchSpaceIdByName('searchspace')))
    ss.add(name='rec_activation_funcs', data_type=SSpaceType.CHOICE,
           choices=getRecurrentActivationFuncsList(getSearchSpaceIdByName('searchspace')))
    ss.add(name='dropout', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='rec_dropout', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.3)
    ss.add(name='kernel_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='bias_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='recurrent_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='activity_l1_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='kernel_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='bias_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='recurrent_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='activity_l2_regularizer', data_type=SSpaceType.FLOAT, min_value=0, max_value=0.1)
    ss.add(name='use_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='unit_forget_bias', data_type=SSpaceType.BOOLEAN)
    ss.add(name='go_backwards', data_type=SSpaceType.BOOLEAN)
    # only work if we load files during the nas callback
    if preprocess_on_nas:
        ss.add(name='normalize', data_type=SSpaceType.BOOLEAN)
        ss.add(name='normalize_prediction_feat', data_type=SSpaceType.BOOLEAN)
    return ss


def getSearchSpaceIdByName(ss_id: str) -> int:
    return _SEARCH_SPACES_MAP[ss_id.lower()]


def getSearchSpaceById(ss_id: Union[str, int], dataset_filename: Optional[str] = None, name: Optional[str] = None,
                       preprocess_on_nas: bool = False) -> SearchSpace:
    if type(ss_id) is str:
        ss_id = getSearchSpaceIdByName(ss_id)

    if ss_id < 0 or ss_id > len(_SEARCH_SPACES_MAP):
        raise ValueError(f'Invalid SS id {ss_id}')
    if ss_id == 0:
        return getSearchSpace(dataset_filename=dataset_filename, name=name, preprocess_on_nas=preprocess_on_nas)
    elif ss_id == 1:
        return getMidSearchSpace(dataset_filename=dataset_filename, name=name, preprocess_on_nas=preprocess_on_nas)
    elif ss_id == 2:
        return getDummySearchSpace(dataset_filename=dataset_filename, name=name, preprocess_on_nas=preprocess_on_nas)
    elif ss_id == 3:
        return getFastSearchSpace(dataset_filename=dataset_filename, name=name, preprocess_on_nas=preprocess_on_nas)
    else:
        raise ValueError(f'Not handled SS id {ss_id}')
