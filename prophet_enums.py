from enum import Enum, IntEnum, auto


class Optimizer(IntEnum):
    SGD = auto()
    ADAM = auto()
    RMSPROP = auto()

    def toKerasName(self) -> str:
        if self == Optimizer.SGD:
            return 'sgd'
        elif self == Optimizer.ADAM:
            return 'adam'
        elif self == Optimizer.RMSPROP:
            return 'rmsprop'
        raise ValueError('Strange error, invalid enum')

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def getAll() -> list[Enum]:
        return list(map(lambda c: c, Optimizer))

    @staticmethod
    def getUpperBound() -> int:
        return max([x.value for x in Optimizer.getAll()])


class Loss(IntEnum):
    BINARY_CROSSENTROPY = auto()
    CATEGORICAL_CROSSENTROPY = auto()
    MEAN_SQUARED_ERROR = auto()
    MEAN_ABSOLUTE_ERROR = auto()
    ROOT_MEAN_SQUARED_ERROR = auto()

    def __str__(self) -> str:
        return self.name

    def toKerasName(self) -> str:
        if self == Loss.BINARY_CROSSENTROPY:
            return 'binary_crossentropy'
        elif self == Loss.CATEGORICAL_CROSSENTROPY:
            return 'categorical_crossentropy'
        elif self == Loss.MEAN_SQUARED_ERROR:
            return 'mean_squared_error'
        elif self == Loss.MEAN_ABSOLUTE_ERROR:
            return 'mean_absolute_error'
        elif self == Loss.ROOT_MEAN_SQUARED_ERROR:
            return 'root_mean_squared_error'
        raise ValueError('Strange error, invalid enum')

    @staticmethod
    def getAll() -> list[Enum]:
        return list(map(lambda c: c, Loss))

    @staticmethod
    def getUpperBound() -> int:
        return max([x.value for x in Loss.getAll()])


class ActivationFunc(IntEnum):
    LEAKY_RELU = auto()
    SIGMOID = auto()
    TANH = auto()
    EXPONENTIAL = auto()
    LINEAR = auto()
    HARD_SIGMOID = auto()
    SOFTMAX = auto()
    SOFTPLUS = auto()
    SOFTSIGN = auto()
    SELU = auto()
    ELU = auto()
    RELU = auto()

    def __str__(self) -> str:
        return self.name

    def toKerasName(self) -> str:
        if self == ActivationFunc.LEAKY_RELU:
            return 'leaky_relu'
        if self == ActivationFunc.RELU:
            return 'relu'
        elif self == ActivationFunc.SOFTMAX:
            return 'softmax'
        elif self == ActivationFunc.SIGMOID:
            return 'sigmoid'
        elif self == ActivationFunc.HARD_SIGMOID:
            return 'hard_sigmoid'
        elif self == ActivationFunc.TANH:
            return 'tanh'
        elif self == ActivationFunc.SOFTPLUS:
            return 'softplus'
        elif self == ActivationFunc.SOFTSIGN:
            return 'softsign'
        elif self == ActivationFunc.SELU:
            return 'selu'
        elif self == ActivationFunc.ELU:
            return 'elu'
        elif self == ActivationFunc.EXPONENTIAL:
            return 'exponential'
        elif self == ActivationFunc.LINEAR:
            return 'linear'
        raise ValueError('Strange error, invalid enum')

    @staticmethod
    def getAll() -> list[Enum]:
        return list(map(lambda c: c, ActivationFunc))

    @staticmethod
    def getUpperBound() -> int:
        return max([x.value for x in ActivationFunc.getAll()])
