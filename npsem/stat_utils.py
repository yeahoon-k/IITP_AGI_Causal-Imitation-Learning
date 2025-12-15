import abc
import dataclasses
from typing import Tuple, Union, Dict, FrozenSet, Iterable, TypeVar

import numpy as np
from numpy.random.mtrand import beta
from scipy.optimize import minimize, brenth
from tqdm import trange

from npsem.utils import sortup, seeded

KT = TypeVar('KT')
VT = TypeVar('VT')
TTSV = Tuple[Tuple[str, VT], ...]


class IQueryable(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def query(self, outcome: Tuple[Tuple[str, int]], condition: Tuple[Tuple[str, int]],
              intervention: Tuple[Tuple[str, int]]) -> float:
        pass


# P(Y|do(X),Z)
@dataclasses.dataclass(init=False, unsafe_hash=True, eq=True, order=True)
class ProbSpec:
    """ Signature of probability with atomic intervention, measurement (outcome), and conditionals

    P(Y|do(X),Z)
    Uses sorted tuple for a set of variables
    """
    intervention: Tuple[str, ...] = tuple()
    outcome: Tuple[str, ...] = tuple()
    condition: Tuple[str, ...] = tuple()

    def __init__(self, intervention: Union[str, Iterable[str]] = tuple(), outcome: Union[str, Iterable[str]] = tuple(), condition: Union[str, Iterable[str]] = tuple()):
        if isinstance(intervention, str):
            intervention = [intervention]
        if isinstance(outcome, str):
            outcome = [outcome]
        if isinstance(condition, str):
            condition = [condition]
        self.intervention = sortup(intervention)
        self.outcome = sortup(outcome)
        self.condition = sortup(condition)

    def with_Y(self, new_Y: Iterable[str]) -> 'ProbSpec':
        """ Change outcome """
        return ProbSpec(self.X, new_Y, self.Z)

    def with_Z(self, new_Z: Iterable[str]) -> 'ProbSpec':
        """ Change outcome """
        return ProbSpec(self.X, self.Y, new_Z)

    def with_X(self, new_X: Iterable[str]) -> 'ProbSpec':
        """ Change outcome """
        return ProbSpec(new_X, self.Y, self.Z)

    def latex(self) -> str:
        latex_str = 'P'
        if self.X:
            latex_str += '_{' + (','.join(self.intervention)) + '}'
        latex_str += "(" + (','.join(self.outcome))
        if self.Z:
            latex_str += '|' + (','.join(self.condition))
        latex_str += ')'
        return latex_str

    def __str__(self):
        given = ''
        if self.intervention:
            given += f'|do({",".join(self.intervention)})'
        if self.condition:
            if not given:
                given += '| '
            else:
                given += ', '
            given += f'{",".join(self.condition)}'

        return f'P({",".join(self.outcome)}{given})'

    def __repr__(self):
        if self.condition:
            return f'ProbSpec({self.intervention},{self.outcome},{self.condition})'
        else:
            return f'ProbSpec({self.intervention},{self.outcome})'

    def __iter__(self):
        return iter([self.Xs, self.Ys, self.Zs])

    @property
    def X(self) -> Tuple[str, ...]:
        return self.intervention

    @property
    def Y(self) -> Tuple[str, ...]:
        return self.outcome

    @property
    def Z(self) -> Tuple[str, ...]:
        return self.condition

    @property
    def Xs(self) -> FrozenSet[str]:
        return frozenset(self.intervention)

    @property
    def Ys(self) -> FrozenSet[str]:
        return frozenset(self.outcome)

    @property
    def Zs(self) -> FrozenSet[str]:
        return frozenset(self.condition)

    @property
    def XYZs(self) -> FrozenSet[str]:
        return self.Xs | self.Ys | self.Zs


# P(Y=y|do(X=x),Z=z)
@dataclasses.dataclass
class ProbSpecInst:
    """ Instantiation of probability specification

    P(Y=y|do(X=x),Z=z)
    """
    spec: ProbSpec
    intervention_values: Tuple = tuple()
    outcome_values: Tuple = tuple()
    condition_values: Tuple = tuple()

    @staticmethod
    def create(intervention_dict: Dict[str, VT], outcome_dict: Dict[str, VT], condition_dict: Dict[str, VT]):
        X, x = zip(*sorted(intervention_dict.items()))
        Y, y = zip(*sorted(outcome_dict.items()))
        Z, z = zip(*sorted(condition_dict.items()))
        return ProbSpecInst(ProbSpec(X, Y, Z), x, y, z)

    def as_dicts(self) -> Tuple[Dict[str, VT], Dict[str, VT], Dict[str, VT]]:
        return dict(self.Xx), dict(self.Yy), dict(self.Zz)

    def as_tuples(self) -> Tuple[TTSV, TTSV, TTSV]:
        return tuple(self.Xx), tuple(self.Yy), tuple(self.Zz)

    @property
    def Y(self) -> Tuple[str, ...]:
        return self.spec.Y

    @property
    def X(self) -> Tuple[str, ...]:
        return self.spec.X

    @property
    def Z(self) -> Tuple[str, ...]:
        return self.spec.Z

    @property
    def Xs(self) -> FrozenSet[str]:
        return frozenset(self.spec.intervention)

    @property
    def Ys(self) -> FrozenSet[str]:
        return frozenset(self.spec.outcome)

    @property
    def Zs(self) -> FrozenSet[str]:
        return frozenset(self.spec.condition)

    @property
    def y(self) -> Tuple[VT, ...]:
        return self.outcome_values

    @property
    def x(self) -> Tuple[VT, ...]:
        return self.intervention_values

    @property
    def z(self) -> Tuple[VT, ...]:
        return self.condition_values

    @property
    def Xx(self) -> TTSV:
        return tuple(zip(self.X, self.x))

    @property
    def Yy(self) -> TTSV:
        return tuple(zip(self.Y, self.y))

    @property
    def Zz(self) -> TTSV:
        return tuple(zip(self.Z, self.z))


def is_marginalized(p: ProbSpec, q: ProbSpec) -> bool:
    assert p != q
    return (p.Ys > q.Ys and p.X == q.X and p.Z == q.Z) or (p.Ys < q.Ys and p.X == q.X and p.Z == q.Z)


def covariate_matrix(estimates: np.ndarray) -> np.ndarray:
    if estimates.shape[1] == 1:
        cov_matrix = np.cov(estimates.T)
        cov_matrix = np.array([[cov_matrix]])
    else:
        cov_matrix = np.cov(estimates.T)

    return cov_matrix


def minimum_variance_weighting(estimates: np.ndarray, cov: np.ndarray, default_index=0, with_variance=False) -> Union[float, Tuple[float, float]]:
    # if cov matrix is made of small values, SLSQP might be stop early.
    # Adjusting tol is one thing, modifying the cov matrix is another workaround.
    assert 0 < np.max(cov)
    if np.max(cov) < 1.:
        cov_factor = 1. / np.max(cov)
    else:
        cov_factor = 1.
    cov *= cov_factor
    if with_variance:
        e, v = __minimum_variance_weighting00(estimates, cov, default_index, True)
        return e, v / cov_factor
    else:
        return __minimum_variance_weighting00(estimates, cov, default_index, False)


def __minimum_variance_weighting00(estimates, cov, default_index=0, with_variance=False, tol=1e-8) -> Union[float, Tuple[float, float]]:
    """ Given the estimates and their covariance matrix, find minimum variance estimation with its variance """
    n_dim = len(cov)
    assert len(estimates) == n_dim
    if n_dim == 1:
        if with_variance:
            return estimates[0], cov[0, 0]
        else:
            return estimates[0]

    if n_dim == 2 and cov[0, 1] == 0:
        w0, w1 = 1 / cov[0, 0], 1 / cov[1, 1]
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        Q_hat = estimates[0] * w0 + estimates[1] * w1
        if with_variance:
            Q_var = w0 ** 2 * cov[0, 0] + w1 ** 2 * cov[1, 1]
            return Q_hat, Q_var
        else:
            return Q_hat

    def func_to_min(xs):
        xs = np.asarray(xs)
        return (xs @ cov) @ xs

    # noinspection PyTypeChecker
    res = minimize(func_to_min,
                   np.ones(n_dim) / n_dim,
                   method='SLSQP',
                   bounds=((0, 1),) * n_dim,
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},),
                   tol=tol)

    if res.success:
        Q_hat = np.asarray(estimates) @ np.asarray(res.x)
        if with_variance:
            Q_var = func_to_min(res.x)
            return Q_hat, Q_var
        else:
            return Q_hat
    else:
        if with_variance:
            return estimates[default_index], cov[default_index, default_index]
        else:
            return estimates[default_index]


def minimum_variance_weight(cov, tol=1e-8):
    """ Given the estimates and their covariance matrix, find minimum variance estimation with its variance """
    n_dim = len(cov)
    if n_dim == 1:
        return np.array([1.])

    if n_dim == 2 and cov[0, 1] == 0:
        w0, w1 = 1 / cov[0, 0], 1 / cov[1, 1]
        w0, w1 = w0 / (w0 + w1), w1 / (w0 + w1)
        if np.isnan(w0) or np.isnan(w1):
            return np.array([0.5, 0.5])
        return np.array([w0, w1])

    def func_to_min(xs):
        xs = np.asarray(xs)
        return (xs @ cov) @ xs

    # noinspection PyTypeChecker
    res = minimize(func_to_min,
                   np.ones(n_dim) / n_dim,
                   method='SLSQP',
                   bounds=((0, 1),) * n_dim,
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},),
                   tol=tol)

    if res.success:
        return np.asarray(res.x)
    raise AssertionError('failed MVW')


def confidence_interval(data: np.ndarray, axis: int, n_bootstrap: int = 2000, ci=0.95, func=np.mean, seed=None):
    assert 0 <= ci <= 1
    if ci < 0.5:
        ci = 1 - ci  # ci to be upper, 1-ci to be lower

    the_length = data.shape[axis]
    b_shape = list(data.shape)
    b_shape.pop(axis)
    b_shape.insert(0, n_bootstrap)
    b_results = np.zeros(tuple(b_shape))

    with seeded(seed):
        for b in trange(n_bootstrap):
            b_data = np.take(data, np.random.choice(the_length, the_length), axis=axis)
            b_results[b] = func(b_data, axis=axis)

    upper = np.percentile(b_results, ci * 100, axis=0)
    lower = np.percentile(b_results, (1 - ci) * 100, axis=0)
    return lower, upper


def beta_sim(e, v, size: int):
    """ Random samples of beta distribution """
    a, b = beta_parameters(e, v)
    return beta(a, b, size)


def beta_parameters(e, v):
    if not (0 < e < 1 and 0 < v <= 0.25):
        return np.nan, np.nan
    a, b = e, 1 - e
    upper = 1 / v
    try:
        xx = brenth(lambda x: beta_var(a * x, b * x) - v, 1e-20, upper)  # noqa
        return a * xx, b * xx
    except ValueError:
        return np.nan, np.nan


def mult_var(e_x, e_y, var_x, var_y):
    """ Variance of multiplication of two random variables """
    return var_x * var_y + var_x * (e_y ** 2) + var_y * (e_x ** 2)


def beta_var(alpha, betta):
    """ Variance of beta distribution """
    assert alpha > 0 and betta > 0, f"non-positive parameters: {alpha}, {betta}"
    return (alpha * betta) / ((alpha + betta + 1) * (alpha + betta) * (alpha + betta))
