import copy
import functools
import itertools
import multiprocessing
import operator
import os
from collections import defaultdict, deque
from contextlib import contextmanager
from itertools import combinations as itercomb, chain
from typing import Iterable, TypeVar, Generator, Tuple, Set, List, FrozenSet, Mapping, Dict, Collection, Callable, \
    Sequence, Iterator, Optional, Union, Container, AbstractSet, Any, Type, Generic, Deque

import numpy as np
from numpy.random.mtrand import beta
# noinspection PyUnresolvedReferences
from scipy.special import binom

T = TypeVar('T')
KT = TypeVar('KT')
KT2 = TypeVar('KT2')
VT = TypeVar('VT')


class MutableBoolean:
    def __init__(self, b: bool):
        self.bool_val = b

    def make_true(self):
        self.bool_val = True

    def make_false(self):
        self.bool_val = False

    def flip(self):
        self.bool_val = not self.bool_val

    def __bool__(self):
        return self.bool_val


def d2t(d: Mapping[KT, VT]) -> Tuple[Tuple[KT, VT], ...]:
    """ dictionary to a sorted tuple of pairs of key and value """
    return tuple((k, d[k]) for k in sorted(d.keys()))


def is_debugging() -> bool:
    # whether the code is running under a debugging mode
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():
        return True
    else:
        return False


@functools.lru_cache()
def domain_with_bits(n_bits: int) -> Tuple[int, ...]:
    """ 1 --> (0,1), 3 --> (0,1,2,3,4,5,6,7), n -> (0,...,2^n-1) """
    return tuple(range(1 << n_bits))


def summation(xs: Iterator[T]) -> Optional[T]:
    """ Sums of elements based on __add__ operation. Returns None if nothing is added """
    s = None
    for x in xs:
        if s is None:
            s = x
        else:
            s = s + x
    return s


def max_default(values: Iterable[T], dflt: Optional[T]) -> Optional[T]:
    try:
        return max(values)
    except ValueError:
        return dflt


def identity(x: T) -> T:
    return x


def sort_with(a_collection: Collection[T], order_reference: Sequence[T]) -> Tuple[T, ...]:
    """ Sort values in a given collection based on the given order """
    return tuple(o for o in order_reference if o in a_collection)  # TODO efficiency


def first_diff_index(l1: Iterable[T], l2: Iterable[T]) -> int:
    """ Returns the minmum index where two iterables differ. Returns min(len(l1), len(l2)) if no diff found """
    i = 0
    for i, (e1, e2) in enumerate(zip(l1, l2)):
        if e1 != e2:
            return i
    return i + 1


def chop_up_list(listlike: Sequence[T], lengths: Iterable[int]) -> List[Sequence[T]]:
    """ Returns list of lists based on given list of lengths. If the given list is shorter than the sum of given lengths, shorter or empty lists will be filled. """
    at = 0
    ll = list()
    for length in lengths:
        ll.append(listlike[at:at + length])
        at += length
    return ll


def connected_components(vs: Iterable[T], adjs: Iterable[Sequence[T]]) -> Set[FrozenSet[T]]:
    """ connected components """
    clusters = {v: {v} for v in vs}
    for x, y in adjs:
        cl_x, cl_y = clusters[x], clusters[y]
        if cl_x is not cl_y:
            if len(cl_x) >= len(cl_y):
                cl_x |= cl_y
                for v in cl_y:
                    clusters[v] = cl_x
            else:
                cl_y |= cl_x
                for v in cl_x:
                    clusters[v] = cl_y

    return {frozenset(v) for v in clusters.values()}


def indexize(vs: Sequence[T]) -> Dict[T, int]:
    """ Return dictionary of elements and their positions

    If there are an element appears multiple times, then later index is used.
    """
    return {v: i for i, v in enumerate(vs)}


def truue():
    """ a functional alternative to True to avoid being caught by IDEs """
    return np.random.rand() < 1.1


def faalse():
    """ a functional alternative to False to avoid being caught by IDEs """
    return np.random.rand() > 1.1


def exiit(code=0):
    """ Exits without Python IDE disable the code below """
    if truue():
        exit(code)


def fair_coin() -> bool:
    """ uniformly generate 0 and 1 """
    return np.random.rand() < 0.5


def sets_sorted(xss: Collection[AbstractSet[T]],
                inner: Callable[[Collection[T]], AbstractSet[T]] = frozenset,
                output: Callable[[Collection[AbstractSet[T]]], Collection[AbstractSet[T]]] = tuple) -> Collection[
    AbstractSet[T]]:
    """ A collection of sets is sorted based on sorted tuple, returns output-type sequence of inner-type elements

    e.g., inner = frozenset, outer = tuple will return frozenset of tuples.
    """
    return output([inner(xs_) for xs_ in sorted([tuple(sorted(xs)) for xs in xss])])


def nondominated_sets(sets: Collection[AbstractSet[T]]) -> Set[AbstractSet[T]]:
    """ A set of sets that are not strictly subsumed by other sets. """
    sets_list = list(sets)
    to_remove = list()
    for i, set1 in enumerate(list(sets_list)):
        if any(set1 < set2 for set2 in sets_list[:i] + sets_list[i + 1:]):
            to_remove.append(i)
    for i in reversed(sorted(set(to_remove))):
        sets_list.pop(i)
    return set(as_fzsets(sets_list))


def rpartitions(xs: Iterable[T], n: int) -> List[List[T]]:
    """ Randomly split data into n (possibly empty) pieces """
    lists = [[] for _ in range(n)]
    for x in xs:
        lists[np.random.randint(n)].append(x)
    return lists


# noinspection PyTypeChecker
def pairs(vs: Iterable[T], sort=True) -> Iterator[Tuple[T, T]]:
    """ All pairs  """
    return itertools.combinations(sorted(vs) if sort else vs, 2)


# noinspection PyTypeChecker
def triples(vs: Iterable[T], sort=True) -> Iterator[Tuple[T, T, T]]:
    return itertools.combinations(sorted(vs) if sort else vs, 3)


def ordered_pairs(vs: Iterable[T], sort=True) -> Generator[Tuple[T, T], None, None]:
    """ Returns all ordered pairs alternating between original (or sorted) order and its reversed """
    for a, b in pairs(vs, sort=sort):
        yield a, b
        yield b, a


def intmap(vals: Sequence[int], dims: Sequence[int]) -> int:
    """ Interpret a sequence of integers as an integer where each position has its own dimension

    Generalization of interpreting k-ary sequence.
    `0` index is understood as the right-most bit.

    Does not check whether a value exceeds corresponding dim. (flexible...)

    e.g.,
    vals [3,1,5] with dims [5,6,7]
    3*(6*7) + 1*(7) + 5 = intmap([3,1,5], [5,6,7])
    """
    assert len(vals) == len(dims)
    x = 0
    for v, d in zip(vals, dims):
        x *= d
        x += v
    return x


def inverse_intmap(val: int, dims: Sequence[int]) -> Tuple[int, ...]:
    vals = [0] * len(dims)
    for i, d in enumerate(reversed(dims)):
        vals[i] = val % d
        val //= d
    return tuple(reversed(vals))


def odd_1s(x: int) -> int:
    """ Whether the number of 1s in the bits of x is odd """
    flag = 0
    while x:
        flag ^= x & 1
        x >>= 1
    return flag


def bits2int_dict(xsd: Dict[T, int]) -> int:
    """ Dictionary of binary values to an integer where bits are sorted based on their keys. """
    return bits2int([xsd[x] for x in sorted(xsd.keys())])


def bits2int(xs: Iterable[int]) -> int:
    """ Interpreting the sequence of bits (0, 1) to an integer, 0 index is the leftest bit. """
    s = 0
    for x in xs:
        s <<= 1
        s += x
    return s


def int2bits(x: int, minimum_n_bits: int) -> List[int]:
    """ Represent as bits with filling 0s if necessary to meet the specified minimum number of bits. """
    xs = []
    while x:
        xs.append(x & 1)
        x >>= 1
    xs = xs[::-1]
    xs = [0] * (minimum_n_bits - len(xs)) + xs
    return xs


def cumsum(xs: Iterable[T]) -> List[T]:
    """ Cumulative sum from left to right

    e.g.,
    [1,2,3] -> [1,3,6]
    """
    xs = list(xs)
    for i in range(1, len(xs)):
        xs[i] += xs[i - 1]
    return xs


def dict_only(a_dict: Dict[KT, VT], keys: Collection[KT]) -> Dict[KT, VT]:
    """ Copy of dictionary with the intersection of the original keys and the specified keys """
    return {k: a_dict[k] for k in keys if k in a_dict}


def dict_except(a_dict: Dict[KT, VT], keys: Collection[KT]) -> Dict[KT, VT]:
    """ Copy of dictionary with the original keys excluding the specified keys """
    return {k: v for k, v in a_dict.items() if k not in keys}


class missingdict(dict):
    """ default dictionary with a key-based value function """

    def __init__(self, fn):
        self.default_factory = fn
        super().__init__()

    def __missing__(self, key):
        default_value = self.default_factory(key)
        self[key] = default_value
        return default_value


def split_by(xs: Iterable[T], bool_func: Callable[[T], bool]) -> Tuple[List[T], List[T]]:
    """ Split iterable of Ts by a given boolean function taking T into two lists """
    yes = []
    no = []
    for x in xs:
        if bool_func(x):
            yes.append(x)
        else:
            no.append(x)

    return yes, no


def class_split_by(xs: Iterable[T], clazz: Type[KT]) -> Tuple[List[KT], List[T]]:
    yes, no = [], []
    for x in xs:
        if isinstance(x, clazz):
            yes.append(x)
        else:
            no.append(x)

    return yes, no


def notnone(xs: Iterable[T]) -> List[T]:
    """ non-none elements in `xs` as a list """
    return [x for x in xs if x is not None]


def prevs(ordered: Sequence[T], x: T) -> Sequence[T]:
    return ordered[:ordered.index(x)]


def dict_or(d1: Mapping[KT, VT], d2: Mapping[KT, VT]) -> Dict[KT, VT]:
    """ merge two dict, the second dict's values will override values from the first dict """
    d3 = dict(d1)
    d3.update(d2)
    return d3


def dict_ors(d1: Mapping[KT, VT], *dicts: Mapping[KT, VT]) -> Dict[KT, VT]:
    dx = dict(d1)
    for d in dicts:
        dx.update(d)
    return dx


def n_cpus(n_jobs: Optional[int] = None, mock_cpu_cnt=None) -> int:
    """
    Return a postive number of number of CPUs based `n_jobs` capped by actual number of CPUs

    if unspecified (`n_jobs is None`), return the number of CPUs.

    e.g.,
    There are 24 CPUs
    if n_jobs is 10, returns 10
    if n_jobs is 30, returns 24
    if n_jobs is None or -1, returns 24
    if n_jobs is -10, returns 15
    if n_jobs is -30, returns n_jobs with -6, which is 19
    ...

    """
    if n_jobs == 0:
        return 1
    cpu_cnt = mock_cpu_cnt if mock_cpu_cnt else multiprocessing.cpu_count()
    if n_jobs is None:
        return cpu_cnt
    if n_jobs < 0:
        n_jobs = 1 + (n_jobs % cpu_cnt)
    n_jobs = min(n_jobs, cpu_cnt)
    return n_jobs


def random_seeds(n: Optional[int] = None) -> Union[int, List[int]]:
    """Random seeds of given size or a random seed if n is None"""
    if n is None:
        return np.random.randint(np.iinfo(np.int32).max)
    else:
        return [np.random.randint(np.iinfo(np.int32).max) for _ in range(n)]


def subseq(xs: Sequence[T], indices: Iterable[int]) -> Union[Tuple[T, ...], List[T]]:
    if isinstance(xs, tuple):
        return tuple(xs[i] for i in indices)
    else:
        return [xs[i] for i in indices]


def pick_randomly(xs: Union[Collection[T], np.ndarray]) -> T:
    if isinstance(xs, Sequence):
        return xs[np.random.randint(len(xs))]
    elif isinstance(xs, Collection):
        xs = tuple(xs)
        return xs[np.random.randint(len(xs))]

    # noinspection PyUnresolvedReferences
    return xs[np.random.randint(len(xs))]  # e.g., numpy array


def optional_next(xs: Iterator[T], default: Optional[T] = None) -> T:
    try:
        x = next(xs)
        return x
    except StopIteration:
        return default


def bitmask(n_bits: int) -> int:
    return (1 << n_bits) - 1


def masked(value: int, n_bits: int) -> int:
    return value & ((1 << n_bits) - 1)


def rand_argmax(xs) -> int:
    """ One of (possibly many) argmax (based on `nanmax`) indices is returned randomly

    Its behavior is undefined when len(xs)==0
    """
    max_val = np.nanmax(xs)
    if max_val is np.nan:
        return pick_randomly(np.arange(len(xs)))

    max_indices = np.where(xs == max_val)[0]
    assert len(max_indices)

    if len(max_indices) == 1:
        return max_indices[0]
    else:
        return pick_randomly(max_indices)


def with_default(x: T, dflt: Optional[KT] = None) -> Union[T, Optional[KT]]:
    return x if x is not None else dflt


def dict_print(dd: dict, key_align='r', val_align='l', key_pad=0, val_pad=0, separator=': ', prefix='', postfix='',
               key_name='', val_name='', top=True, bottom=True, header=True):
    key_align = key_align.lower()
    val_align = val_align.lower()
    assert key_align in {'r', 'c', 'l'}
    assert val_align in {'r', 'c', 'l'}
    kstrs, vstrs = [key_name], [val_name]
    for k, v in dd.items():
        kstrs.append(str(k))
        vstrs.append(str(v))

    max_keylen = max(len(k) for k in kstrs) + key_pad
    max_vallen = max(len(v) for v in vstrs) + val_pad

    if key_align == 'r':
        kstrs = [k.rjust(max_keylen) for k in kstrs]
    elif key_align == 'c':
        kstrs = [k.center(max_keylen) for k in kstrs]
    elif key_align == 'l':
        kstrs = [k.ljust(max_keylen) for k in kstrs]

    if val_align == 'r':
        vstrs = [v.rjust(max_vallen) for v in vstrs]
    elif val_align == 'c':
        vstrs = [v.center(max_vallen) for v in vstrs]
    elif val_align == 'l':
        vstrs = [v.ljust(max_vallen) for v in vstrs]

    header_str = f'{prefix}{kstrs[0]}{separator}{vstrs[0]}{postfix}'
    if top:
        print('-' * len(header_str))
    if header:
        print(header_str)
        print('-' * len(header_str))
    for k, v in zip(kstrs[1:], vstrs[1:]):
        print(f'{prefix}{k}{separator}{v}{postfix}')
    if bottom:
        print('_' * len(header_str))


def bits_at(value: int, pos: int) -> int:
    """ read bit at the given position, where pos: offset from the rightmost """
    return (value >> pos) & 1


def bound(value, lower, upper):
    return max(lower, min(value, upper))


def rand_bw(lower: float, upper: float, precision=None) -> float:
    """ random number between lower and upper with a specified precision

    rand_bw(0,1,2) -> 0.00, 0.01, ... 0.99, 1.0

    """
    assert lower <= upper
    if lower == upper:
        return lower
    if precision is not None:
        return round(np.random.rand() * (upper - lower) + lower, precision)
    else:
        return np.random.rand() * (upper - lower) + lower


@contextmanager
def seeded(seed=None):
    if seed is not None:
        st0 = np.random.get_state()
        np.random.seed(seed)
        yield
        # noinspection PyTypeChecker
        np.random.set_state(st0)
    else:
        yield


def only(W: Iterable[T], Z: Container[T]) -> List[T]:
    if not Z:
        return []
    return [w for w in W if w in Z]


def excluding(W: Iterable[T], Z: Container[T]) -> List[T]:
    return [w for w in W if w not in Z]


def pop(xs: Set[T]) -> T:
    """ Pop an element from the given mutable set """
    x = next(iter(xs))
    xs.remove(x)
    return x


def set_and(sets: Iterable[AbstractSet[T]]) -> FrozenSet[T]:
    temp = None
    for s in sets:
        if temp is None:
            temp = s
        else:
            temp &= s
        if not temp:
            return frozenset(temp)

    if temp is None:
        return frozenset()

    return frozenset(temp)


def set_union(sets: Iterable[Collection[T]]) -> Set[T]:
    return set(chain(*sets))


def set_ands(xss: Iterable[AbstractSet[T]]) -> Optional[FrozenSet[T]]:
    xss = iter(xss)
    outs = optional_next(xss)
    if outs is None:
        return None

    for xs in xss:
        outs &= xs
    return frozenset(outs)


def set_ors(xss: Iterable[AbstractSet[T]]) -> FrozenSet[T]:
    x = set()
    for v in xss:
        x = x | v
    return frozenset(x)


def fzset_union(sets: Iterable[Collection[T]]) -> FrozenSet[T]:
    return frozenset(chain(*sets))


union = fzset_union


def as_lists(xs: Iterable[Iterable[T]]) -> Generator[List[T], None, None]:
    """ Generate list-applied elements of a given iterable """
    for x in xs:
        yield list(x)


def as_tuples(xss: Iterable[Iterable[T]]) -> Generator[Tuple[T, ...], None, None]:
    """ Generate tuple-applied elements of a given iterable """
    for xs in xss:
        yield tuple(xs)


def as_sortups(xss: Iterable[Iterable[T]]) -> Generator[Tuple[T, ...], None, None]:
    """ Generate tuple-applied elements of a given iterable """
    for xs in xss:
        yield sortup(xs)


def as_sets(xss: Iterable[Iterable[T]]) -> Generator[Set[T], None, None]:
    """ Generate set-applied elements of a given iterable """
    for xs in xss:
        yield set(xs)


def as_fzsets(xss: Iterable[Iterable[T]]) -> Generator[FrozenSet[T], None, None]:
    """ Generate frozenset-applied elements of a given iterable """
    for xs in xss:
        yield frozenset(xs)


def nonempties(xs: Iterable[T]) -> Generator[T, None, None]:
    for x in xs:
        if x:
            yield x


def sortup(xs: Iterable[T]) -> Tuple[T, ...]:
    """ Syntactic sugar for tuple(sorted(...)) """
    # sorted tuple
    return tuple(sorted(xs))


def sortup2(xxs: Iterable[Iterable[T]]) -> Tuple[Tuple[T, ...], ...]:
    # twice sorted tuples
    return sortup([sortup(xs) for xs in xxs])


def shuffled(xs: Iterable[T], reproducible=False) -> List[T]:
    """ A list with shuffled elements """
    if reproducible:
        xs = sorted(xs)
    else:
        xs = list(xs)
    np.random.shuffle(xs)
    return xs


def combinations(xs: Iterable[T]) -> Generator[Tuple[T, ...], None, None]:
    """ all combinations of given in the order of increasing its size """
    xs = list(xs)
    for i in range(len(xs) + 1):
        for comb in itercomb(xs, i):
            yield comb


def subsets(xs: AbstractSet[T], exit_per_comb=None, exit_per_len=None, maxsize=None) -> Generator[
    FrozenSet[T], None, None]:
    xs = sorted(xs)
    for i in range((len(xs) + 1) if maxsize is None else min(maxsize, len(xs) + 1)):
        for comb in itercomb(xs, i):
            yield frozenset(comb)
            if exit_per_comb:
                return
        if exit_per_len:
            return


def reversed_combinations(xs: Iterable[T]) -> Generator[Tuple[T, ...], None, None]:
    """ all combinations of given in the order of decreasing its size """
    xs = list(xs)
    for i in reversed(range(len(xs) + 1)):
        for comb in itercomb(xs, i):
            yield comb


def random_comb(xs: Iterable[T], nonempty=False) -> List[T]:
    xs = list(xs)
    pp = np.array([binom(len(xs), k) for k in range(len(xs) + 1)])
    pp /= np.sum(pp)
    how_many = np.random.choice(np.arange(len(xs) + 1), p=pp)
    if nonempty:
        idxs = np.random.choice(len(xs), max(1, how_many), replace=False)
    else:
        idxs = np.random.choice(len(xs), how_many, replace=False)
    return [xs[idx] for idx in idxs]


def xors(values: Iterable[int], start=0) -> int:
    x = start
    for v in values:
        x = x ^ v
    return x


def ors(values: Iterable[int]) -> int:
    x = 0
    for v in values:
        x = x | v
    return x


def mults(values: Iterable[float]) -> float:
    x = 1
    for v in values:
        x = x * v
    return x


def mults_zero(values: Iterable[float]) -> float:
    """ whatever nan or inf, -inf exists, if 0 appears, return 0 """
    x = 1
    for v in values:
        if v == 0.0:
            return 0.0
        x = x * v
    return x


def rand_subset(xs: Iterable[T], prob=0.5) -> FrozenSet[T]:
    """ Random subset of a given set """
    return frozenset({x for x in xs if np.random.rand() < prob})


def mkdirs(newdir: str):
    os.makedirs(newdir, mode=0o777, exist_ok=True)


def np_ands(xs, *others) -> Optional[np.ndarray]:
    """ Logical ands of multiple ndarrays """
    if others:
        xs = [xs, *others]
    temp = None
    for x in xs:
        if temp is None:
            temp = x
        else:
            temp = np.logical_and(temp, x)
    return temp


def unique(vs: Iterable[T]) -> List[T]:
    seen = set()
    at = list()
    for v in vs:
        if v not in seen:
            at.append(v)
            seen.add(v)
    return at


def unique_eq(vs: Iterable[T]) -> List[T]:
    at = list()
    for v in vs:
        if all(a != v for a in at):
            at.append(v)
    return at


def sl_group_by(xs: Iterable[T],
                criteria: Callable[[T], KT],
                store_as: Callable[[Collection[T]], Collection[T]] = tuple,
                to_sort=False,
                deduplicate=False,
                as_defaultdict=True,
                transformer: Optional[Callable[[T], KT2]] = None) -> Mapping[KT, Collection[Union[T, KT2]]]:
    """
    Categorize given elements based on `criteria` and return as a dictionary where
    a key corresponds to a category and
    its value corresponds to a list of elements in the category.

    Parameters
    ----------
    xs:
        elements
    criteria:
        function mapping elements to categories
    store_as:

    to_sort:
        whether to sort elements in the values
    deduplicate:
        whether to keep only unique elements in the values
    as_defaultdict
    transformer

    Returns
    -------

    """
    sorter = defaultdict(list)  # type: Dict[KT, List[T]]
    for x in xs:
        sorter[criteria(x)].append(x if transformer is None else transformer(x))

    if deduplicate:
        apply_to_values(sorter, unique)

    if to_sort:
        apply_to_values(sorter, sorted)

    apply_to_values(sorter, store_as)

    if as_defaultdict:
        sorter_out = defaultdict(store_as)
        sorter_out.update(sorter)
        return sorter_out
    else:
        return dict(sorter)


def op_dicts(dict1: Mapping[KT, VT], dict2: Mapping[KT, VT], op=operator.add, to_copy: bool = True) -> Dict[KT, VT]:
    """ Apply an operation on values of two dictionaries

    If a key exists only in one dictionary, its value is simply copied.
    e.g.,
        op_dicts({1:2, 2:3}, {2:4,3:5}) = {1:2, 2:7, 3:5}
        op_dicts({1:2, 2:3}, {2:4,3:5}, op=operator.mul) = {1:2, 2:12, 3:5}
    """

    def transform(x):
        if to_copy:
            return copy.copy(x)
        else:
            return x

    out_dict = dict()
    for k in dict1.keys() | dict2.keys():
        if k in dict1 and k in dict2:
            out_dict[k] = op(dict1[k], dict2[k])
        else:
            if k in dict1:
                out_dict[k] = transform(dict1[k])
            else:
                out_dict[k] = transform(dict2[k])

    return out_dict


def sorted_dict(pdict: Mapping[KT, VT]) -> Dict[KT, VT]:
    """ sort dict items by keys """
    return {k: pdict[k] for k in sorted(pdict.keys())}


def apply_to_values(dic: Dict[KT, VT], fun: Callable[[VT], Any] = identity):
    for k, v in list(dic.items()):
        dic[k] = fun(v)


class visited_queue(Generic[T]):
    def __init__(self, elements: Iterable[T] = tuple(), mapper: Callable[[T], VT] = identity):
        """
        Queue that does not prohibit adding an element
        if the element has already been queued before
        regardless of whether the element is in the queue now.
        """
        self.mapper = mapper  # the test of equality among elements is based on this mapper
        self.queue = deque()  # type: Deque[T]
        self.visited = set()  # type: Set[VT]
        self.push_all(elements)

    def push_all(self, elems: Iterable[T]):
        for x in elems:
            if (x_mapped := self.mapper(x)) not in self.visited:
                self.queue.append(x)
                self.visited.add(x_mapped)

    def push(self, elem: T):
        if (x_mapped := self.mapper(elem)) not in self.visited:
            self.queue.append(elem)
            self.visited.add(x_mapped)

    def pop(self) -> T:
        return self.queue.pop()

    def __bool__(self):
        """ Whether the queue is not empty """
        return bool(self.queue)
