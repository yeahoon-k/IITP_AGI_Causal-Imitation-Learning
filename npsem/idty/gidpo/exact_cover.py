from collections import defaultdict
from typing import Collection, AbstractSet, Optional, Sequence, FrozenSet, TypeVar

T = TypeVar('T')

Chunk = FrozenSet[T]


class ExactCoverSolver:
    """ Simple implementation of exact cover solver """

    def __init__(self, pieces: Collection[T], subsets: AbstractSet[Chunk]):
        """

        Parameters
        ----------
        pieces :
            a universe
        subsets :
            a subcollection of the universe
        """
        self.subsets = subsets
        self.membership = defaultdict(set)
        for subset in subsets:
            for element in subset:
                self.membership[element].add(subset)

        self.elements = frozenset(pieces)  # = universe
        self.failed = not all(self.membership[elem] for elem in self.elements)

    def print(self):
        if self.subsets:
            print(*list(self.subsets), sep=', ')

    def solve(self) -> Optional[Sequence[Chunk]]:
        if self.failed:
            return None
        return self.__solve(set(), [])

    def __solve(self, covered, selected) -> Optional[Sequence[Chunk]]:
        # if there is no piece left to fill.
        if self.elements == covered:
            return sorted(selected)

        # for every uncovered piece
        for uncovered in sorted(self.elements - covered, key=lambda up: len(self.membership[up])):
            # if a candidate piece
            for subset in filter(lambda _: covered.isdisjoint(_), self.membership[uncovered]):
                output = self.__solve(covered | set(subset), selected + [subset])
                if output is not None:
                    return sorted(output)

        return None
