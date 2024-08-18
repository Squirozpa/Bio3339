from dataclasses import dataclass, field
import logging
import copy


@dataclass
class Individual:
    """Dataclass that represents an individual in the genetic algorithm"""
    iteration: int
    genes: list
    population_id: int
    _id: int | None = field(default=None, init=False)
    _fitness: float | int | None = field(default=None, init=False)

    @property
    def fitness(self) -> float | int:
        """Fitness getter method"""
        if self._fitness is None:
            raise ValueError("Fitness has not been calculated")
        return self._fitness

    @fitness.setter
    def fitness(self, value: float | int) -> None:
        """Fitness setter method"""
        if not isinstance(value, (float, int)):
            raise ValueError(
                f"Fitness must be a float or int, not {type(value)}")
        if self._fitness is not None:
            raise ValueError("Fitness has already been calculated")
        if value < 0:
            raise ValueError("Fitness cannot be negative")
        self._fitness = value

    @property
    def id(self) -> int:
        """ID getter method"""
        if self._id is None:
            raise ValueError("ID has not been set")
        return self._id

    @id.setter
    def id(self, value: int) -> None:
        """ID setter method"""
        if not isinstance(value, int):
            raise ValueError(f"ID must be an int, not {type(value)}")
        if self._id is not None:
            raise ValueError("ID has already been set")
        self._id = value

    def __post_init__(self):
        """Post init method that sets the ID"""
        self.loggger = logging.getLogger(__name__)
        self._id = id(self)
        self._population_id = self.population_id

    # region Magic Methods
    def __str__(self):
        return f'Individual: {self.id}, Fitness: {self.fitness}, Iteration {self.iteration}'

    def __repr__(self):
        return f'Individual: {self.id}, Fitness: {self.fitness}, Iteration {self.iteration},' + \
            f'Population: {self.population_id}, Genes: {self.genes}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.genes == other.genes

    @staticmethod
    def _validate_other(other: "Individual"):
        if not isinstance(other, Individual):
            raise ValueError("Can't compare individual with non-individual")
        if other.fitness is None:
            raise ValueError(
                "Can't compare individuals with None fitness values")
        return other.fitness

    def __lt__(self, other: "Individual") -> bool:
        other_fitness = self._validate_other(other)

        return self.fitness < other_fitness

    def __le__(self, other: "Individual") -> bool:
        other_fitness = self._validate_other(other)
        # If the fitness is the same, compare the iteration
        if self.fitness == other_fitness:
            return self.iteration <= other.iteration
        else:
            return self.fitness <= other_fitness

    def __gt__(self, other: "Individual"):
        other_fitness = self._validate_other(other)
        return self.fitness > other_fitness

    def __ge__(self, other: "Individual"):
        other_fitness = self._validate_other(other)
        # If the fitness is the same, compare the iteration
        if self.fitness == other.fitness:
            return self.iteration >= other.iteration
        return self.fitness >= other_fitness

    def __ne__(self, other: object):
        if not isinstance(other, Individual):
            raise ValueError("Can't compare individual with non-individual")
        return self.genes != other.genes

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, key):
        return self.genes[key]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __iter__(self):
        return iter(self.genes)

    def __contains__(self, item):
        return item in self.genes

    def __add__(self, other: "Individual"):
        if isinstance(other, Individual):
            if self.fitness is not None and other.fitness is not None:
                return self.fitness + other.fitness
            else:
                raise ValueError(
                    "Can't add individuals with None fitness values")
        else:
            raise ValueError("Can't add individual with non-individual")

    def deepcopy(self, memo=None):
        """Deepcopy method."""
        new_indidivual = copy.deepcopy(self, memo)  # type: ignore
        new_indidivual._id = id(  # pylint: disable=protected-access
            new_indidivual)
        return new_indidivual

    def __bool__(self):
        return bool(self.genes)

    def __int__(self):
        return int(''.join(str(bit) for bit in self.genes), 2)

    # endregion
