"""
Class for population of individuals. This class is designed to hold a list of individuals, which its
genes are binary vectors. Also contains all the logic to generate a new population from the current
one, such as selection, crossover, and mutation. The class is designed to be used in a genetic
algorithm, but can be used in any optimization algorithm that requires a binary representation of
the solution. It contains all the logic to run the genetic algorithm, refer to the README for more
information on how to use it.
"""

# Standard Library Imports
from dataclasses import dataclass, field
from typing import Type, cast
from abc import ABC, abstractmethod
from typing import Optional
import random
import logging
import copy
import time

# Local Imports
from genetic_algorithm.abstract_classes import (
    GenePool, StrategyConfig, FitnessStrategy, CrossoverStrategy, SurvivorSelectionStrategy, ParentSelectionStrategy,
    PopulationParams, PopulationState, StrategyConfig)
from genetic_algorithm.population_classes import Individual
from logger import CustomLogger
####################################################################################################


class AbstractPopulation(ABC):

    # region Initialization

    def __init__(self,
                 strategies: StrategyConfig,
                 params: PopulationParams,
                 state: Optional[PopulationState] = None,
                 name: Optional[str] = None,
                 shared_dict: dict[tuple, float] | None = None
                 ):

        self.logger = logging.getLogger(__name__)
        self.logger.debug(
            "Initializing Abstract Population with name %s", name)

        if not isinstance(strategies, StrategyConfig):
            raise TypeError(
                "Strategies must be an instance of StrategyConfig")
        self.strategies = strategies

        if not isinstance(params, PopulationParams):
            raise TypeError("Params must be an instance of PopulationParams")
        self.params = params

        if state is not None and not isinstance(state, PopulationState):
            raise TypeError("State must be an instance of PopulationState")
        self.state = state if state is not None else None

        self._id = id(self)
        self.name = name if name is not None else f"Population {self.id}"

    # endregion

    # region Properties

    @property
    def id(self):
        return self._id

    def refresh_id(self):
        self._id = id(self)

    # endregion

    # region Magic Methods

    def __repr__(self):
        return (f'*---------------------------------------- Report for Population: '
                f"{self.id} ----------------------------------------*\n"
                f'Population Size: {self.params.population_size}\n'
                f'Best Individual: {self.state.best_individual}\n'
                f'Iteration: {self.state.iteration}\n'
                f'Seed: {self.params.seed}\n'
                f'Total Fitness: {self.state.total_fitness}\n'
                f'*---------------------------------------- End Report for Population: '
                f'{self.id} ----------------------------------------*\n') if self.state is not None else ("Population not initialized")

    def __str__(self):
        return (f"Population ID: {self.id}, Size: {self.params.population_size}, Best Individual: "
                f"{self.state.best_individual}, Iteration: {
                    self.state.iteration}, "
                f" Seed: {self.params.seed}, Total Fitness: {self.state.total_fitness}") if self.state is not None else ("Population not initialized")

    def __len__(self):
        return len(self.state.individuals) if self.state is not None else 0

    def __getitem__(self, key: int):
        if not isinstance(key, int):
            raise TypeError("Index must be an integer.")
        return self.state.individuals[key] if self.state is not None else None

    def __setitem__(self, key: int, value: Individual):
        if not isinstance(key, int):
            raise TypeError("Index must be an integer.")
        if not isinstance(value, Individual):
            raise TypeError(
                f"Value must be an instance of 'Individual', not '{type(value).__name__}'")
        if self.state is None:
            raise ValueError("Population not initialized")
        self.state.individuals[key] = value
        return self

    def __iter__(self):
        if self.state is None:
            raise ValueError("Population not initialized")
        return iter(self.state.individuals)

    def __contains__(self, item: Individual):
        if not isinstance(item, Individual):
            raise TypeError(
                f"Unsupported operand type for 'in': 'Individual' expected, "
                f"'{type(item).__name__}' found")
        if self.state is None:
            raise ValueError("Population not initialized")
        return item in self.state.individuals

    # endregion

    # region set_strategy_methods

    def set_random(self, seed: Optional[int | str | float] = None):
        if seed is not None:
            self.params.seed = seed
        if self.params.seed is not None:
            self.params.random = random.Random(self.params.seed)
        else:
            self.params.random = random.Random()

    def set_fitness_strategy(self, strategy):
        self.params.random = cast(random.Random, self.params.random)
        if isinstance(strategy, FitnessStrategy):
            self.strategies.fitness = copy.deepcopy(strategy)
        elif issubclass(strategy, FitnessStrategy):
            self.strategies.fitness = strategy(
                gene_pool=self.strategies.gene_pool)
        else:
            raise ValueError("Invalid fitness strategy")

    def set_crossover_strategy(self, strategy):
        self.params.random = cast(random.Random, self.params.random)
        if isinstance(strategy, CrossoverStrategy):
            self.strategies.crossover = copy.deepcopy(strategy)
            self.strategies.crossover.population = self
            self.strategies.crossover.random = self.params.random
        elif issubclass(strategy, CrossoverStrategy):
            self.strategies.crossover = strategy(
                population=self, random=self.params.random)
        else:
            raise ValueError("Invalid crossover strategy")

    def set_survivor_selection_strategy(self, strategy):
        self.params.random = cast(random.Random, self.params.random)
        if isinstance(strategy, SurvivorSelectionStrategy):
            self.strategies.survivor_selection = copy.deepcopy(strategy)
            self.strategies.survivor_selection.population = self
            self.strategies.survivor_selection.random = self.params.random
        elif issubclass(strategy, SurvivorSelectionStrategy):
            self.strategies.survivor_selection = strategy(
                population=self, random=self.params.random)
        else:
            raise ValueError("Invalid survivor selection strategy")

    def set_parent_selection_strategy(self, strategy):
        self.params.random = cast(random.Random, self.params.random)
        if isinstance(strategy, ParentSelectionStrategy):
            self.strategies.parent_selection = copy.deepcopy(strategy)
            self.strategies.parent_selection.population = self
            self.strategies.parent_selection.random = self.params.random
        elif issubclass(strategy, ParentSelectionStrategy):
            self.strategies.parent_selection = strategy(
                population=self, random=self.params.random)
        else:
            raise ValueError("Invalid parent selection strategy")

    # endregion

    # region Genetic Pool Manipulation Methods

    def set_gene_pool(self, new_gene_pool):
        self.params.random = cast(random.Random, self.params.random)
        if not isinstance(new_gene_pool, GenePool):
            raise TypeError("Gene pool must be a GenePoolStrategy class.")
        if self.params.seed is not None:
            new_gene_pool.random = self.params.random
        self.strategies.gene_pool = new_gene_pool
        self.strategies.fitness.gene_pool = new_gene_pool

    def add_to_gene_pool(self, gene):
        if gene is not type(self.strategies.gene_pool.genes[0]):
            raise TypeError("Gene must be of the same type as the gene pool.")
        self.strategies.gene_pool.genes.append(gene)

    def remove_from_gene_pool(self, gene):
        if gene is not type(self.strategies.gene_pool.genes[0]):
            raise TypeError("Gene must be of the same type as the gene pool.")
        if gene in self.strategies.gene_pool.genes:
            self.strategies.gene_pool.genes.remove(gene)
        else:
            raise ValueError("Gene not in gene pool.")

    def clear_gene_pool(self):
        self.strategies.gene_pool.genes = []

    # endregion

    # region Abstract Methods
    @abstractmethod
    def _generate_initial_population(self):
        pass

    @abstractmethod
    def _compute_fitness(self):
        pass

    @abstractmethod
    def _evolve_population(self):
        pass

    @abstractmethod
    def _generate_random_individual(self) -> Individual:
        pass

    @abstractmethod
    def _run(self, iterations: int):
        pass

    def _mutation_rate_change(self, iterations: int):
        raise NotImplementedError(
            "Subclasses must implement mutation_rate_change if mutation_change is not None")

    # region Private Methods
    @staticmethod
    def _check_fitness(individual: Individual):
        """Static Method to check if the fitness of an individual has been calculated. If not, raise
        an error."""
        if individual.fitness is None:
            raise ValueError(
                "Fitness not calculated for one or more individuals")
        return individual.fitness

    # endregion

    # region Population Manipulation Methods

    def append(self, individual: Individual):
        if not isinstance(individual, Individual):
            raise TypeError(
                f"Unsupported operand type for append: 'Individual' and "
                f"'{(type(individual).__name__)}'")
        if self.state is None:
            raise ValueError("Population not initialized")

        self.state.individuals.append(individual)

    def extend(self, individuals: list[Individual]):
        if not all(isinstance(i, Individual) for i in individuals):
            raise TypeError(
                "All elements of 'individuals' must be of type 'Individual'")
        if self.state is None:
            raise ValueError("Population not initialized")

        self.state.individuals.extend(individuals)

    def copy(self):
        new_population = copy.copy(self)
        return new_population

    def deepcopy(self, memo=None):
        new_population = copy.deepcopy(self, memo)
        new_population.refresh_id()
        return new_population

    def sort(self, **kwargs):
        kwargs.setdefault('reverse', True)
        if self.state is None:
            raise ValueError("Population not initialized")

        self.state.individuals.sort(**kwargs)

    def clear(self):
        if self.state is None:
            raise ValueError("Population not initialized")
        self.state.individuals.clear()

    # endregion

    # region Public Methods

    def run(self, iterations) -> "AbstractPopulation":
        self.logger = cast(CustomLogger, self.logger)
        self.logger.info("Starting evolution for Population: %d", self.id)
        self.logger.benchmark(
            "Running evolution for Population ID: %d for %d iterations", self.id, iterations)
        run_start = time.perf_counter()
        if self.state is None:
            raise ValueError("Population not initialized")
        for _ in range(iterations):
            print(
                f"Starting iteration {self.state.iteration} for Population ID: {self.id}")
            iteration_start = time.perf_counter()
            self.logger.total_benchmark(
                "Starting iteration %d for Population ID: %d", self.state.iteration, self.id)

            self._run(iterations)

            iteration_end = time.perf_counter()
            self.logger.total_benchmark(
                "Iteration complete for Population ID: %d in %f seconds", self.id,
                iteration_end - iteration_start)

        self.logger.benchmark("Evolution complete for Population ID: %d in %f seconds",
                              self.id, time.perf_counter() - run_start)
        return self
    # endregion


if __name__ == "__main__":
    print("This Module is not meant to be run directly.")
