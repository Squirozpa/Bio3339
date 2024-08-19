from __future__ import annotations
from dataclasses import dataclass, field
from random import Random
from typing import TYPE_CHECKING
from typing import Optional

if TYPE_CHECKING:
    from genetic_algorithm.abstract_classes.abstract_strategies import (
        CrossoverStrategy, FitnessStrategy, GenePool, ParentSelectionStrategy,
        SurvivorSelectionStrategy)
    from genetic_algorithm.population_classes.individual import Individual


class StrategyConfig():
    def __init__(self,
                 gene_pool: GenePool,
                 fitness: FitnessStrategy,
                 crossover: CrossoverStrategy,
                 survivor_selection: SurvivorSelectionStrategy,
                 parent_selection: ParentSelectionStrategy,
                 ):
        self.gene_pool = gene_pool
        self.fitness = fitness
        self.crossover = crossover
        self.survivor_selection = survivor_selection
        self.parent_selection = parent_selection


@dataclass
class PopulationParams():
    mutation_rate: float = 0.01
    mutation_change: Optional[bool] = None
    population_size: int = 100
    immigration_rate: Optional[float] = None
    num_survivors: int = 2
    seed: None | int | str | float = None
    random: Random = field(default_factory=Random)


@dataclass
class PopulationState:
    individuals: list[Individual] = field(default_factory=list)
    _best_individual: Optional[Individual] = None
    _total_fitness: float = 0.0
    _best_individuals: list[Individual] = field(default_factory=list)
    _total_fitnesses: list[float] = field(default_factory=list)
    _iteration: int = field(default=0, init=False)

    def __iter__(self):
        return iter(self.individuals)

    @property
    def iteration(self):
        return self._iteration

    def add_iteration(self):
        self._iteration += 1

    @property
    def best_individual(self):
        return self._best_individual

    def set_best_individual(self):
        self._best_individual = self.individuals[0]

    @property
    def total_fitness(self):
        return self._total_fitness

    def set_total_fitness(self):
        self._total_fitness = sum(
            individual.fitness for individual in self.individuals)

    @property
    def total_fitnesses(self):
        return self._total_fitnesses

    def set_total_fitnesses(self):
        self._total_fitnesses.append(self._total_fitness)

    def refresh_state(self):
        self._best_individual = self.individuals[0]
        self._total_fitness = sum(
            individual.fitness for individual in self.individuals)
        self._total_fitnesses.append(self._total_fitness)
        self._best_individuals.append(self._best_individual)
