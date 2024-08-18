"""
Survivor Selection Strategies
"""

# Standard Library Imports
from __future__ import annotations
from typing import List, TYPE_CHECKING
# Local Library Imports
from genetic_algorithm.abstract_classes.abstract_strategies import SurvivorSelectionStrategy
if TYPE_CHECKING:
    from genetic_algorithm.population_classes.individual import Individual
####################################################################################################


class ElitismSurvivorSelection(SurvivorSelectionStrategy):
    """Class for Eltitism Survivor Selection Strategy."""

    def select_survivors(self, num_survivors=2) -> List[Individual]:
        """Select the best individuals from the population."""
        if self.population is None or self.population.state is None:
            raise ValueError(
                "Population not initialized or initialized incorrectly")
        self.population.state.individuals.sort(reverse=True)
        return self.population.state.individuals[:num_survivors]


class AgeBasedSurvivorSelection(SurvivorSelectionStrategy):
    """Class for Age-Based Survivor Selection Strategy."""

    def select_survivors(self, num_survivors=2) -> List[Individual]:
        """Select the youngest individuals from the population."""
        if self.population is None or self.population.state is None:
            raise ValueError(
                "Population not initialized or initialized incorrectly")
        self.population.state.individuals.sort(
            key=lambda x: (x.iteration, -x.fitness))
        return self.population.state.individuals[:num_survivors]


class FitnessProportionateSurvivorSelection(SurvivorSelectionStrategy):
    """Class for FitnessPropotionate Survivor Selection Strategy."""

    def select_survivors(self, num_survivors=2) -> List[Individual]:
        """Select survivors based on fitness chance."""
        if self.population is None or self.population.state is None:
            raise ValueError(
                "Population not initialized or initialized incorrectly")
        survivors = []
        for _ in range(num_survivors):
            r = self.random.uniform(0, self.population.state.total_fitness)
            s = 0
            for individual in self.population.state.individuals:
                r += individual.fitness
                if s > r:
                    survivors.append(individual)
                    break
        return survivors
