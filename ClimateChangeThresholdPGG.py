import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt

from egttools.games import AbstractNPlayerGame
from egttools.analytical import PairwiseComparison
from egttools import sample_simplex, calculate_nb_states
from egttools.utils import calculate_stationary_distribution


class ClimateChangeThresholdPGG(AbstractNPlayerGame):
    def __init__(self,
                 group_size: int,  # N - Number of individuals that be sampled to play
                 population_size_Z: int,  # Total population - Will be divided 80-20% (Poor-Rich)
                 b_R: float,  # Endowment of the rich
                 b_P: float,  # Endowment of the poor
                 c: float,  # Fraction of endowment Cs give
                 r: float,  # Risk factor [0, 1]
                 h: float,  # Homophily [0, 1] - "Like imitates like" factor
                 M: float,  # Factor which will calculate success (M x c x avg(b) should be met)
                 beta: float  # Intensity of selection for imitating strategies
                 ):

        AbstractNPlayerGame.__init__(self, 4, group_size)  # (self, nb_strategies, group_size)

        self.nb_strategies_ = 4
        self.strategies = ["C_R", "D_R", "C_P", "D_P"]  # Cooperate/Defect_Rich/Poor

        self.group_size_ = group_size
        self.N = group_size  # These two^ are the same, just being more consistent with the paper

        assert b_R > b_P, "The Marxists are back! Rise of the proletariat!"
        self.b_R = b_R
        self.b_P = b_P

        self.c = c
        self.r = r
        self.h = h
        self.M = M
        self.beta = beta

        # Proportions of rich and poor
        self.rich_ratio = 0.2  # 20% rich
        self.poor_ratio = 0.8  # 80% poor

        self.nb_group_configurations_ = self.nb_group_configurations()
        self.calculate_payoffs()

    def play(self,
             group_composition: Union[List[int], np.ndarray],
             game_payoffs: np.ndarray):
        """
        Simulates one round of the game
        """
        game_payoffs[:] = 0.0

        # Calculate total contributions
        rich_cooperators = group_composition[0]
        poor_cooperators = group_composition[1]
        rich_defectors = group_composition[2]
        poor_defectors = group_composition[3]

        total_contributions = (rich_cooperators * self.b_R * self.c +
                               poor_cooperators * self.b_P * self.c)
        group_size = group_composition.sum()

        # Calculate average endowment (b) for the group
        total_endowment = (rich_cooperators + rich_defectors) * self.b_R + \
                          (poor_cooperators + poor_defectors) * self.b_P
        average_endowment = total_endowment / group_size

        # Check if the threshold is met
        threshold = self.M * self.c * average_endowment
        success = total_contributions >= threshold
        disaster = np.random.rand() < self.r if not success else False

        # Assign payoffs based on contributions and disaster
        for idx, count in enumerate(group_composition):
            if count > 0:
                if idx == 0:  # Rich cooperators
                    loss = self.b_R * (1 - self.c) if disaster else 0
                    payoff = self.b_R * self.c * success - loss
                elif idx == 1:  # Poor cooperators
                    loss = self.b_P * (1 - self.c) if disaster else 0
                    payoff = self.b_P * self.c * success - loss
                elif idx == 2:  # Rich defectors
                    loss = self.b_R if disaster else 0
                    payoff = self.b_R - loss
                elif idx == 3:  # Poor defectors
                    loss = self.b_P if disaster else 0
                    payoff = self.b_P - loss
                else:
                    raise ValueError(f"Unknown strategy index: {idx}")

                # Update payoffs for this strategy
                game_payoffs[idx] += payoff

    def calculate_payoffs(self):
        payoffs_container = np.zeros(self.nb_strategies_)
        for i in range(self.nb_group_configurations_):
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            print(group_composition)
            self.play(group_composition, payoffs_container)
            for strategy_idx, payoff in enumerate(payoffs_container):
                self.update_payoff(strategy_idx, i, payoff)
            print(payoffs_container)

            payoffs_container[:] = 0.0