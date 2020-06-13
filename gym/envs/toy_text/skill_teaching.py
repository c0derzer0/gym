import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

giveHint_0 = 0
giveHint_1 = 1
askProb_0 = 2
askProb_1 = 3
noop = 4

class SkillTeachingEnv(gym.Env):
    def __init__(self, input_rate=0.2, max_steps=10):
        self._max_episode_steps = max_steps
        self.t = 0
        self.seed()
        self.FALSE_POS_0 = 0.05016588
        self.FALSE_POS_1 = 0.0917274

        self.PROB_ALL_PRE_0 = 0.7199796
        self.PROB_ALL_PRE_MED_0 = 0.73191
        self.PROB_HIGH_0 = 0.85399747
        self.SKILL_WEIGHT_0 = 1.1563843

        self.PROB_ALL_PRE_1 = 0.6515699
        self.PROB_ALL_PRE_MED_1 = 0.74608207
        self.PROB_HIGH_1 = 0.9513694
        self.SKILL_WEIGHT_1 = 1.0460582

        self.updateTurn_0 = 0
        self.updateTurn_1 = 0

        self.fpos_0 = 0
        self.fpos_1 = 0

        self.answeredRight_0 = 0
        self.answeredRight_1 = 0

        self.proficiencyMed_0 = 0
        self.proficiencyHigh_0 = 0
        self.proficiencyMed_1 = 0
        self.proficiencyHigh_1 = 0

        self.hintedRight_0 = 0
        self.hintedRight_1 = 0
        self.hintDelay_0 = 0
        self.hintDelay_1 = 0

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Tuple((
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2),
                spaces.Discrete(2)
        ))
        self.prev_obs = self._get_obs()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.t += 1
        done = False if self.t < self._max_episode_steps else True

        reward = (self.SKILL_WEIGHT_0 * self.proficiencyHigh_0) \
            - (self.SKILL_WEIGHT_0 * (not self.proficiencyMed_0)) \
            + (self.SKILL_WEIGHT_1 * self.proficiencyHigh_1) \
            - (self.SKILL_WEIGHT_1 * (not self.proficiencyMed_1))

        if not self.updateTurn_0 and not self.updateTurn_1:
            if action == askProb_0:
                if self.proficiencyHigh_0:
                    if self.np_random.rand() < self.PROB_HIGH_0:
                        self.answeredRight_0 = 1
                    else: self.answeredRight_0 = 0
                elif self.proficiencyMed_0:
                    if self.np_random.rand() < self.PROB_ALL_PRE_MED_0:
                        self.answeredRight_0 = 1
                    else: self.answeredRight_0 = 0
                else:
                    if self.np_random.rand() < self.PROB_ALL_PRE_0:
                        self.answeredRight_0 = 1
                    else: self.answeredRight_0 = 0
            elif action == askProb_1:
                if self.proficiencyHigh_1:
                    if self.np_random.rand() < self.PROB_HIGH_1:
                        self.answeredRight_1 = 1
                    else: self.answeredRight_1 = 0
                elif self.proficiencyMed_1:
                    if self.np_random.rand() < self.PROB_ALL_PRE_MED_1:
                        self.answeredRight_1 = 1
                    else: self.answeredRight_1 = 0
                else:
                    if self.np_random.rand() < self.PROB_ALL_PRE_1:
                        self.answeredRight_1 = 1
                    else: self.answeredRight_1 = 0
            if action == giveHint_0:
                self.hintedRight_0 = 1
                self.hintDelayVar_0 = 1
            else:
                self.hintedRight_0 = 0
                self.hintDelayVar_0 = 0
            if action == giveHint_1:
                self.hintedRight_1 = 1
                self.hintDelayVar_1 = 1
            else:
                self.hintedRight_1 = 0
                self.hintDelayVar_1 = 0

        if (not self.updateTurn_0 and self.proficiencyMed_0):
            self.proficiencyMed_0 = 1
        elif (self.updateTurn_0 and self.hintedRight_0):
            self.proficiencyMed_0 = 1
        elif (self.updateTurn_0 and self.answeredRight_0 and not self.proficiencyHigh_0 and not self.fpos_0):
            self.proficiencyMed_0 = 1
        elif (self.proficiencyMed_0 and self.updateTurn_0 and self.answeredRight_0):
            self.proficiencyMed_0 = 1
        elif (self.proficiencyHigh_0):
            self.proficiencyMed_0 = 1
        elif (self.proficiencyMed_0 and self.updateTurn_0 and self.hintDelayVar_0):
            self.proficiencyMed_0 = 1
        else:
            self.proficiencyMed_0 = 0

        if (not self.updateTurn_1 and self.proficiencyMed_1):
            self.proficiencyMed_1 = 1
        elif (self.updateTurn_1 and self.hintedRight_1):
            self.proficiencyMed_1 = 1
        elif (self.updateTurn_1 and self.answeredRight_1 and not self.proficiencyHigh_1 and not self.fpos_1):
            self.proficiencyMed_1 = 1
        elif (self.proficiencyMed_1 and self.updateTurn_1 and self.answeredRight_1):
            self.proficiencyMed_1 = 1
        elif (self.proficiencyHigh_1):
            self.proficiencyMed_1 = 1
        elif (self.proficiencyMed_1 and self.updateTurn_1 and self.hintDelayVar_1):
            self.proficiencyMed_1 = 1
        else:
            self.proficiencyMed_1 = 0

        if (not self.updateTurn_0 and self.proficiencyHigh_0):
            self.proficiencyHigh_0 = 1
        elif (self.proficiencyMed_0 and self.updateTurn_0 and self.answeredRight_0 and not self.fpos_0):
            self.proficiencyHigh_0 = 1
        elif (self.proficiencyHigh_0 and self.updateTurn_0 and self.answeredRight_0):
            self.proficiencyHigh_0 = 1
        elif (self.proficiencyHigh_0 and self.updateTurn_0 and (self.hintDelayVar_0 or self.answeredRight_0)):
            self.proficiencyHigh_0 = 1
        else:
            self.proficiencyHigh_0 = 0

        if (not self.updateTurn_1 and self.proficiencyHigh_1):
            self.proficiencyHigh_1 = 1
        elif (self.proficiencyMed_1 and self.updateTurn_1 and self.answeredRight_1 and not self.fpos_1):
            self.proficiencyHigh_1 = 1
        elif (self.proficiencyHigh_1 and self.updateTurn_1 and self.answeredRight_1):
            self.proficiencyHigh_1 = 1
        elif (self.proficiencyHigh_1 and self.updateTurn_1 and (self.hintDelayVar_1 or self.answeredRight_1)):
            self.proficiencyHigh_1 = 1
        else:
            self.proficiencyHigh_1 = 0

        if (not self.updateTurn_0 and not self.updateTurn_0) and (action==askProb_0 or action==giveHint_0):
            self.updateTurn_0 = 1
        else:
            self.updateTurn_0 = 0

        if (not self.updateTurn_1 and not self.updateTurn_1) and (action==askProb_1 or action==giveHint_1):
            self.updateTurn_1 = 1
        else:
            self.updateTurn_1 = 0

        if (not self.updateTurn_0 and not self.updateTurn_0) and action==askProb_0 \
                and self.np_random.rand() < self.FALSE_POS_0:
            self.fpos_0 = 1
        else:
            self.fpos_0 = 0

        if (not self.updateTurn_1 and not self.updateTurn_1) and action==askProb_1 \
                and self.np_random.rand() < self.FALSE_POS_1:
            self.fpos_1 = 1
        else:
            self.fpos_1 = 0

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = (
            self.hintDelay_0,
            self.hintDelay_1,

            self.updateTurn_0,
            self.updateTurn_1,

            self.answeredRight_0,
            self.answeredRight_1,

            self.proficiencyMed_0,
            self.proficiencyHigh_0,

            self.proficiencyMed_1,
            self.proficiencyHigh_1,

            self.hintedRight_0,
            self.hintedRight_1,
        )
        return obs

    def reset(self):
        self.t = 0
        self.updateTurn_0 = 0
        self.updateTurn_1 = 0

        self.fpos_0 = 0
        self.fpos_1 = 0

        self.answeredRight_0 = 0
        self.answeredRight_1 = 0

        self.proficiencyMed_0 = 0
        self.proficiencyHigh_0 = 0
        self.proficiencyMed_1 = 0
        self.proficiencyHigh_1 = 0

        self.hintedRight_0 = 0
        self.hintedRight_1 = 0
        self.hintDelay_0 = 0
        self.hintDelay_1 = 0

        return self._get_obs()

    def render(self):
        pass
