import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

close = 0
move = 1
open_up = 2
open_down = 3

class ElevatorsEnv(gym.Env):
    """elevators environment
    Gym implementation of the RDDLsim POMDP problem found here:
    https://github.com/ssanner/rddlsim/blob/master/files/final_comp/rddl/elevators_pomdp.rddl
    Author: Layton Hayes (laytonc.hayes [at] gmail.com)

    Author of the problem in RDDLsim: Author: Tom Walsh (thomasjwalsh [at] gmail.com)
    // Modified for competition and translation purposes by Scott Sanner.
    //
    // The "elevators" domain has a number of elevators delivering passengers
    // to either the top or the bottom floor (the only allowable destinations).
    // Potential passengers arrive at a floor based on Bernoulli draws
    // with a potentially different arrival probability for each floor.
    //
    // The elevator can move in its current direction if the doors are closed,
    // can remain stationary (noop), or can open its door while indicating
    // the direction that it will go in next (this allows potential passengers
    // to determine whether to board or not).  Note that the elevator can only
    // change direction by opening its door while indicating the opposite
    // direction.
    //
    // A passable plan in this domain is to pick up a passenger every time
    // they appear and take them to their destination.  A better plan includes
    // having the elevator "hover" near floors where passengers are likely to
    // arrive and coordinating multiple elevators for up and down passengers.
    //
    // This domain was designed to support extension to multiple elevators
    // and may be used in either single or multi-elevator mode.
    """
    def __init__(self, max_steps=10, arrival_rate=0.14635538):
        # params:
        self.penalty_right_dir = 0.75
        self.penalty_wrong_dir = 3.0
        self.arrival_rate = arrival_rate
        self.max_steps = max_steps
        self.t = 0
        self.seed()
        # fluents:
        self.elevator_floor = 0
        self.direction = 1 # up
        self.closed = 1
        self.person_waiting_up = 0
        self.person_waiting_down = 0
        self.person_in_elevator_going_down = 0
        self.person_in_elevator_going_up = 0
        # spaces
        self.action_space = spaces.Discrete(4)
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
        ))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.t += 1
        done = False if self.t < self.max_steps else True

        if self.np_random.rand() < self.arrival_rate:
            self.person_waiting_up = 1

        if self.np_random.rand() < self.arrival_rate:
            self.person_waiting_down == 1

        if self.elevator_floor == 1 and self.direction == 1 and not self.closed \
                and self.person_waiting_up:
            self.person_in_elevator_going_up = 1
            self.person_waiting_up = 0

        if self.elevator_floor == 1 and self.direction == -0 and not self.closed \
                and self.person_waiting_down:
            self.person_in_elevator_going_up = 1
            self.person_waiting_down = 0

        if self.elevator_floor == 2 and not self.closed:
            self.person_in_elevator_going_up = 0

        if self.elevator_floor == 0 and not self.closed:
            self.person_in_elevator_going_down = 0

        if action == close:
            self.closed = 1
        elif action == open_up:
            self.closed = 0
            self.direction = 1 # up
        elif action == open_down:
            self.closed = 1
            self.direction = 0 # down
        elif action == move and self.closed:
            if self.direction == 1 and self.elevator_floor < 2:
                self.elevator_floor += 1
            elif self.direction == 0 and self.elevator_floor > 0:
                self.elevator_floor -= 1


        reward = (
            -self.penalty_right_dir * (self.person_in_elevator_going_up and self.direction == 1) \
            -self.penalty_right_dir * (self.person_in_elevator_going_down and self.direction == 0) \
            -self.penalty_wrong_dir * (self.person_in_elevator_going_up and self.direction == 0) \
            -self.penalty_wrong_dir * (self.person_in_elevator_going_down and self.direction == 1) \
            -self.person_waiting_up - self.person_waiting_down
        )
        self.penalty_right_dir = 0.75;
        self.penalty_wrong_dir = 3.0;

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = (
            self.closed,
            self.person_in_elevator_going_down,
            1 if self.elevator_floor == 0 else 0,
            1 if self.elevator_floor == 1 else 0,
            1 if self.elevator_floor == 2 else 0,
            0,  # p waiting up at f0
            self.person_waiting_up,  # p waiting up at f1
            0,  # p waiting up at f2
            self.direction,  # going up
            0,  # p waiting down at f0
            self.person_waiting_down,  #  p waiting down at f1
            0,  # p waiting down at f2
            self.person_in_elevator_going_up
            # self.direction,
            # 0,
            # max(self.person_waiting_up, self.person_waiting_down),
            # 0,
            # self.person_in_elevator_going_up,
        )
        return obs

    def reset(self):
        self.elevator_floor = 0
        self.direction = 1 # up
        self.closed = 1
        self.person_waiting_up = 0
        self.person_waiting_down = 0
        self.person_in_elevator_going_down = 0
        self.person_in_elevator_going_up = 0
        return self._get_obs()

    def render(self):
        grid = [['-','-','G'],self.road,['-','-','-']]
        if self.robot_alive:
            grid[self.robotY][self.robotX] = 'R'
        for i in range(len(grid)):
            row = ''
            for j in range(len(grid[0])):
                if str(grid[i][j]) == '1': row += 'O'
                elif str(grid[i][j]) == '0': row += '-'
                else: row += grid[i][j]
            print(row)
