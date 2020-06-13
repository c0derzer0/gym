import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

east = 0
north = 1
south = 2
west = 3

class CrossingTrafficEnv(gym.Env):
    """crossing_traffic environment
    Gym implementation of the RDDLsim POMDP problem found here:
    https://github.com/ssanner/rddlsim/blob/master/files/final_comp/rddl/crossing_traffic_pomdp.rddl
    Author: Layton Hayes (laytonc.hayes [at] gmail.com)

    Author of the problem in RDDLsim: Sungwook Yoon (sungwook.yoon [at] gmail.com)
    // Modified for competition and translation purposes by Scott Sanner.
    //
    // In a grid, a robot (R) must get to a goal (G) and avoid obstacles (O)
    // arriving randomly and moving left.  If an obstacle overlaps with the
    // robot, the robot disappears and can no longer move around.  The robot
    // can "duck" underneath a car by deliberately moving right/east when
    // a car is to the right of it (this can make the solution interesting...
    // the robot should start at the left side of the screen then).  The robot
    // receives -1 for every time step it has not reached the goal.  The goal
    // state is absorbing with 0 reward.
    //
    // ****************
    // *            R *
    // *  <-O <-O <-O *
    // *    <-O   <-O *
    // * <-O    <-O   *
    // *     <-O  <-O *
    // *            G *
    // ****************
    //
    // You can think of this as the RDDL version of Frogger:
    //
    //   http://en.wikipedia.org/wiki/Frogger
    //

    grid starts like this:

    --G
    ???
    --R

    where 'G' is the goal, 'R' is the robot agent, '-' is an empty space, and
    '?' can either be empty or contain a car ('O')
    """
    def __init__(self, input_rate=0.2, max_steps=10):
        self._max_episode_steps = 10
        self.input_rate = input_rate
        self.max_steps = max_steps
        self.t = 0
        self.seed()
        self.robot_alive = True
        self.robotX = 2
        self.robotY = 2
        self.at_goal = False
        self.car_coming = 1 if self.np_random.rand() < self.input_rate else 0
        self.road = [
                self.np_random.randint(2),
                self.np_random.randint(2),
                self.np_random.randint(2)
            ]
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
        done = False if self.t < self.max_steps else True
        self.road[0] = 1 if self.road[1] else 0
        self.road[1] = 1 if self.road[2] else 0
        self.road[2] = 1 if self.car_coming else 0
        self.car_coming = 1 if self.np_random.rand() < self.input_rate else 0
        reward = -1
        if self.at_goal:
            reward = 0
        elif self.robot_alive:
            if self.robotY == 1 and  self.road[self.robotX]:
                self.robot_alive = False
            elif action == east and  self.robotX < 2: self.robotX += 1
            elif action == west and  self.robotX > 0: self.robotX -= 1
            elif action == north and  self.robotY > 0: self.robotY -= 1
            elif action == south and  self.robotY < 2: self.robotY -= 1
            if self.robotY == 1 and  self.road[self.robotX]:
                self.robot_alive = False
            elif self.robotX == 2 and self.robotY == 0:
                self.at_goal = True
                reward = 0
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        obs = (
            # 0,# - observ: arrival-max-xpos[$y1] := false
            # self.car_coming,# - observ: arrival-max-xpos[$y2] := false
            # 0,# - observ: arrival-max-xpos[$y3] := false
            1 if self.robotX==0 and self.robotY==2 else 0,# - states: robot-at[$x1, $y1] := false
            1 if self.robotX==0 and self.robotY==1 else 0,# - states: robot-at[$x1, $y2] := false
            1 if self.robotX==0 and self.robotY==0 else 0,# - states: robot-at[$x1, $y3] := false
            1 if self.robotX==1 and self.robotY==2 else 0,# - states: robot-at[$x2, $y1] := false
            1 if self.robotX==1 and self.robotY==1 else 0,# - states: robot-at[$x2, $y2] := false
            1 if self.robotX==1 and self.robotY==0 else 0,# - states: robot-at[$x2, $y3] := false
            1 if self.robotX==2 and self.robotY==2 else 0,# - states: robot-at[$x3, $y1] := false
            1 if self.robotX==2 and self.robotY==1 else 0,# - states: robot-at[$x3, $y2] := true
            1 if self.robotX==2 and self.robotY==0 else 0,# - states: robot-at[$x3, $y3] := false

            0,                              # obstacle_at_x1y1
            1 if self.road[0] == 1 else 0,  # obstacle_at_x1y2
            0,                              # obstacle_at_x1y3
            0,                              # obstacle_at_x2y1
            1 if self.road[1] == 1 else 0,  # obstacle_at_x2y2
            0,                              # obstacle_at_x2y3
            0,                              # obstacle_at_x3y1
            1 if self.road[2] == 1 else 0,  # obstacle_at_x3y2
            0,                              # obstacle_at_x3y3
        )
        return obs

    def reset(self):
        self.t = 0
        self.robot_alive = True
        self.robotX = 2
        self.robotY = 2
        self.at_goal = False
        self.road = [
                1,
                0,
                1
            ]
        self.car_coming = 1 if self.np_random.rand() < self.input_rate else 0
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
