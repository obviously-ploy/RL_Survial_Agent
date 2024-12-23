import numpy as np
import random
from environment.resources import Berry, Wood, Animal
import gymnasium as gym
from config import ENV_CONFIG


class GridWorld(gym.Env):
    def __init__(self, config, map_layout):
        # movement, with cutting, creating, eating, and placing
        # 0-3 -> movement
        # 4 -> pick up berry
        # 5 -> pick up wood
        # 6 -> make boat
        # 7 -> make sword
        # 8 -> hunt animal
        # will add more later if less lazy (ig cutting works with hunting (check inv))
        # just punish if it makes an invalid move
        self._n_actions = 9
        self._directions = [np.array((-1, 0)), 
                    np.array((1, 0)), 
                    np.array((0, -1)), 
                    np.array((0, 1))]
        # self._directions = [[-1, 0], 
        #             [1, 0], 
        #             [0, -1], 
        #             [0, 1]]

        # will be [x, y]
        self._current_cell = None

        self.grid_size = config['grid_size']

        self.berry_spawn_rate = config['berry_spawn_rate']
        self.wood_spawn_rate = config['wood_spawn_rate']
        self.deer_spawn_rate = config['animal_spawn_rate']
        self.hunger_decay_rate = config['hunger_decay_rate']

        self.config = config
        self.map_layout = map_layout
        self.agents = []
        self.resources = {}
        # self.resources = []
        # self.setup_resources()
    
    def reset(self):
        row, col = self.grid_size
        self.visited_cells = np.zeros([row, col])
        self.grid = self._initialize_grid_from_string(self.config, self.map_layout)
        self.setup_resources()

        self.hunger = 100
        self.wood = 0
        self.boat = 0
        self.sword = 0

        # start on bottom right corner
        x, y = self.grid_size
        x -= 1
        y -= 1
        self._current_cell = np.array((x, y))
        state = (self._current_cell[0], self._current_cell[1], self.wood, self.boat, self.sword) # total states is 10 * 10 * 8???
        return state, {}

    def _initialize_grid_from_string(self, config, map_string):
        map_lines = [line.strip() for line in config[map_string].strip().split('\n')]
        
        grid = np.array([list(line) for line in map_lines], dtype=object)

        return grid

    def setup_resources(self):
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if self.grid[y, x] == 'L':
                    if random.random() < self.berry_spawn_rate:
                        self.resources[(x, y)] = Berry((x, y))
                    elif random.random() < self.wood_spawn_rate:
                        self.resources[(x, y)] = Wood((x, y))
                    elif random.random() < self.deer_spawn_rate:
                        self.resources[(x, y)] = Animal((x, y))
    
    def render(self):
        self.reset()
        render_txt = ''
        for y in range(self.grid_size[0]):
            row = ''
            for x in range(self.grid_size[1]):
                resource = self.get_resource_at(x, y)
                if resource:
                    row += resource.symbol
                else:
                    row += '.' if self.grid[y, x] == 'L' else '~'
            render_txt += row +  "\n"
        return render_txt

    def get_resource_at(self, x, y):
        if (x, y) in self.resources:
            return self.resources[(x, y)]
        return None

    def get_block_type_at(self, x, y):
        return self.grid[y, x]
    
    def step(self, action):
        self.hunger -= self.hunger_decay_rate
        terminated = self.hunger <= 0

        reward = 0
        if self.hunger < 0:
            reward += ENV_CONFIG['agent_rewards']['dies'] 

        # move agent
        if action < 4:
            next_blockX, next_blockY = self._current_cell + self._directions[action]

            # agent only moves if still within grid
            edgeX, edgeY = self.grid_size

            # if on water, try to traverse
            useBoat = False
            if 0 <= next_blockX < edgeX and 0 <= next_blockY < edgeY and self.grid[next_blockX, next_blockY] == "W" and self.boat > 0:

                useBoat = True
                # go as far as you can within water
                while 0 <= next_blockX < edgeX and 0 <= next_blockY <  edgeY and self.grid[next_blockX, next_blockY] == "W":
                    next_blockX, next_blockY = (next_blockX, next_blockY) + self._directions[action]

            # if still in bounds, or out of water, change position
            if 0 <= next_blockX < edgeX and 0 <= next_blockY < edgeY and self.grid[next_blockX, next_blockY] != "W":
                self._current_cell = next_blockX, next_blockY
                if useBoat:
                    self.boat -= 1
                    reward += 50

                # reward if visiting new cell
                if not self.visited_cells[next_blockX, next_blockY]:
                    reward += ENV_CONFIG['agent_rewards']['explore_new_cell']
                    self.visited_cells[next_blockX, next_blockY] = 1
                
        elif action == 4:
            x, y = self._current_cell
            res = self.get_resource_at(x, y)
            if res and res.symbol == "B":
                self.hunger += 10
                reward += ENV_CONFIG['agent_rewards']['pick_up_berry']
                del self.resources[(x, y)]
            else:
                reward -= 20 # arbitrary reward for invalid action
        elif action == 5:
            x, y = self._current_cell
            res = self.get_resource_at(x, y)
            if res and res.symbol == "W":
                self.wood += 1
                reward += ENV_CONFIG['agent_rewards']['pick_up_wood']
                del self.resources[(x, y)]
            else:
                reward -= 20 # arbitrary reward for invalid action
        elif action == 6:
            if self.wood > 0:
                self.wood -= 1
                self.boat += 1
                reward += ENV_CONFIG['agent_rewards']['build_boat']
            else: reward -= 20
        elif action == 7:
            if self.wood > 0:
                self.wood -= 1
                self.sword += 1
                reward += ENV_CONFIG['agent_rewards']['build_sword']
            else: reward -= 20
        elif action == 8:
            if self.sword > 0:
                x, y = self._current_cell
                res = self.get_resource_at(x, y)
                if res and res.symbol == "A":
                    self.hunger += 20
                    del self.resources[(x, y)]
                    self.sword -= 1
                    reward += ENV_CONFIG['agent_rewards']['hunt_animal']
                else: reward -= 20
            else: reward -= 20

        state = (self._current_cell[0], self._current_cell[1], self.wood, self.boat, self.sword)
        return state, reward, terminated, False, {}
    
    @property
    def n_actions(self):
        return self._n_actions
    
    def remove_resource_at(self, x, y):
        for resource in self.resources:
            if resource.position == (x, y):
                self.resources.remove(resource)
                self.grid[y, x] = "L"
                return True 
        return False
