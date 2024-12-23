class Agent:
    def __init__(self, initial_state, config, env):
        self.position = initial_state['position']
        self.inventory = config['initial_inventory']
        self.hunger_level = 100
        self.env = env

    def move(self, direction):
        new_x, new_y = self.position
        if direction == 'up':
            new_y -= 1
        elif direction == 'down':
            new_y += 1
        elif direction == 'left':
            new_x -= 1
        elif direction == 'right':
            new_x += 1
        
        
        if self.env.get_block_type_at(new_x, new_y) == 'W':
            self._move_water(direction)
        else:
            self.position = (new_x, new_y)
        

    def pick_up(self, resource, x, y):
        if resource == "wood":
            self.inventory[resource] += 1

        if resource == 'berry':
            self.hunger_level = min(100, self.hunger_level + 5)
            return f"Ate a berry. Hunger level is now {self.hunger_level}."
        self.env.remove_resource_at(x, y)

    def build(self, item):
        if item == 'boat':
            if self.inventory.get('wood', 0) >= 3:
                self.inventory['boat'] += 1
                self.inventory['wood'] -= 3
                return "Built a boat!"
            else:
                return "Not enough wood to build a boat."
        
        elif item == 'sword':
            if self.inventory.get('wood', 0) >= 2:
                self.inventory['sword'] += 1
                self.inventory['wood'] -= 2
                return "Built a sword!"
            else:
                return("Not enough resources to build a sword.")
        else:
            return(f"Cannot build {item}. Only 'boat' or 'sword' are buildable.")

    def hunt(self, x, y):
        if self.inventory['sword'] == 0:
           return ("You need a sword to hunt!")
          
        self.env.remove_resource_at(x, y)
        self.hunger_level = min(100, self.hunger_level + 20)
        self.inventory['sword'] -= 1;
        return(f"Hunger level after hunting: {self.hunger_level}")

    def _move_water(self, direction):
        x, y = self.position

        while True:
            if direction == 'up':
                y += 1
            elif direction == 'down':
                y -= 1
            elif direction == 'left':
                x -= 1
            elif direction == 'right':
                x += 1

            block_type = self.env.get_block_type_at(x, y)
            
            if block_type == 'L':
                self.inventory['boat'] -= 1;
                self.position = (x, y)
                break
            
            if block_type == 'W':
                continue
            
            