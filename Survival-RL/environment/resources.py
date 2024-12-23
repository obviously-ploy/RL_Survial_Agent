class Berry:
    def __init__(self, position):
        self.position = position
        self.type = 'berry'
        self.symbol = "B"

class Wood:
    def __init__(self, position):
        self.position = position
        self.type = 'wood'
        self.symbol = "W"

class Animal:
    def __init__(self, position):
        self.position = position
        self.is_hunted = False
        self.symbol = "A"
    
    
    def move(self):
        # Animal's movement behavior (random or stationary)
        pass
