ENV_CONFIG = {
    'grid_size': (10, 10),          # Dimensions of the grid
    'berry_spawn_rate': 0.1,        # Chance of berry spawning on each tile
    'wood_spawn_rate': 0.25,        # Chance of wood appearing on forest tiles
    'animal_spawn_rate': 0.05,      # Chance of deer spawning on forest tiles
    'hunger_decay_rate': 1,         # Hunger level decay per time step
    # Agent-related configurations
    'agent_params': {
        'learning_rate': 0.1,
        'discount_factor': 0.99,
        'epsilon': 0.1,
    },
    "agent_rewards": {
        "pick_up_berry": 10,
        "pick_up_wood": 10,
        "build_boat": 20,
        "build_sword": 20,
        "hunt_animal": 40, 
        "dies": -50,
        "explore_new_cell": 1,
    },
    'initial_inventory':{
        'sword': 0,
        'boat': 2,
        'wood': 0
    },
    'all-land-map': """
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    LLLLLLLLLL
                    """,
    'river-map':
                """
                LLLLWWLLLL
                LLLLWWLLLL
                LLLLWLLLLL
                LLLLWLLLLL
                LLLLWWLLLL
                LLLLWWLLLL
                LLLLWLLLLL
                LLLLWLLLLL
                LLLLWWLLLL
                LLLLWWLLLL
                """,
    'lakes-map':
                """
                LLLLWWLLLL
                LLLLWWLLLL
                LLLLWLLWWL
                LLLLWLLLWL
                LWWLWWLLWL
                LLWLWWLWLL
                LLLLWLLLLL
                LLWLWLLLLL
                LLLLWWLLLL
                LLLLWWLLLL
                """,
    'island-map':
                """
                LLLWWLLLLL
                LLLWWLLLLL
                LLWWWLWWLL
                LLWWLLWWLL
                LLWLLLWWLL
                LLLWWWWWLL
                LLLWWWLLLL
                LLLWWWLLLL
                LLLWWWLLLL
                LLLWWWLLLL
                """
}
