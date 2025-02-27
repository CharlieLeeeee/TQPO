
config0 = {
    'placements_extents': [-1.5, -1.5, 1.5, 1.5],
    'robot_base': 'xmls/point.xml',
    'task': 'button',
    'observe_goal_lidar': True,
    'lidar_max_dist': 3,
    'lidar_num_bins': 16,

    'constrain_hazards': True,
    'constrain_buttons': True,
    'observe_hazards': True,
    'observe_buttons': True,

    'buttons_num': 2,
    'buttons_size': 0.1,
    'buttons_keepout': 0.2,
    'buttons_locations': [(-1, -1), (1, 1)],
    'hazards_num': 3,
    'hazards_size': 0.3,
    'hazards_keepout': 0.305,
    'hazards_locations': [(0, 0), (-1, 1), (0.5, -0.5)],
}