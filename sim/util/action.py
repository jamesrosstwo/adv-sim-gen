def steering(action):
    return action[0, 0]

def gas(action):
    return action[0, 1]


def action_difference(action_a, action_b):
    return abs(steering(action_a) - steering(action_b))