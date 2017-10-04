from gym import envs
from gym.envs.cmu_biorobotics import minecraft

import numpy as np


def main():
    # Create initial and objective world
    world0 = np.zeros([20, 10, 20])

    world = world0.copy()
    world[10,0,10] = -1
    world[10,0,11] = -1
    world[10,0,12] = 1
    world[11,0,10] = 2

    world_plan = world0.copy()
    world_plan[10,0,9] = 1
    world_plan[10,0,13] = 1

    # Initialize environment
    env = envs.make('minecraft-v0')
    #env.world_init = world
    #env.state_obj = world_plan
    #print('FUCK')
    #env.reset()
    env._reset(world_init=world, world_plan=world_plan)

    # Get action space
    action_space = env.action_space

    # Plan some random stuff to do
    actions=[       # agent_id, action_num, display_result
        (1, 0, 1),
        (2, 4, 0),
        (2, 3, 1),
        (2, 0, 0),
        (1, 2, 0),
        (2, 2, 0),
        (2, 5, 1),
    ]

    # Do some random stuff to the env for 2 times
    for i in range(2):
        for a in actions:
            state1, reward, done, info = env.step(action_space[a[1]], a[0])

            if a[3]:
                state0 = env.observe(a[0])

                # output results    

        # Reset env
        env.reset()

if __name__ == '__main__':
    main()
