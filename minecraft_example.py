from gym import envs
from gym.envs.cmu_biorobotics import minecraft

import numpy as np


def main():
    # Initialize environment
    env = envs.make('minecraft-v0')
    env.reset()

    # Get action space
    #action_space = env.action_space

    # Plan some random stuff to do
    actions=[       # agent_id, action_num
        (1, 0),
        (2, 4),
        (2, 3),
        (2, 0),
        (1, 3),
        (2, 2),
        (2, 5), # gets reward of 1
    ]

    # Do some random stuff to the env for 2 times
    for i in range(2):
        for a in actions:
            state1, reward, done, info = env.step(a)

            if i < 1:
                # output results    
                print('reward: ', reward)
                print('facing:', state1['facing'])
                print('position:', state1['position'])
                print(state1['view'][::-1,1,:])

        # Reset env
        env.reset()

if __name__ == '__main__':
    main()
