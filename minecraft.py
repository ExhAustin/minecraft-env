import gym
from gym import spaces

import numpy as np
from collections import OrderedDict

import sys

class Grid3DState(object):
    '''
    3D Grid State. 
    Implemented as a 3d numpy array.
            air = -1
            block = 0
            agent = positive integer (agent_id)
    '''
    def __init__(self, world0):
        self.state = world0.copy()
        self.shape = np.array(world.shape)

    # Get value of block
    def get(self, coord):
        try:
            return self.state[coord[0], coord[1], coord[2]]
        except IndexError:
            return -2

    # Set block to input value
    def set(self, coord, val):
        try:
            self.state[coord[0], coord[1], coord[2]] = val
            return True
        except IndexError:
            return False

    # Swap two blocks
    def swap(self, coord1, coord2):
        temp = self.get(coord1)
        self.set(coord1, self.get(coord2))
        self.set(coord2, temp)

    # Get observation
    def getObservation(self, coord, ob_range):
        '''
        Observation: Box centered around agent position
            (returns -2 for blocks outside world boundaries)

        Args:
            coord: Position of agent. Numpy array of length 3.
            ob_range: Vision range. Numpy array of length 3.

        observation.shape is (2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1)
        '''
        ob = -2*np.ones([2*ob_range[0]+1, 2*ob_range[1]+1, 2*ob_range[2]+1])

        # two corners of view in world coordinate
        c0 = coord - ob_range
        c1 = coord + ob_range

        # clip according to world boundaries
        c0_c = np.clip(c0, [0,0,0], self.shape) 
        c1_c = np.clip(c1, [0,0,0], self.shape) 

        # two corners of view in observation coordinates
        ob_c0 = -(c0 - c0_c)
        ob_c1 = (ob_range - 1) - (c1 - c1_c)

        # assign data from world to observation
        data = self.state[c0_c[0]:c1_c[0], c0_c[1]:c1_c[1], c0_c[2]:c1_c[2]]
        ob[ob_c0[0]:ob_c1[0], ob_c0[1]:ob_c1[1], ob_c0[2]:ob_c1[2]] = data

        return ob

    # Compare with a plan to determine job completion
    def done(self, state_obj):
        return np.min((self.state == -1) == state_obj)

class AgentState(object):
    '''
    Agent states. Keeps track of every agent in a database.
    Implemented as a 2d numpy array.
    Attributes:
        agent_id    integer (index of first dimension)
        facing      {0:+Z, 1:+X, 2:-Z, 3:-X}
        position    vector - (x, y, z)
    '''

    def __init__(self):
        self.agent_data = -np.ones([0,2])

    # Scan world for agents and load them into database
    def scanWorld(self, world):
        agent_list = []
        attr_list = []

        # list all agents
        for i in world.shape[0]:
            for j in world.shape[1]:
                for k in world.shape[2]:
                    val = world.get([i, j, k])
                    if val > 0:
                        assert val not in agent_list, 'ID conflict between agents'
                        assert type(val) is int, 'Non-integer agent ID'
                        agent_list.append(val)
                        attr_list.append(np.array([i, j, k, 0]))

        # load agents into database
        self.agent_data = -np.ones([max(agent_list), 2])
        for i in len(agent_list):
            self.agent_data[agent_list[i]] = attr_list[i]

    # Get attributes of an agent
    def get(self, agent_id):
        return self.agent_data[agent_id, :]

    # Set attributes of an agent
    def set(self, agent_id, new_state):
        self.agent_data[agent_id, :] = new_state

    # Return predicted new state after action
    def act(self, agent_id, action):
        current_state = self.agent_data[agent_id,:]
        new_state = current_state.copy()

        # Move
        if action in [0,1]:
            sign = (-1)**action
            move_vec = sign*self.facing2vec(self, current_state[3])
            new_state[0:3] += move_vec

        # Turn
        elif action in [2,3]:
            sign = (-1)**(action-2)
            new_state[3] += sign

        return new_state

    # Transform facing to x, z
    def facing2vec(self, fac):
        dx = (fac % 2)*(2 - fac)
        dy = 0
        dz = ((fac+1) % 2)*(1 - fac)
        return np.asarray([dx, dy, dz])

    # Check if agent ID is valid
    def validateID(self, agent_id):
        assert type(agent_id) is int, 'Non-integer agent ID'
        assert agent_id > 0, 'Non-positive agent ID'
        assert agent_id < self.agent_data.shape[0], 'Agent ID out of range'
        assert self.agent_data[agent_id,0] != -1, 'Non-existent agent ID'


class MinecraftEnv(gym.Env):
    '''
    3D Grid Environment
        Observation: An agent's position, facing and a limited view of the world.
            Position:   X, Y, Z  (+Y = up)
            Facing:     {0:+Z, 1:+X, 2:-Z, 3:-X}
            View:       A box centered around the agent
                block = -1
                air = 0
                agent = 1 (agent_id in id_visible mode, agent_id is a positive integer)
                out of world range = -2
        Action space: {0:MOVE_FORWARDS, 1:MOVE_BACKWARDS, 2:TURN_LEFT, 3:TURN_RIGHT, 4:PICK, 5:PLACE}
        Reward: -1 for each action, +1 for each block correctly placed
    '''
    metadata = {"render.modes": ["human", "ansi"]}

    # Initialize env
    def __init__(self, observation_range=1, observation_mode='default'):
        """
        Args:
            observation_range: Integer for cube. List of length 3 for box.
            observation_mode: {'default', 'id_visible'}
        """
        # Parse input parameters and check if valid
        #   observation_range
        if type(observation_range) is int:
            ob_range = observation_range*np.ones(3)
        else:
            assert len(observation_range) == 3, 'Wrong number of dimensions for \'observation_range\''
            ob_range = np.array(observation_range)

        #   observation_mode
        assert observation_mode in ['default', 'id_visible'], 'Invalid \'observation_mode\''

        # Initialize member variables
        self.ob_range = ob_range
        self.ob_shape = 2*ob_range + 1
        self.ob_mode = observation_mode

        # Action space
        self.action_space = spaces.Discrete(6)

        # To be defined in other functions
        self.world = None
        self.state_init = None
        self.state_obj = None
        self.observation_space = None
        self.agents = None

    # Initialize observation space
    def _initObSpace(self):
        # Observation space
        box_low = -2*np.ones(self.ob_shape)
        if observation_mode == 'default':
            box_high = 1*np.ones(self.ob_shape)
        elif observation_mode == 'id_visible':
            box_high = self.state.shape[0]*self.state.shape[2]*np.ones(self.ob_shape) # max agent id = area of horizontal space
            
        pos_space = spaces.Tuple(spaces.Discrete(self.state.shape[0]), spaces.Discrete(self.state.shape[1]), spaces.Discrete(self.state.shape[2]))
        fac_space = spaces.Discrete(4)
        view_space = spaces.box(box_low, box_high)

        self.observation_space = spaces.Dict({'facing': fac_space, 'position': pos_space, 'view': view_space})

    # Resets environment
    #TODO: place state_init and state_obj initializations in @_state_init.setter, and @state_obj.setter
    def _reset(self):
        assert self.state_obj is not None, 'Objective world not initialized, please assign a plan to env._state_obj first.'
        assert self.state_init is not None, 'Initial world not initialized, plase assign a world to env.state_init first.'

        # Check everything is all right
        assert state_init.shape == state_obj.shape, '\'state_init\' and \'state_obj\' dimensions do not match'
        
        # Initialize data structures
        self.world = Grid3DState(state_init)

        if self.agents is None:
            self.agents = AgentState()
        self.agents.scanWorld(self.world)

    # Returns an observation of an agent
    def _observe(self, agent_id):
        # Check input
        self.agents.validateID(agent_id)

        # Get agent states
        agent_state = self.agents.get(agent_id)
        agent_pos = agents_state[0:3]
        agent_fac = agents_state[3]

        # Get world observation
        ob_view = self.world.getObservation(agent_pos, self.ob_range)
        if self.ob_mode == 'default':
            ob_view = np.clip(ob_view, -2, 1)

        return OrderedDict({'facing': agent_fac, 'position': agent_pos, 'view': ob_view})

    # Executes an action by an agent
    def _step(self, action, agent_id):
        # Check input
        assert action in range(5), 'Invalid action'
        self.agents.validateID(agent_id)

        # Get current agent state
        agent_state = self.agents.get(agent_id)
        agent_pos = agent_state[0:3]
        agent_fac = agent_state[3]

        # Get estimated new agent state
        new_agent_state = self.agents.act(agent_id, action)
        new_agent_pos = new_agent_state[0:3]
        new_agent_fac = new_agent_state[3]

        # Execute action & determine reward
        reward = -1

        if action in [0,1]:     # Move
            # get coordinates and blocks near new position
            new_pos = new_agent_pos
            new_pos_upper = new_pos + np.array([0,1,0])
            new_pos_lower = new_pos + np.array([0,-1,0])
            new_pos_lower2 = new_pos + np.array([0,-2,0])

            block_newpos = self.world.get(new_pos)
            block_upper = self.world.get(new_pos_upper)
            block_lower = self.world.get(new_pos_lower)
            block_lower2 = self.world.get(new_pos_lower2)

            # execute movement if valid
            if block_newpos == 0:  # air in front?
                if block_lower == -1 or block_lower == -2:    # block or ground beneath?
                    new_agent_state[0:3] = new_pos    # horizontal movement
                elif block_lower == 0 and block_lower2 in [-1, -2]:   # block or ground beneath?
                    new_agent_state[0:3] = new_pos_lower  # downstairs movement
            elif block_newpos == -1 and block_upper == 0:   # block in front and air above?
                new_agent_state[0:3] = new_pos_upper    #upstairs movement

            self.world.swap(agent_pos, new_agent_state[0:3])
            self.world.swap(agent_pos + np.array([0,1,0]), new_agent_state[0:3] + np.array([0,1,0]))
            self.agents.set(agent_id, new_agent_state)

        elif action in [2,3]:   # Turn
            self.agents.set(agent_id, new_agent_state)

        elif action in [4,5]:       # Pick & Place
            # determine block movement
            top = agent_pos + np.array([0,1,0])
            front = agent_pos + agent.facing2vec(agent_fac)

            if action == 4:
                source = front
                dest = top
            else:
                source = top
                dest = front

            # execute
            if self.world.get(source) == -1 and self.world.get(dest) == 0:
                self.world.swap(source, dest)
                if action == 5 and self.world_plan[dest[0], dest[1], dest[2]] == 1:
                    reward = 1

        # Perform observation
        state = self.observe(agent_id)

        # Done?
        done = self.world.done(self.state_obj)

        # Additional info
        info = None

        return state, reward, done, info

    # Render gridworld state
    #TODO: Find a way to render the gridworld
    def _render(self, mode="human", close=False):
        return self.world.state





    @property
    def _state(self):
        return self.world.state

    @property
    def _state_init(self):
        return self.state_init

    @_state_init.setter
    def _state_init(self, s0):
        if s0 is not None:
            s0 = np.array(s0)
            assert len(s0.shape) == 3, 'Wrong number of dimensions for \'state_init\''
            self.state_init = s0.copy()
            self.initObSpace()

    @property
    def _state_obj(self):
        return self.state_obj

    @_state_obj.setter
    def _state_obj(self, plan):
        if plan is not None:
            print(plan)
            plan = np.array(plan)
            assert len(plan.shape) == 3, 'Wrong number of dimensions for \'state_obj\''
            self.state_obj = plan
