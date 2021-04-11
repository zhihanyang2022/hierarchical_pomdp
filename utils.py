import numpy as np

class ReplayBuffer:
    def __init__(self, max_size=5e5, for_top_level=False):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
        self.for_top_level = for_top_level
    
    def add(self, transition):
        if self.for_top_level:
            assert len(transition) == 6, "transition must have length = 6"
        else:
            assert len(transition) == 7, "transition must have length = 7"
        
        # transiton is tuple of (state, action, reward, next_state, goal, gamma, done)
        self.buffer.append(transition)
        self.size +=1
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        if self.for_top_level:
            states, actions, rewards, next_states, gamma, dones = [], [], [], [], [], []
        else:
            states, actions, rewards, next_states, goals, gamma, dones = [], [], [], [], [], [], []
        
        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            if self.for_top_level:
                gamma.append(np.array(self.buffer[i][4], copy=False))
                dones.append(np.array(self.buffer[i][5], copy=False))
            else:
                goals.append(np.array(self.buffer[i][4], copy=False))
                gamma.append(np.array(self.buffer[i][5], copy=False))
                dones.append(np.array(self.buffer[i][6], copy=False))

        if self.for_top_level:
            return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(gamma), np.array(dones)
        else:
            return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(goals),  np.array(gamma), np.array(dones)
    
