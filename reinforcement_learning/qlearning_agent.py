# qlearning_agent.py
import gym
import numpy as np

class QLearningAgent(object):
    # num bins: granularity to quantize continuous values (position and velocity)
    # num episodes: number of times to play the game
    # learning rate: step size to adjust parameters when learning
    # min explore rate: min rate of exploration (trying new things) reached at end of training
    # discount: rewards decrease over time (over more actions), incentivizes reaching the goal quickly (in fewest actions)
    # decay rate: rate of decrease in exploration rate
    def __init__(self, num_bins=10, num_episodes=1000, learning_rate=0.10, min_explore=0.05, discount=1.0, decay=50):
        # ------------------------------------------------------------ #
        # Mountain Car
        # 1D: state has car position and velocity
        # goal: move car to the top of the hill
        # allowed car actions: move left, move right, and no movement
        # ------------------------------------------------------------ #
        # note: enviroment contains state
        self.env            = gym.make('MountainCar-v0')
        self.num_bins       = num_bins
        self.num_episodes   = num_episodes
        self.learning_rate  = learning_rate
        self.min_explore    = min_explore
        self.discount       = discount
        self.decay          = decay
        self.explore_rate   = self.get_explore_rate(episode=0)
        self.q_table        = np.zeros((self.num_bins, self.num_bins) + (self.env.action_space.n,))

    def discretize_state(self, obs):
        # discretize continuous state into finite bins
        # observation array: [position, velocity]
        env_low         = self.env.observation_space.low
        env_high        = self.env.observation_space.high
        bin_width       = (env_high - env_low) / self.num_bins
        position_bin    = int((obs[0] - env_low[0]) / bin_width[0])
        velocity_bin    = int((obs[1] - env_low[1]) / bin_width[1])
        return (position_bin, velocity_bin)

    def get_explore_rate(self, episode):
        # exponential decay, don't let it go under minimum
        return max(self.min_explore, np.exp(-episode / self.decay))

    def choose_action(self, state):
        # choose random action (sample) based on explore rate (probability of exploring)
        if np.random.random() < self.explore_rate:
            return self.env.action_space.sample()
        # choose optimal action (action with highest Q value)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, new_state):
        q_hat = reward + self.discount * np.max(self.q_table[new_state])
        self.q_table[state][action] += self.learning_rate * (q_hat - self.q_table[state][action])
    
    def train(self):
        show = False
        num_iters = 200
        for episode in range(self.num_episodes):
            iteration = episode + 1
            if show and iteration % num_iters == 0:
                print("Iteration: {0}".format(iteration))
            # start and run game
            # initial state
            current_state       = self.discretize_state(self.env.reset())
            self.explore_rate   = self.get_explore_rate(episode=episode)
            
            done = False
            while not done:
                #print(current_state)
                if show and iteration % num_iters == 0:
                    self.env.render()
                # choose action
                action = self.choose_action(current_state)
                # take action
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(new_state)
                # update q table
                self.update_q_table(current_state, action, reward, new_state)
                # update current state
                current_state = new_state

    def run(self):
        while True:
            # start and run game
            # initial state
            current_state = self.discretize_state(self.env.reset())
            
            done = False
            while not done:
                #print(current_state)
                self.env.render()
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self.discretize_state(new_state)
                current_state = new_state

if __name__ == "__main__":
    agent = QLearningAgent(num_bins=10, num_episodes=1000, discount=1.0)
    agent.train()
    agent.run()

