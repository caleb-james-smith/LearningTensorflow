# dqn.py
import gym
import numpy as np
import random
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQN(object):
    # num episodes: number of times to play the game
    # num steps: max number of steps allowed in game (before quitting, declaring end)
    # min explore rate: min rate of exploration (trying new things) reached at end of training
    # learning rate: step size to adjust parameters when learning
    # discount: rewards decrease over time (over more actions), incentivizes reaching the goal quickly (in fewest actions)
    # batch size: size of batch
    # decay rate: rate of decrease in exploration rate
    def __init__(self, num_episodes=500, num_steps=200, min_explore=0.05, learning_rate=0.10, discount=1.0, batch_size=32, decay=0.02):
        # ------------------------------------------------------------ #
        # Breakout
        # 2D: state is an image showing background, bricks, ball, and platform
        # goal: break all bricks with ball by moving platform to hit the ball
        # allowed platform actions: move left, move right, and no movement
        # ------------------------------------------------------------ #
        # https://gym.openai.com/envs/Breakout-v0
        # https://github.com/JackFurby/Breakout
        # ------------------------------------------------------------ #
        self.model_name     = "breakout"
        # note: enviroment contains state
        self.env            = gym.make('BreakoutDeterministic-v4')
        # note: memory is replay memory containing game states, actions, rewards, and done status
        # each element in memory: [state, action, reward, new_state, done]
        self.memory         = deque(maxlen=2000)
        self.num_episodes   = num_episodes
        self.num_steps      = num_steps
        self.min_explore    = min_explore
        self.learning_rate  = learning_rate
        self.discount       = discount
        self.batch_size     = batch_size
        self.decay          = decay
        self.explore_rate   = self.get_explore_rate(0)
        self.model          = self.create_model()
        self.target_model   = self.create_model()
        print(self.model.summary())
    
    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape 
        # image shape: 2 spacial dimensions, 1 grayscale dimension
        # remove color dimensions, reduce to grayscale
        input_shape = state_shape[:-1] + (1,)

        # build CNN
        model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=input_shape))
        model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        # loss: use mean squared error (mse) to minimize the difference between predicted and actual Q value
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def get_explore_rate(self, episode):
        # exponential decay, don't let it go under minimum
        return max(self.min_explore, np.exp(-episode * self.decay))
    
    def _preprocess(self, img):
        # average across color axis to convert to approximate grayscale (not using full grayscale equation)
        processed = img.mean(axis=2)
        processed = processed[..., np.newaxis]
        processed = processed[np.newaxis, ...]
        return processed

    def choose_action(self, state):
        # exploration: choose random action
        if np.random.random() < self.explore_rate:
            return self.env.action_space.sample()
        # choose best action based on model
        else:
            return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        # check that there are at least one batch of frames stored in memory
        if len(self.memory) < self.batch_size:
            return
        # randomly sample from memory (random selection, shuffled)
        samples = random.sample(self.memory, self.batch_size)
        for sample in samples:
            # unpack sample: samples have same structure as memory
            state, action, reward, new_state, done = sample
            # model: input is state, output is Q value
            target = reward
            if not done:
                target = reward + self.discount * np.amax(self.target_model.predict(new_state)[0])
            target_fn = self.target_model.predict(state) 
            target_fn[0][action] = target
            self.model.fit(state, target_fn, epochs=1, verbose=0)

    def train(self):
        for episode in range(self.num_episodes):
            this_episode = episode + 1
            current_state = self.env.reset()
            current_state = self._preprocess(current_state)
            self.explore_rate = self.get_explore_rate(episode)
            
            done = False
            step = 0
            while not done and step < self.num_steps:
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self._preprocess(new_state)

                self.remember(current_state, action, reward, new_state, done)
                current_state = new_state
                step += 1
            
            self.replay()
            self.update_target_model()
            print("Episode {0:04}/{1:04}".format(this_episode, self.num_episodes))
            if this_episode % 1000 == 0:
                self.model.save("models/{0}_{1:04d}.h5".format(self.model_name, this_episode))

    def run(self):
        self.model          = load_model("models/{0}_{1:04d}.h5".format(self.model_name, self.num_episodes))
        self.explore_rate   = self.min_explore
        
        while True:
            # start and run game
            # initial state
            current_state       = self.env.reset()
            current_state       = self._preprocess(current_state)
            
            done = False
            while not done:
                self.env.render()
                action = self.choose_action(current_state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = self._preprocess(new_state)
                current_state = new_state

if __name__ == "__main__":
    agent = DQN(num_episodes=10000, num_steps=200)
    agent.train()
    #agent.run()

