# AI 2018

# Importing the libraries
import numpy as np
import gym

from btgym import BTgymEnv
import IPython.display as Display
import PIL.Image as Image
from gym import spaces
from btgym.spaces import DictSpace



# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):
        self.nb_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = BTgymEnv(
                        filename='DAT_ASCII_EURUSD_M1_2016.csv',
                        # This param is the only one changed:
                        state_shape={
                                'raw_state': spaces.Box(
                                    shape=(30, 4),
                                    low=-100, 
                                    high=100,
                                    dtype=np.float32,
                                ),
                                'metadata': DictSpace(
                                    {
                                        'type': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=1,
                                            dtype=np.uint32
                                        ),
                                        'trial_num': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=10 ** 10,
                                            dtype=np.uint32
                                        ),
                                        'trial_type': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=1,
                                            dtype=np.uint32
                                        ),
                                        'sample_num': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=10 ** 10,
                                            dtype=np.uint32
                                        ),
                                        'first_row': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=10 ** 10,
                                            dtype=np.uint32
                                        ),
                                        'timestamp': spaces.Box(
                                            shape=(),
                                            low=0,
                                            high=np.finfo(np.float64).max,
                                            dtype=np.float64
                                        ),
                                    }
                                )
                            },
                        skip_frame=5,
                        start_cash=100,
                        render_ylabel='Price Lines',
                        render_size_episode=(12,8),
                        render_size_human=(8, 3.5),
                        render_size_state=(10, 3.5),
                        render_dpi=75,
                        verbose=0,
                    )
      

# Normalizing the states

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
        
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step

    def save_model(self, fn):
        self.policy.save(fn)
              
    def show_rendered_image(self, rgb_array):
        """
        Convert numpy array to RGB image using PILLOW and
        show it inline using IPykernel.
        """
        Display.display(Image.fromarray(rgb_array))

    def render_all_modes(self, env):
        """
        Retrieve and show environment renderings
        for all supported modes.
        """
        for mode in self.env.metadata['render.modes']:
            print('[{}] mode:'.format(mode))
            self.show_rendered_image(self.env.render(mode))

            
# Exploring the policy on one specific direction and over one episode
def explore(env, normalizer, policy, direction = None, delta = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        state, reward, done, _ = env.step(action)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    return sum_rewards


# Training the AI
def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        print("Completed trial #{} ".format(step))
        policy.render_all_modes(env)
        policy.save_model("model.model".format(step))


# Running the main code
hp = Hp()
env = hp.env_name
nb_inputs = list(env.observation_space.shape.items())[0]
nb_outputs = env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)







