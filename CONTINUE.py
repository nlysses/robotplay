import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sai_rl import SAIClient
import sai_mujoco  # Registers the MuJoCo-based environments like LowerT1KickToTarget-v0

# Preprocessor class to pad obs to 54 dims and append 3-dim task one-hot for generalization across the scene
class Preprocessor:
    def __init__(self):
        pass

    def get_task_onehot(self, info):
        if "task_index" in info:
            return info["task_index"]
        else:
            return np.array([0, 0, 0])  # Fallback for single-task envs

    def modify_state(self, obs, info):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        pad_len = 54 - obs.shape[1]
        if pad_len > 0:
            pad = np.zeros((obs.shape[0], pad_len))
            obs_padded = np.hstack((obs, pad))
        else:
            obs_padded = obs
        return np.hstack((obs_padded, task_onehot))

# Wrapper to apply preprocessor to the environment
class PreprocessWrapper(gym.Wrapper):
    def __init__(self, env, preprocessor):
        super().__init__(env)
        self.preprocessor = preprocessor
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(57,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        new_obs = self.preprocessor.modify_state(obs, info)
        return new_obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)  # Gymnasium returns 5 values
        done = terminated or truncated
        new_obs = self.preprocessor.modify_state(obs, info)
        return new_obs, rew, done, info

# Modified callback for entropy decay correlated to raw eval rewards
class EvalCorrelatedEntDecayCallback(BaseCallback):
    def __init__(self, initial_ent=0.025, base_decay_rate=0.00005, min_ent=0.001, reward_threshold=10.0):
        super().__init__()
        self.initial_ent = initial_ent
        self.base_decay_rate = base_decay_rate
        self.min_ent = min_ent
        self.reward_threshold = reward_threshold
        self.eval_rew_mean = 0  # Track latest eval reward

    def _on_step(self) -> bool:
        # Update from eval logs if available (accessed via logger)
        if 'eval/ep_rew_mean' in self.locals.get('logger', {}).name_to_value:
            self.eval_rew_mean = self.locals['logger'].name_to_value['eval/ep_rew_mean']
        
        # Calculate dynamic decay rate based on eval rewards (higher rewards -> faster decay for exploitation)
        decay_rate = self.base_decay_rate * 2 if self.eval_rew_mean > self.reward_threshold else self.base_decay_rate
        new_ent = self.initial_ent * np.exp(-decay_rate * self.num_timesteps)
        new_ent = max(new_ent, self.min_ent)
        self.model.ent_coef_tensor.data.fill_(new_ent)
        print(f"Adjusted ent_coef to {new_ent} based on eval ep_rew_mean: {self.eval_rew_mean}")
        return True

# Initialize SAIClient and create the scene environment (meta-env for all 3 tasks)
sai = SAIClient(comp_id="booster-soccer-showdown")
env = sai.make_env()
preprocessor = Preprocessor()
env = PreprocessWrapper(env, preprocessor)

# Load the specific model to continue from
model_path = r"C:\Users\ulyss\RoboAthletes\sac_models_multitask\stage1_continued_ent03\best_model.zip"
model = SAC.load(model_path, env=env, ent_coef=0.025)  # Override to start at 0.025

# Setup eval callback (evaluates every 50k steps, saves best model)
eval_env = sai.make_env()  # Separate eval env
eval_env = PreprocessWrapper(eval_env, preprocessor)
eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=50000, deterministic=True, render=False)

# Custom decay callback, starts immediately
decay_callback = EvalCorrelatedEntDecayCallback()

# Continue training with callbacks
additional_steps = 75000000  # Additional steps for this continuation
model.learn(total_timesteps=additional_steps, callback=[eval_callback, decay_callback], log_interval=10)

# Save the final continued model
model.save("continued_best_model_ent025_decay.zip")

env.close()
eval_env.close()