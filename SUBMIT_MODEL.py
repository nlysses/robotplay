import os
os.environ['SAI_API_KEY'] = 'sai_xaAEewt57gwAHfiP7vPG5MaxlXHFa3NG'

import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import SAC

class Preprocessor:
    def __init__(self):
        self.max_obs_dim = 54
        self.n_tasks = 3
    
    def get_task_onehot(self, info):
        if "task_index" in info:
            # Assume task_index is already one-hot (array/list per docs); convert to np.array
            return np.array(info["task_index"], dtype=np.float32)
        else:
            return np.zeros(self.n_tasks, dtype=np.float32)
    
    def modify_state(self, obs, info):
        # Ensure obs is at least 2D (add batch dim if single env)
        obs = np.atleast_2d(obs)
        batch_size, obs_dim = obs.shape
        
        # Pad each batched observation to max_dim
        if obs_dim < self.max_obs_dim:
            padding = np.zeros((batch_size, self.max_obs_dim - obs_dim), dtype=np.float32)
            obs_padded = np.hstack((obs, padding))
        else:
            obs_padded = obs[:, :self.max_obs_dim]
        
        # Ensure task_onehot is at least 2D (add batch dim if single)
        task_onehot = np.atleast_2d(self.get_task_onehot(info))
        
        # Concatenate along feature axis (handles batch matching)
        return np.hstack((obs_padded, task_onehot)).astype(np.float32)

model_path = r"C:\Users\ulyss\RoboAthletes\sac_models_multitask\stage1_continued_ent03\LegDays.zip"
model = SAC.load(model_path, device="cpu")

sai = SAIClient(comp_id="cmp_xnSCxcJXQclQ")

sai.submit("LegDaaay", model, preprocessor_class=Preprocessor)