import os
os.environ['SAI_API_KEY'] = 'sai_xaAEewt57gwAHfiP7vPG5MaxlXHFa3NG'

import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import SAC

class SubmissionPreprocessor:
    def __init__(self):
        self.max_obs_dim = 54
        self.n_tasks = 3

    def get_task_onehot(self, info):
        if "task_index" in info:
            task_id = info["task_index"]
            onehot = np.zeros(self.n_tasks, dtype=np.float32)
            onehot[task_id] = 1.0
            return onehot
        return np.zeros(self.n_tasks, dtype=np.float32)

    def modify_state(self, obs, info):
        if len(obs.shape) > 1:
            obs = obs.squeeze()
        if len(obs) < self.max_obs_dim:
            padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
            obs_padded = np.concatenate([obs, padding])
        else:
            obs_padded = obs
        task_onehot = self.get_task_onehot(info)
        return np.concatenate([obs_padded, task_onehot]).astype(np.float32)

# Load your best model
best_model_path = r"C:\Users\ulyss\RoboAthletes\sac_models_multitask\stage1_continued_ent03\best_model.zip"
model = SAC.load(best_model_path)

# Now SAIClient will use the API key from environment
sai = SAIClient(scene_id="scn_fk2IPfTF7cVe")

# Export to ONNX
sai.save_model("best_model_onnx", model, use_onnx=True)
print("Model exported successfully!")