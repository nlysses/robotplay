import os
os.environ['SAI_API_KEY'] = 'sai_xaAEewt57gwAHfiP7vPG5MaxlXHFa3NG'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

import torch
torch.backends.cudnn.benchmark = True
torch.set_num_threads(12)

import torch.nn as nn
import numpy as np
from sai_rl import SAIClient
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
import signal
import sys
import gymnasium as gym
import json

"""
================================================================================
CONTINUE TRAINING SCRIPT WITH PREPROCESSOR EXPORT
================================================================================
Location: C:/Users/ulyss/RoboAthletes/CONTINUE_TRAINING.py

Purpose: Resume training with lowered entropy coefficient
         All saves (Ctrl+C, checkpoints, final) include preprocessor info

Key Features:
  - ent_coef: 0.03 (reduced from 0.05)
  - Preprocessor saved alongside every model checkpoint
  - Ready for SAI submission with proper observation transformation
================================================================================
"""

# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = r"C:\Users\ulyss\RoboAthletes"
IL_MODELS_DIR = os.path.join(BASE_DIR, "il_models_clustered_v2")
OUTPUT_DIR = os.path.join(BASE_DIR, "sac_models_multitask")

# CHECKPOINT TO RESUME FROM
CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR,
    "stage1_continued_ent03",
    "emergency_52150380.zip"
)

# ENTROPY COEFFICIENT
NEW_ENT_COEF = 0.025

# TOTAL TARGET
TOTAL_TIMESTEPS = 75_000_000

IL_MODEL_MAPPING = {
    0: "approach_and_balance",
    1: "obstacle_kicking",
    2: "powerful_goalie_beater"
}

ENV_IDS = {
    0: "LowerT1KickToTarget-v0",
    1: "LowerT1ObstaclePenaltyKick-v0",
    2: "LowerT1GoaliePenaltyKick-v0"
}


# ==========================================
# SUBMISSION PREPROCESSOR
# ==========================================
# This class transforms raw SAI observations into the format your model expects
# It will be saved as a .py file alongside your model for submission

PREPROCESSOR_CODE = '''
import numpy as np

class Preprocessor:
    """
    Preprocessor for SAI Booster Soccer Showdown submission.
    
    Transforms raw environment observations into the format the model expects:
    - Pads observation to 54 dimensions (max across all 3 tasks)
    - Appends 3-dim task one-hot encoding
    - Final output: 57 dimensions
    
    Raw observation dimensions per task:
    - Task 0 (LowerT1KickToTarget-v0): 39 dims
    - Task 1 (LowerT1ObstaclePenaltyKick-v0): 54 dims
    - Task 2 (LowerT1GoaliePenaltyKick-v0): 45 dims
    """
    
    def __init__(self):
        self.max_obs_dim = 54
        self.n_tasks = 3
    
    def get_task_onehot(self, info):
        """Extract task one-hot from info dict"""
        if "task_index" in info:
            task_id = info["task_index"]
            onehot = np.zeros(self.n_tasks, dtype=np.float32)
            if isinstance(task_id, (list, np.ndarray)):
                return np.array(task_id, dtype=np.float32)
            else:
                onehot[int(task_id)] = 1.0
            return onehot
        return np.zeros(self.n_tasks, dtype=np.float32)
    
    def modify_state(self, obs, info):
        """
        Transform raw observation to model input format.
        
        Args:
            obs: Raw observation from environment (39, 45, or 54 dims)
            info: Info dict containing task_index
            
        Returns:
            Preprocessed observation (57 dims)
        """
        if len(obs.shape) > 1:
            obs = obs.squeeze()
        
        if len(obs) < self.max_obs_dim:
            padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
            obs_padded = np.concatenate([obs, padding])
        else:
            obs_padded = obs[:self.max_obs_dim]
        
        task_onehot = self.get_task_onehot(info)
        
        return np.concatenate([obs_padded, task_onehot]).astype(np.float32)
'''


def save_preprocessor(save_dir):
    """Save preprocessor.py file for submission"""
    preprocessor_path = os.path.join(save_dir, "preprocessor.py")
    with open(preprocessor_path, 'w') as f:
        f.write(PREPROCESSOR_CODE)
    return preprocessor_path


def save_model_with_preprocessor(model, save_path, save_dir):
    """Save model and preprocessor together"""
    # Save the model
    model.save(save_path)
    
    # Save preprocessor alongside
    preprocessor_path = save_preprocessor(save_dir)
    
    # Save submission instructions
    instructions_path = os.path.join(save_dir, "SUBMISSION_README.txt")
    with open(instructions_path, 'w') as f:
        f.write(f"""
================================================================================
BOOSTER SOCCER SHOWDOWN - SUBMISSION INSTRUCTIONS
================================================================================

Model: {os.path.basename(save_path)}
Preprocessor: preprocessor.py

To submit via Python:

    import numpy as np
    from sai_rl import SAIClient
    from stable_baselines3 import SAC
    
    # Define preprocessor class (copy from preprocessor.py)
    class Preprocessor:
        def __init__(self):
            self.max_obs_dim = 54
            self.n_tasks = 3
        
        def get_task_onehot(self, info):
            if "task_index" in info:
                task_id = info["task_index"]
                onehot = np.zeros(self.n_tasks, dtype=np.float32)
                if isinstance(task_id, (list, np.ndarray)):
                    return np.array(task_id, dtype=np.float32)
                else:
                    onehot[int(task_id)] = 1.0
                return onehot
            return np.zeros(self.n_tasks, dtype=np.float32)
        
        def modify_state(self, obs, info):
            if len(obs.shape) > 1:
                obs = obs.squeeze()
            if len(obs) < self.max_obs_dim:
                padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
                obs = np.concatenate([obs, padding])
            task_onehot = self.get_task_onehot(info)
            return np.concatenate([obs, task_onehot]).astype(np.float32)
    
    # Load and submit
    model = SAC.load("{save_path}", device="cpu")
    sai = SAIClient(scene_id="scn_fk2IPfTF7cVe")
    sai.submit("Your Model Name", model, preprocessor_class=Preprocessor)

================================================================================
""")
    
    return save_path, preprocessor_path


# ==========================================
# CLUSTERED IL MODEL
# ==========================================

class ClusteredImitationPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, n_clusters=5, embed_dim=32):
        super().__init__()
        self.cluster_embedding = nn.Embedding(n_clusters, embed_dim)
        self.fc1 = nn.Linear(state_dim + embed_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc_out = nn.Linear(256, action_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.tanh = nn.Tanh()
        
    def forward(self, state, cluster_label=None):
        if cluster_label is None:
            cluster_label = torch.zeros(state.shape[0], 1, dtype=torch.long, device=state.device)
        cluster_label = torch.clamp(cluster_label, min=0, max=self.cluster_embedding.num_embeddings - 1)
        cluster_emb = self.cluster_embedding(cluster_label.squeeze(-1))
        x = torch.cat([state, cluster_emb], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        return self.tanh(self.fc_out(x))


def load_il_model(phase_name, device="cuda"):
    model_path = os.path.join(IL_MODELS_DIR, phase_name, f"{phase_name}_best.pt")
    if not os.path.exists(model_path):
        print(f"[WARNING] IL model not found: {model_path}")
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dim = checkpoint.get('state_dim')
        action_dim = checkpoint.get('action_dim')
        n_clusters = checkpoint.get('n_clusters', 4)
        if state_dim is None or action_dim is None:
            return None
        model = ClusteredImitationPolicy(state_dim, action_dim, n_clusters=n_clusters).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"  ✅ Loaded: {phase_name} (clusters: {n_clusters})")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load {phase_name}: {e}")
        return None


# ==========================================
# TASK PREPROCESSOR (for training)
# ==========================================

class TaskPreprocessor:
    def __init__(self, n_tasks=3):
        self.n_tasks = n_tasks
    
    def get_task_onehot(self, task_id):
        onehot = np.zeros(self.n_tasks, dtype=np.float32)
        onehot[task_id] = 1.0
        return onehot


class TaskConditionedEnv(gym.Wrapper):
    def __init__(self, env, task_id, preprocessor, max_obs_dim=54):
        super().__init__(env)
        self.task_id = task_id
        self.preprocessor = preprocessor
        self.max_obs_dim = max_obs_dim
        self.original_obs_dim = env.observation_space.shape[0]
        final_dim = max_obs_dim + preprocessor.n_tasks
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(final_dim,), dtype=np.float32
        )
    
    def _pad_observation(self, obs):
        if len(obs.shape) > 1:
            obs = obs.squeeze()
        if len(obs) < self.max_obs_dim:
            padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
            obs_padded = np.concatenate([obs, padding])
        else:
            obs_padded = obs
        task_onehot = self.preprocessor.get_task_onehot(self.task_id)
        return np.concatenate([obs_padded, task_onehot]).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info['task_id'] = self.task_id
        return self._pad_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info['task_id'] = self.task_id
        return self._pad_observation(obs), reward, terminated, truncated, info


# ==========================================
# REWARD WRAPPER
# ==========================================

class MultiTaskRewardWrapper(gym.Wrapper):
    def __init__(self, env, il_model, device="cuda", task_id=0):
        super().__init__(env)
        self.il_model = il_model
        self.device = device
        self.task_id = task_id
        self.obs_dims = {0: 39, 1: 54, 2: 45}
        self.original_obs_dim = self.obs_dims[task_id]
        self.kp = 50.0
        self.kd = 2.0
        self.prev_ball_distance = None
        self.prev_target_distance = None
        self.steps_survived = 0
        self.max_episode_steps = 400
        self._setup_reward_weights()
    
    def _setup_reward_weights(self):
        if self.task_id == 0:
            self.env_reward_scale = 1.0
            self.il_weight = 0.15
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.55
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
        elif self.task_id == 1:
            self.env_reward_scale = 1.0
            self.il_weight = 0.2
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.45
            self.obstacle_weight = 0.35
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
        elif self.task_id == 2:
            self.env_reward_scale = 1.0
            self.il_weight = 0.25
            self.approach_weight = 0.25
            self.balance_weight = 0.1
            self.target_weight = 0.35
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.5
            self.time_penalty_weight = 0.002
    
    def _get_raw_obs(self, obs):
        return obs[:self.original_obs_dim]
    
    def compute_il_bonus(self, obs):
        if not self.il_model:
            return 0.0
        try:
            obs_raw = self._get_raw_obs(obs)
            joint_pos = obs_raw[:12]
            joint_vel = obs_raw[12:24]
            root_pos = np.zeros(3)
            root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            root_vel = np.zeros(6)
            full_qpos = np.concatenate([root_pos, root_quat, joint_pos])
            full_qvel = np.concatenate([root_vel, joint_vel]).astype(np.float32)
            state = np.concatenate([full_qpos, full_qvel]).astype(np.float32)
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                delta_qpos = self.il_model(state_tensor, cluster_label=None)
                delta_qpos = delta_qpos.squeeze(0).cpu().numpy()
            actuated_delta = delta_qpos[-12:]
            target_qpos = joint_pos + actuated_delta
            position_error = target_qpos - joint_pos
            torque = self.kp * position_error - self.kd * joint_vel
            torque_magnitude = np.mean(np.abs(torque))
            return np.clip(torque_magnitude * 0.1, 0.0, 3.0)
        except Exception:
            return 0.0
    
    def compute_approach_bonus(self, obs):
        try:
            obs_raw = self._get_raw_obs(obs)
            ball_pos = obs_raw[24:27]
            dist = np.linalg.norm(ball_pos)
            if self.prev_ball_distance is None:
                self.prev_ball_distance = dist
                return 0.0
            improvement = self.prev_ball_distance - dist
            self.prev_ball_distance = dist
            return improvement * 10.0
        except Exception:
            return 0.0
    
    def compute_balance_bonus(self, obs):
        try:
            obs_raw = self._get_raw_obs(obs)
            joint_pos = obs_raw[:12]
            balance_score = 1.0 - np.mean(np.abs(joint_pos)) * 1.2
            return np.clip(balance_score, 0, 1.0) * 5.0
        except Exception:
            return 0.0
    
    def compute_target_bonus(self, obs):
        try:
            obs_raw = self._get_raw_obs(obs)
            if self.task_id == 0:
                if len(obs_raw) < 36:
                    return 0.0
                target_pos = obs_raw[33:36]
            elif self.task_id == 1:
                if len(obs_raw) < 42:
                    return 0.0
                target_pos = obs_raw[39:42]
            elif self.task_id == 2:
                if len(obs_raw) < 36:
                    return 0.0
                target_pos = obs_raw[33:36]
            else:
                return 0.0
            dist = np.linalg.norm(target_pos)
            if self.prev_target_distance is None:
                self.prev_target_distance = dist
                return 0.0
            improvement = self.prev_target_distance - dist
            self.prev_target_distance = dist
            return improvement * 12.0
        except Exception:
            return 0.0
    
    def compute_obstacle_avoidance_bonus(self, obs):
        if self.task_id != 1:
            return 0.0
        try:
            obs_raw = self._get_raw_obs(obs)
            if len(obs_raw) < 54:
                return 0.0
            obstacles = [obs_raw[45:48], obs_raw[48:51], obs_raw[51:54]]
            ball_pos = obs_raw[24:27]
            min_dist = min([np.linalg.norm(ball_pos - o) for o in obstacles])
            if min_dist > 2.5:
                return 4.0
            elif min_dist > 1.5:
                return 2.0
            elif min_dist > 1.0:
                return 0.0
            else:
                return -3.0
        except Exception:
            return 0.0
    
    def compute_ball_velocity_bonus(self, obs):
        try:
            obs_raw = self._get_raw_obs(obs)
            ball_vel = obs_raw[27:30]
            if self.task_id == 0:
                goal_pos = obs_raw[33:36]
            elif self.task_id == 1:
                goal_pos = obs_raw[33:36]
            elif self.task_id == 2:
                goal_pos = obs_raw[33:36]
            else:
                return 0.0
            goal_dist = np.linalg.norm(goal_pos)
            if goal_dist < 0.1:
                return 0.0
            goal_dir = goal_pos[:2] / (goal_dist + 1e-8)
            vel_toward_goal = np.dot(ball_vel[:2], goal_dir)
            return np.clip(vel_toward_goal * 3.0, 0.0, 8.0)
        except Exception:
            return 0.0
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        il_bonus = self.compute_il_bonus(obs)
        approach_bonus = self.compute_approach_bonus(obs)
        balance_bonus = self.compute_balance_bonus(obs)
        target_bonus = self.compute_target_bonus(obs)
        obstacle_bonus = self.compute_obstacle_avoidance_bonus(obs)
        velocity_bonus = self.compute_ball_velocity_bonus(obs)
        survival_bonus = 0.01
        time_penalty = -self.time_penalty_weight * self.steps_survived
        
        total_reward = (
            reward * self.env_reward_scale +
            target_bonus * self.target_weight +
            il_bonus * self.il_weight +
            approach_bonus * self.approach_weight +
            balance_bonus * self.balance_weight +
            obstacle_bonus * self.obstacle_weight +
            velocity_bonus * self.velocity_weight +
            survival_bonus +
            time_penalty
        )
        self.steps_survived += 1
        if self.steps_survived >= self.max_episode_steps:
            trunc = True
            info['TimeLimit.truncated'] = True
        return obs, total_reward, term, trunc, info
    
    def reset(self, **kwargs):
        self.prev_ball_distance = None
        self.prev_target_distance = None
        self.steps_survived = 0
        return self.env.reset(**kwargs)


# ==========================================
# CALLBACKS
# ==========================================

class PerformanceCallback(BaseCallback):
    def __init__(self, total_timesteps, check_freq=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.start_time = None
        self.last_log_time = None
        self.starting_timesteps = 0
    
    def _on_training_start(self):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
        self.starting_timesteps = self.model.num_timesteps
    
    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            now = datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            since_last = (now - self.last_log_time).total_seconds()
            self.last_log_time = now
            
            steps_this_session = self.num_timesteps - self.starting_timesteps
            progress = self.num_timesteps / self.total_timesteps * 100
            fps = steps_this_session / elapsed if elapsed > 0 else 0
            recent_fps = self.check_freq / since_last if since_last > 0 else 0
            
            ep_buffer = self.model.ep_info_buffer
            if len(ep_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in ep_buffer])
                mean_length = np.mean([ep['l'] for ep in ep_buffer])
            else:
                mean_reward = 0
                mean_length = 0
            
            gpu_mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
            
            remaining = self.total_timesteps - self.num_timesteps
            if steps_this_session > 0:
                eta = (elapsed / steps_this_session) * remaining
            else:
                eta = 0
            
            print(f"\n{'='*80}")
            print(f"CONTINUED TRAINING PROGRESS (ent_coef={NEW_ENT_COEF})")
            print(f"{'='*80}")
            print(f"Step: {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%)")
            print(f"This Session: {steps_this_session:,} steps")
            print(f"FPS: {fps:.0f} (avg) | {recent_fps:.0f} (recent)")
            print(f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.0f}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {gpu_mem:.2f} GB")
            print(f"Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h")
            print(f"{'='*80}\n")
        return True


class GracefulShutdownCallback(BaseCallback):
    """Save model WITH preprocessor on Ctrl+C"""
    
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.requested = False
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, sig, frame):
        print("\n[SHUTDOWN] Ctrl+C detected - Saving model with preprocessor...")
        self.requested = True
    
    def _on_step(self):
        if self.requested:
            # Save model
            model_filename = f"emergency_{self.num_timesteps}.zip"
            model_path = os.path.join(self.save_path, model_filename)
            self.model.save(model_path)
            print(f"[SAVED] Model: {model_path}")
            
            # Save preprocessor
            preprocessor_path = save_preprocessor(self.save_path)
            print(f"[SAVED] Preprocessor: {preprocessor_path}")
            
            # Save submission instructions
            instructions_path = os.path.join(self.save_path, "SUBMISSION_README.txt")
            with open(instructions_path, 'w') as f:
                f.write(f"""
================================================================================
EMERGENCY SAVE - BOOSTER SOCCER SHOWDOWN
================================================================================
Timestep: {self.num_timesteps:,}
Model: {model_filename}
Preprocessor: preprocessor.py

To submit this model:

    from sai_rl import SAIClient
    from stable_baselines3 import SAC
    import numpy as np
    
    class Preprocessor:
        def __init__(self):
            self.max_obs_dim = 54
            self.n_tasks = 3
        
        def get_task_onehot(self, info):
            if "task_index" in info:
                task_id = info["task_index"]
                onehot = np.zeros(self.n_tasks, dtype=np.float32)
                if isinstance(task_id, (list, np.ndarray)):
                    return np.array(task_id, dtype=np.float32)
                else:
                    onehot[int(task_id)] = 1.0
                return onehot
            return np.zeros(self.n_tasks, dtype=np.float32)
        
        def modify_state(self, obs, info):
            if len(obs.shape) > 1:
                obs = obs.squeeze()
            if len(obs) < self.max_obs_dim:
                padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
                obs = np.concatenate([obs, padding])
            task_onehot = self.get_task_onehot(info)
            return np.concatenate([obs, task_onehot]).astype(np.float32)
    
    model = SAC.load("{model_path}", device="cpu")
    sai = SAIClient(scene_id="scn_fk2IPfTF7cVe")
    sai.submit("Your Model Name", model, preprocessor_class=Preprocessor)
================================================================================
""")
            print(f"[SAVED] Instructions: {instructions_path}")
            
            print("\n[COMPLETE] All files saved. Ready for submission.")
            sys.exit(0)
        return True


class CheckpointWithPreprocessorCallback(BaseCallback):
    """Custom checkpoint callback that also saves preprocessor"""
    
    def __init__(self, save_freq, save_path, name_prefix="checkpoint", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        os.makedirs(save_path, exist_ok=True)
        
        # Save preprocessor once at start
        save_preprocessor(save_path)
    
    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, 
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"[CHECKPOINT] Saved: {model_path}")
        return True


# ==========================================
# MAIN
# ==========================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n" + "="*80)
    print("CONTINUE TRAINING - WITH PREPROCESSOR EXPORT")
    print("="*80)
    print(f"\nCheckpoint: {CHECKPOINT_PATH}")
    print(f"ent_coef: {NEW_ENT_COEF}")
    print(f"Device: {device}")
    print(f"\nAll saves will include preprocessor.py for submission")
    print("="*80 + "\n")
    
    # Verify checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nAvailable checkpoints:")
        checkpoint_dir = os.path.join(OUTPUT_DIR, "stage1_continued_ent03")
        if os.path.exists(checkpoint_dir):
            for f in sorted(os.listdir(checkpoint_dir)):
                if f.endswith('.zip'):
                    fpath = os.path.join(checkpoint_dir, f)
                    fsize = os.path.getsize(fpath) / 1024 / 1024
                    print(f"  - {f} ({fsize:.1f} MB)")
        return
    
    print("[1/4] Loading IL models...")
    il_models = {}
    for task_id, phase_name in IL_MODEL_MAPPING.items():
        il_models[task_id] = load_il_model(phase_name, device)
    
    print("\n[2/4] Creating environments...")
    preprocessor = TaskPreprocessor(n_tasks=3)
    n_envs_per_task = 5
    
    env_makers = []
    for task_id in range(3):
        env_id = ENV_IDS[task_id]
        il_model = il_models.get(task_id, None)
        sai = SAIClient(env_id=env_id)
        
        for _ in range(n_envs_per_task):
            def make_env(task=task_id, sai_inst=sai, il_mod=il_model):
                base_env = sai_inst.make_env()
                reward_wrapped = MultiTaskRewardWrapper(
                    base_env, il_model=il_mod, device=device, task_id=task
                )
                task_wrapped = TaskConditionedEnv(
                    reward_wrapped, task_id=task, preprocessor=preprocessor
                )
                return Monitor(task_wrapped)
            env_makers.append(make_env)
    
    train_env = DummyVecEnv(env_makers)
    print(f"  ✅ Created {len(env_makers)} training environments")
    
    # Eval environments
    eval_makers = []
    for task_id in range(3):
        def make_eval(task=task_id):
            sai = SAIClient(env_id=ENV_IDS[task])
            base = sai.make_env()
            task_wrapped = TaskConditionedEnv(base, task, preprocessor)
            return Monitor(task_wrapped)
        eval_makers.append(make_eval)
    eval_env = DummyVecEnv(eval_makers)
    print(f"  ✅ Created {len(eval_makers)} eval environments")
    
    print(f"\n[3/4] Loading checkpoint...")
    model = SAC.load(CHECKPOINT_PATH, env=train_env, device=device)
    
    # Update entropy coefficient
    model.ent_coef_tensor = torch.tensor([NEW_ENT_COEF], device=device)
    print(f"  ✅ Set ent_coef = {model.ent_coef_tensor.item():.4f}")
    print(f"  ✅ Current timesteps: {model.num_timesteps:,}")
    
    # Output directory
    continued_dir = os.path.join(OUTPUT_DIR, "stage1_continued_ent03")
    checkpoints_dir = os.path.join(continued_dir, "checkpoints")
    os.makedirs(continued_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Save preprocessor to main output dir
    save_preprocessor(continued_dir)
    print(f"  ✅ Saved preprocessor.py to {continued_dir}")
    
    # Callbacks
    callbacks = CallbackList([
        CheckpointWithPreprocessorCallback(
            save_freq=100_000 // 15,
            save_path=checkpoints_dir,
            name_prefix="continued",
            verbose=1
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=continued_dir,
            log_path=continued_dir,
            eval_freq=50_000 // 15,
            n_eval_episodes=9,
            deterministic=True,
            verbose=1
        ),
        PerformanceCallback(TOTAL_TIMESTEPS),
        GracefulShutdownCallback(continued_dir)
    ])
    
    print(f"\n[4/4] Resuming training...")
    print(f"  Target: {TOTAL_TIMESTEPS:,} total timesteps")
    print(f"  Remaining: ~{TOTAL_TIMESTEPS - model.num_timesteps:,} steps")
    print(f"  Press Ctrl+C to save and exit (with preprocessor)\n")
    
    response = input("Start training? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        train_env.close()
        eval_env.close()
        return
    
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=False,
        progress_bar=True
    )
    
    # Save final model with preprocessor
    final_path = os.path.join(continued_dir, "final_model.zip")
    save_model_with_preprocessor(model, final_path, continued_dir)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Time: {elapsed/3600:.2f} hours")
    print(f"Final Model: {final_path}")
    print(f"Best Model: {os.path.join(continued_dir, 'best_model.zip')}")
    print(f"Preprocessor: {os.path.join(continued_dir, 'preprocessor.py')}")
    print(f"\nSee SUBMISSION_README.txt for submission instructions")
    print("="*80 + "\n")
    
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
