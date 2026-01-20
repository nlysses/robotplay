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
ROBUST MULTI-TASK SAC TRAINING
================================================================================
Location: C:/Users/ulyss/RoboAthletes/

IL Models Location: C:/Users/ulyss/RoboAthletes/il_models_clustered_v2/

Phase Mapping:
  Task 0 (KickToTarget)     → approach_and_balance
  Task 1 (ObstaclePenalty)  → obstacle_kicking  
  Task 2 (GoaliePenalty)    → powerful_goalie_beater

Training Strategy:
  Stage 1: 25M steps with specialized IL models (current)
  Stage 2: 10M steps with complete_generalization (polish)

Expected Results:
  Step 5M:   40-50% success
  Step 15M:  70-75% success
  Step 25M:  85-90% success
  After Polish: 93-96% success
================================================================================
"""


# ==========================================
# CONFIGURATION - EDIT THESE PATHS
# ==========================================

# Base directory where you're running from
BASE_DIR = r"C:\Users\ulyss\RoboAthletes"

# IL models directory (relative to BASE_DIR)
IL_MODELS_DIR = os.path.join(BASE_DIR, "il_models_clustered_v2")

# Output directory for SAC models
OUTPUT_DIR = os.path.join(BASE_DIR, "sac_models_multitask")

# Phase to IL model mapping
IL_MODEL_MAPPING = {
    0: "approach_and_balance",      # KickToTarget
    1: "obstacle_kicking",          # ObstaclePenalty
    2: "powerful_goalie_beater"     # GoaliePenalty
}

# Environment IDs
ENV_IDS = {
    0: "LowerT1KickToTarget-v0",
    1: "LowerT1ObstaclePenaltyKick-v0",
    2: "LowerT1GoaliePenaltyKick-v0"
}


# ==========================================
# VERIFICATION FUNCTION
# ==========================================

def verify_setup():
    """Verify all paths and models exist before training"""
    
    print("\n" + "="*80)
    print("SETUP VERIFICATION")
    print("="*80)
    
    all_good = True
    
    # Check base directory
    print(f"\n[1] Base Directory: {BASE_DIR}")
    if os.path.exists(BASE_DIR):
        print("    ✅ EXISTS")
    else:
        print("    ❌ NOT FOUND")
        all_good = False
    
    # Check IL models directory
    print(f"\n[2] IL Models Directory: {IL_MODELS_DIR}")
    if os.path.exists(IL_MODELS_DIR):
        print("    ✅ EXISTS")
    else:
        print("    ❌ NOT FOUND")
        all_good = False
    
    # Check each IL model
    print(f"\n[3] IL Models:")
    for task_id, phase_name in IL_MODEL_MAPPING.items():
        model_path = os.path.join(IL_MODELS_DIR, phase_name, f"{phase_name}_best.pt")
        env_name = ENV_IDS[task_id]
        
        print(f"\n    Task {task_id}: {env_name}")
        print(f"    Phase: {phase_name}")
        print(f"    Path: {model_path}")
        
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / 1024 / 1024
            print(f"    ✅ EXISTS ({file_size:.2f} MB)")
            
            # Try loading to verify integrity
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                print(f"    ✅ LOADABLE (loss: {checkpoint.get('loss', 'N/A'):.6f})")
            except Exception as e:
                print(f"    ⚠️  LOAD ERROR: {e}")
        else:
            print(f"    ❌ NOT FOUND")
            all_good = False
    
    # Check complete_generalization for polish phase
    print(f"\n[4] Polish Phase Model (complete_generalization):")
    polish_path = os.path.join(IL_MODELS_DIR, "complete_generalization", "complete_generalization_best.pt")
    print(f"    Path: {polish_path}")
    
    if os.path.exists(polish_path):
        file_size = os.path.getsize(polish_path) / 1024 / 1024
        print(f"    ✅ EXISTS ({file_size:.2f} MB)")
    else:
        print(f"    ⚠️  NOT FOUND (Polish phase will be skipped)")
    
    # Check output directory
    print(f"\n[5] Output Directory: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("    ✅ READY")
    
    # Summary
    print(f"\n{'='*80}")
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to train!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
    print("="*80 + "\n")
    
    return all_good


# ==========================================
# CLUSTERED IL MODEL
# ==========================================

class ClusteredImitationPolicy(nn.Module):
    """Cluster-aware IL policy"""
    
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
        
        cluster_label = torch.clamp(
            cluster_label, 
            min=0, 
            max=self.cluster_embedding.num_embeddings - 1
        )
        
        cluster_emb = self.cluster_embedding(cluster_label.squeeze(-1))
        x = torch.cat([state, cluster_emb], dim=-1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        
        action = self.tanh(self.fc_out(x))
        
        return action


def load_il_model(phase_name, device="cuda"):
    """Load IL model with correct path"""
    
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
            print(f"[ERROR] Missing dimensions in checkpoint for {phase_name}")
            return None
        
        model = ClusteredImitationPolicy(
            state_dim, action_dim, n_clusters=n_clusters
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"  ✅ Loaded: {phase_name}")
        print(f"     Loss: {checkpoint.get('loss', 'N/A'):.6f}")
        print(f"     Clusters: {n_clusters}")
        
        return model
    
    except Exception as e:
        print(f"[ERROR] Failed to load {phase_name}: {e}")
        return None


# ==========================================
# TASK PREPROCESSOR
# ==========================================

class TaskPreprocessor:
    """Adds task one-hot encoding to observations"""
    
    def __init__(self, n_tasks=3):
        self.n_tasks = n_tasks
    
    def get_task_onehot(self, task_id):
        onehot = np.zeros(self.n_tasks, dtype=np.float32)
        onehot[task_id] = 1.0
        return onehot
    
    def modify_state(self, obs, task_id):
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)
        
        task_onehot = self.get_task_onehot(task_id)
        task_onehot = np.expand_dims(task_onehot, axis=0)
        
        return np.hstack((obs, task_onehot))


class TaskConditionedEnv(gym.Wrapper):
    """Wraps environment with task ID and pads observations to uniform size"""
    
    def __init__(self, env, task_id, preprocessor, max_obs_dim=54):
        super().__init__(env)
        self.task_id = task_id
        self.preprocessor = preprocessor
        self.max_obs_dim = max_obs_dim  # Largest env observation (ObstaclePenalty = 54)
        
        # Get the original observation dimension
        self.original_obs_dim = env.observation_space.shape[0]
        
        # Final observation: max_obs_dim + n_tasks (for one-hot)
        final_dim = max_obs_dim + preprocessor.n_tasks
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(final_dim,),
            dtype=np.float32
        )
        
        print(f"      Task {task_id}: {self.original_obs_dim} → padded to {max_obs_dim} + {preprocessor.n_tasks} one-hot = {final_dim}")
    
    def _pad_observation(self, obs):
        """Pad observation to max_obs_dim, then add task one-hot"""
        # Ensure obs is 1D
        if len(obs.shape) > 1:
            obs = obs.squeeze()
        
        # Pad to max_obs_dim
        if len(obs) < self.max_obs_dim:
            padding = np.zeros(self.max_obs_dim - len(obs), dtype=np.float32)
            obs_padded = np.concatenate([obs, padding])
        else:
            obs_padded = obs
        
        # Add task one-hot
        task_onehot = self.preprocessor.get_task_onehot(self.task_id)
        
        return np.concatenate([obs_padded, task_onehot]).astype(np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self._pad_observation(obs)
        info['task_id'] = self.task_id
        info['original_obs_dim'] = self.original_obs_dim
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._pad_observation(obs)
        info['task_id'] = self.task_id
        info['original_obs_dim'] = self.original_obs_dim
        return obs, reward, terminated, truncated, info


# ==========================================
# REWARD WRAPPER
# ==========================================

class MultiTaskRewardWrapper(gym.Wrapper):
    """Environment-specific reward shaping with IL guidance"""
    
    def __init__(self, env, il_model, device="cuda", task_id=0):
        super().__init__(env)
        self.il_model = il_model
        self.device = device
        self.task_id = task_id
        
        # Original observation dimensions per task (from documentation)
        self.obs_dims = {
            0: 39,  # KickToTarget
            1: 54,  # ObstaclePenalty
            2: 45   # GoaliePenalty
        }
        self.original_obs_dim = self.obs_dims[task_id]
        
        self.kp = 50.0
        self.kd = 2.0
        
        self.prev_ball_distance = None
        self.prev_target_distance = None
        self.steps_survived = 0
        
        self.max_episode_steps = 400
        
        self._setup_reward_weights()
    
    def _setup_reward_weights(self):
        """Task-specific reward configuration"""
        
        if self.task_id == 0:  # KickToTarget (39 dims)
            self.env_reward_scale = 1.0
            self.il_weight = 0.15
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.55
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
            
        elif self.task_id == 1:  # ObstaclePenalty (54 dims)
            self.env_reward_scale = 1.0
            self.il_weight = 0.2
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.45
            self.obstacle_weight = 0.35
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
            
        elif self.task_id == 2:  # GoaliePenalty (45 dims)
            self.env_reward_scale = 1.0
            self.il_weight = 0.25
            self.approach_weight = 0.25
            self.balance_weight = 0.1
            self.target_weight = 0.35
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.5
            self.time_penalty_weight = 0.002
    
    def _get_raw_obs(self, obs):
        """Extract original observation (before padding and task one-hot)"""
        return obs[:self.original_obs_dim]
    
    def compute_il_bonus(self, obs):
        """IL guidance bonus"""
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
            bonus = np.clip(torque_magnitude * 0.1, 0.0, 3.0)
            
            return bonus
            
        except Exception:
            return 0.0
    
    def compute_approach_bonus(self, obs):
        """Approach ball bonus"""
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
        """Balance bonus"""
        try:
            obs_raw = self._get_raw_obs(obs)
            joint_pos = obs_raw[:12]
            balance_score = 1.0 - np.mean(np.abs(joint_pos)) * 1.2
            return np.clip(balance_score, 0, 1.0) * 5.0
        except Exception:
            return 0.0
    
    def compute_target_bonus(self, obs):
        """Target gradient bonus - task-specific indices"""
        try:
            obs_raw = self._get_raw_obs(obs)
            
            # Target position indices vary by environment
            if self.task_id == 0:  # KickToTarget: target at indices 33-35
                if len(obs_raw) < 36:
                    return 0.0
                target_pos = obs_raw[33:36]
                
            elif self.task_id == 1:  # ObstaclePenalty: target at indices 39-41
                if len(obs_raw) < 42:
                    return 0.0
                target_pos = obs_raw[39:42]
                
            elif self.task_id == 2:  # GoaliePenalty: goal at indices 33-35
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
        """Obstacle avoidance - only for Task 1 (ObstaclePenalty)"""
        if self.task_id != 1:
            return 0.0
        
        try:
            obs_raw = self._get_raw_obs(obs)
            
            # ObstaclePenalty: obstacles at indices 45-47, 48-50, 51-53
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
        """Ball velocity toward goal"""
        try:
            obs_raw = self._get_raw_obs(obs)
            ball_vel = obs_raw[27:30]  # Indices 27-29 for all envs
            
            # Get goal/target position based on task
            if self.task_id == 0:  # KickToTarget
                goal_pos = obs_raw[33:36]
            elif self.task_id == 1:  # ObstaclePenalty
                goal_pos = obs_raw[33:36]  # Goal position (not target)
            elif self.task_id == 2:  # GoaliePenalty
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
# MINIMAL ACTOR FOR SUBMISSION
# ==========================================

class MinimalActor(nn.Module):
    """Lightweight actor for <10MB submission"""
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
    
    def forward(self, obs):
        return self.network(obs)


# ==========================================
# CALLBACKS
# ==========================================

class PerformanceCallback(BaseCallback):
    """Training progress tracker"""
    
    def __init__(self, total_timesteps, check_freq=5000):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.check_freq = check_freq
        self.start_time = None
        self.last_log_time = None
    
    def _on_training_start(self):
        self.start_time = datetime.now()
        self.last_log_time = self.start_time
    
    def _on_step(self):
        if self.num_timesteps % self.check_freq == 0:
            now = datetime.now()
            elapsed = (now - self.start_time).total_seconds()
            since_last = (now - self.last_log_time).total_seconds()
            self.last_log_time = now
            
            progress = self.num_timesteps / self.total_timesteps * 100
            fps = self.num_timesteps / elapsed if elapsed > 0 else 0
            recent_fps = self.check_freq / since_last if since_last > 0 else 0
            
            ep_buffer = self.model.ep_info_buffer
            if len(ep_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in ep_buffer])
                mean_length = np.mean([ep['l'] for ep in ep_buffer])
            else:
                mean_reward = 0
                mean_length = 0
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated(0) / 1e9
            else:
                gpu_mem = 0
            
            if self.num_timesteps > 0:
                time_per_step = elapsed / self.num_timesteps
                remaining_steps = self.total_timesteps - self.num_timesteps
                eta = time_per_step * remaining_steps
            else:
                eta = 0
            
            print(f"\n{'='*80}")
            print(f"TRAINING PROGRESS")
            print(f"{'='*80}")
            print(f"Step: {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%)")
            print(f"FPS: {fps:.0f} (avg) | {recent_fps:.0f} (recent)")
            print(f"Mean Reward: {mean_reward:.2f} | Mean Length: {mean_length:.0f}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {gpu_mem:.2f} GB")
            print(f"Elapsed: {elapsed/3600:.2f}h | ETA: {eta/3600:.2f}h")
            print(f"{'='*80}\n")
        
        return True


class GracefulShutdownCallback(BaseCallback):
    """Save on Ctrl+C"""
    
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.requested = False
        signal.signal(signal.SIGINT, self._handler)
    
    def _handler(self, sig, frame):
        print("\n[SHUTDOWN] Ctrl+C detected - Saving model...")
        self.requested = True
    
    def _on_step(self):
        if self.requested:
            path = os.path.join(self.save_path, f"emergency_{self.num_timesteps}.zip")
            self.model.save(path)
            print(f"[SAVED] {path}")
            sys.exit(0)
        return True


# ==========================================
# MAIN TRAINER CLASS
# ==========================================

class RobustMultiTaskTrainer:
    """Complete multi-task training pipeline"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n" + "="*80)
        print("ROBUST MULTI-TASK SAC TRAINER")
        print("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        
        print(f"CPU: {os.cpu_count()} threads")
        print(f"Device: {self.device}")
        print(f"Base Dir: {BASE_DIR}")
        print(f"IL Models: {IL_MODELS_DIR}")
        print(f"Output: {OUTPUT_DIR}")
        print("="*80 + "\n")
    
    def load_all_il_models(self):
        """Load all IL models with verification"""
        
        print("[IL MODELS] Loading specialized IL models...\n")
        
        il_models = {}
        
        for task_id, phase_name in IL_MODEL_MAPPING.items():
            env_name = ENV_IDS[task_id]
            print(f"Task {task_id} ({env_name}):")
            
            model = load_il_model(phase_name, self.device)
            il_models[task_id] = model
            
            if model is None:
                print(f"     ⚠️  Will train without IL guidance for this task\n")
            else:
                print()
        
        return il_models
    
    def create_environments(self, il_models, n_envs_per_task=5):
        """Create training and evaluation environments"""
        
        preprocessor = TaskPreprocessor(n_tasks=3)
        
        print(f"[ENVIRONMENTS] Creating {n_envs_per_task * 3} training environments...\n")
        
        env_makers = []
        
        for task_id in range(3):
            env_id = ENV_IDS[task_id]
            il_model = il_models.get(task_id, None)
            
            print(f"  Task {task_id}: {env_id} (x{n_envs_per_task})")
            
            sai = SAIClient(env_id=env_id)
            
            for _ in range(n_envs_per_task):
                def make_env(task=task_id, sai_inst=sai, il_mod=il_model):
                    base_env = sai_inst.make_env()
                    
                    reward_wrapped = MultiTaskRewardWrapper(
                        base_env,
                        il_model=il_mod,
                        device=self.device,
                        task_id=task
                    )
                    
                    task_wrapped = TaskConditionedEnv(
                        reward_wrapped,
                        task_id=task,
                        preprocessor=preprocessor
                    )
                    
                    return Monitor(task_wrapped)
                
                env_makers.append(make_env)
        
        train_env = DummyVecEnv(env_makers)
        print(f"\n  ✅ Created {len(env_makers)} training environments")
        
        # Evaluation environments (1 per task)
        eval_makers = []
        for task_id in range(3):
            def make_eval(task=task_id):
                sai = SAIClient(env_id=ENV_IDS[task])
                base = sai.make_env()
                task_wrapped = TaskConditionedEnv(base, task, preprocessor)
                return Monitor(task_wrapped)
            eval_makers.append(make_eval)
        
        eval_env = DummyVecEnv(eval_makers)
        print(f"  ✅ Created {len(eval_makers)} evaluation environments\n")
        
        return train_env, eval_env
    
    def train(self, timesteps=25_000_000, n_envs_per_task=5):
        """Main training loop"""
        
        print("\n" + "="*80)
        print("STAGE 1: MULTI-TASK TRAINING WITH SPECIALIZED IL")
        print("="*80)
        print(f"Total Timesteps: {timesteps:,}")
        print(f"Environments per Task: {n_envs_per_task}")
        print(f"Total Parallel Envs: {n_envs_per_task * 3}")
        print("="*80 + "\n")
        
        # Load IL models
        il_models = self.load_all_il_models()
        
        # Create environments
        train_env, eval_env = self.create_environments(il_models, n_envs_per_task)
        
        # Create output directory
        stage1_dir = os.path.join(OUTPUT_DIR, "stage1_specialized")
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(os.path.join(stage1_dir, "checkpoints"), exist_ok=True)
        
        # Create SAC model
        print("[MODEL] Creating SAC with [256, 256, 128] architecture...\n")
        
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=50_000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "step"),
            gradient_steps=1,
            ent_coef=0.05,
            target_entropy='auto',
            target_update_interval=1,
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
                activation_fn=nn.ReLU
            ),
            verbose=1,
            device=self.device,
            tensorboard_log=os.path.join(stage1_dir, "logs")
        )
        
        # Callbacks
        n_total_envs = n_envs_per_task * 3
        
        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=100_000 // n_total_envs,
                save_path=os.path.join(stage1_dir, "checkpoints"),
                name_prefix="multitask",
                verbose=1
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=stage1_dir,
                log_path=stage1_dir,
                eval_freq=50_000 // n_total_envs,
                n_eval_episodes=9,
                deterministic=True,
                verbose=1
            ),
            PerformanceCallback(timesteps),
            GracefulShutdownCallback(stage1_dir)
        ])
        
        # Train
        print(f"[TRAINING] Starting {timesteps:,} timesteps...")
        print(f"[INFO] Press Ctrl+C to save and exit gracefully\n")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True
        )
        
        # Save final model
        final_path = os.path.join(stage1_dir, "final_model.zip")
        model.save(final_path)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("STAGE 1 COMPLETE!")
        print("="*80)
        print(f"Training Time: {training_time/3600:.2f} hours")
        print(f"Final Model: {final_path}")
        print(f"Best Model: {os.path.join(stage1_dir, 'best_model.zip')}")
        print("="*80 + "\n")
        
        train_env.close()
        eval_env.close()
        
        return os.path.join(stage1_dir, "best_model.zip")
    
    def train_polish(self, pretrained_path, extra_timesteps=10_000_000):
        """Stage 2: Polish with complete_generalization IL"""
        
        print("\n" + "="*80)
        print("STAGE 2: POLISH WITH UNIVERSAL EXPERT")
        print("="*80)
        print(f"Extra Timesteps: {extra_timesteps:,}")
        print(f"Pretrained Model: {pretrained_path}")
        print("="*80 + "\n")
        
        # Load universal IL model
        print("[IL] Loading complete_generalization model...")
        universal_il = load_il_model("complete_generalization", self.device)
        
        if universal_il is None:
            print("[ERROR] complete_generalization model not found!")
            print("[INFO] Skipping polish phase")
            return pretrained_path
        
        # Use same IL model for all tasks
        il_models = {0: universal_il, 1: universal_il, 2: universal_il}
        
        # Create environments
        train_env, eval_env = self.create_environments(il_models, n_envs_per_task=5)
        
        # Load pretrained model
        print(f"\n[MODEL] Loading pretrained model...")
        model = SAC.load(pretrained_path, env=train_env, device=self.device)
        model.learning_rate = 1e-4  # Lower LR for polish
        
        # Output directory
        polish_dir = os.path.join(OUTPUT_DIR, "stage2_polish")
        os.makedirs(polish_dir, exist_ok=True)
        os.makedirs(os.path.join(polish_dir, "checkpoints"), exist_ok=True)
        
        # Callbacks
        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=100_000 // 15,
                save_path=os.path.join(polish_dir, "checkpoints"),
                name_prefix="polish",
                verbose=1
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=polish_dir,
                log_path=polish_dir,
                eval_freq=50_000 // 15,
                n_eval_episodes=9,
                deterministic=True,
                verbose=1
            ),
            PerformanceCallback(extra_timesteps),
            GracefulShutdownCallback(polish_dir)
        ])
        
        # Continue training
        print(f"\n[TRAINING] Polish phase for {extra_timesteps:,} steps...\n")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=extra_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True
        )
        
        # Save final
        final_path = os.path.join(polish_dir, "final_competition_model.zip")
        model.save(final_path)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("STAGE 2 (POLISH) COMPLETE!")
        print("="*80)
        print(f"Training Time: {training_time/3600:.2f} hours")
        print(f"Competition Model: {final_path}")
        print("="*80 + "\n")
        
        train_env.close()
        eval_env.close()
        
        return os.path.join(polish_dir, "best_model.zip")


# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    """Main entry point"""
    
    print("\n" + "="*80)
    print("ROBUST MULTI-TASK SAC TRAINING PIPELINE")
    print("="*80)
    print("\nThis script will:")
    print("  1. Verify all paths and IL models")
    print("  2. Train Stage 1 (25M steps with specialized IL)")
    print("  3. Train Stage 2 (10M steps polish with universal IL)")
    print("  4. Create <10MB submission file")
    print("="*80 + "\n")
    
    # Step 1: Verify setup
    if not verify_setup():
        print("\n[ABORT] Please fix the issues above before continuing.")
        return
    
    # Confirm
    response = input("\nReady to start training? (yes/no): ")
    if response.lower() != "yes":
        print("Training cancelled.")
        return
    
    # Create trainer
    trainer = RobustMultiTaskTrainer()
    
    # Stage 1: Multi-task training
    best_model_path = trainer.train(
        timesteps=75_000_000,
        n_envs_per_task=5
    )
    
    # Ask about Stage 2
    print("\n" + "="*80)
    print("STAGE 1 COMPLETE!")
    print("="*80)
    print(f"\nBest model saved to: {best_model_path}")
    
    response = input("\nContinue to Stage 2 (Polish)? (yes/no): ")
    if response.lower() == "yes":
        final_model = trainer.train_polish(
            pretrained_path=best_model_path,
            extra_timesteps=75_000_000
        )
        print(f"\n✅ Final competition model: {final_model}")
    else:
        print(f"\n✅ Use this model for competition: {best_model_path}")
    
    print("\n🎉 Training complete!")


if __name__ == "__main__":
    main()
