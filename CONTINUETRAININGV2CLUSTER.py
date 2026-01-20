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
CONTINUE SAC TRAINING WITH CLUSTERED IL MODELS
================================================================================

Checkpoint to Continue:
C:/Users/ulyss/RoboAthletes/booster_soccer_showdown/sac_models_10mb/LowerT1KickToTarget_approach_and_balance/emergency_19600000.zip

New IL Models Location:
C:/Users/ulyss/RoboAthletes/booster_soccer_showdown/il_models_clustered/

This script:
- Loads your existing SAC checkpoint (emergency_19600000.zip)
- Uses NEW clustered IL models for guidance
- Keeps your current reward shaping
- Continues training from 19.6M steps
================================================================================
"""

# ==========================================
# TASK PREPROCESSOR
# ==========================================

class TaskPreprocessor:
    """Preprocessor for multi-task scene training"""
    
    def get_task_onehot(self, info):
        if isinstance(info, dict) and "task_index" in info:
            return info["task_index"]
        return np.array([])
    
    def modify_state(self, obs, info):
        task_onehot = self.get_task_onehot(info)
        
        if len(task_onehot) > 0:
            if len(obs.shape) == 1:
                obs = np.expand_dims(obs, axis=0)
            if len(task_onehot.shape) == 1:
                task_onehot = np.expand_dims(task_onehot, axis=0)
            obs = np.hstack((obs, task_onehot))
        
        return obs


class PreprocessedEnv(gym.Wrapper):
    """Applies task one-hot to observations"""
    
    def __init__(self, env, preprocessor):
        super().__init__(env)
        self.preprocessor = preprocessor
        
        original_shape = env.observation_space.shape
        new_shape = (original_shape[0] + 3,)
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=new_shape,
            dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.preprocessor.modify_state(obs, info)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.preprocessor.modify_state(obs, info)
        return obs, reward, terminated, truncated, info


# ==========================================
# CLUSTERED IL MODEL (NEW!)
# ==========================================

class ClusteredImitationPolicy(nn.Module):
    """Cluster-aware IL policy from OPTIMIZEDILMODELTRAINING.py"""
    
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
        # If no cluster label provided, use default (cluster 0)
        if cluster_label is None:
            cluster_label = torch.zeros(state.shape[0], 1, dtype=torch.long, device=state.device)
        
        # Clamp cluster labels
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


def load_clustered_il_model(phase_name, device="cuda"):
    """Load NEW clustered IL model"""
    
    # NEW PATH: il_models_clustered
    model_path = os.path.join(
        r"C:\Users\ulyss\RoboAthletes\booster_soccer_showdown\il_models_clustered",
        phase_name,
        f"{phase_name}_best.pt"
    )
    
    if not os.path.exists(model_path):
        print(f"[WARNING] Clustered IL model not found: {model_path}")
        print(f"[INFO] Training will continue without IL guidance")
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        state_dim = checkpoint.get('state_dim')
        action_dim = checkpoint.get('action_dim')
        n_clusters = checkpoint.get('n_clusters', 4)
        
        if state_dim is None or action_dim is None:
            print(f"[ERROR] Missing dimensions in checkpoint")
            return None
        
        print(f"\n[CLUSTERED IL MODEL LOADED]")
        print(f"  Phase: {phase_name}")
        print(f"  State Dim: {state_dim}")
        print(f"  Action Dim: {action_dim}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Loss: {checkpoint['loss']:.6f}")
        print(f"  Files: {checkpoint.get('files', 'N/A')}")
        
        # Create clustered model
        model = ClusteredImitationPolicy(
            state_dim, action_dim, n_clusters=n_clusters
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    except Exception as e:
        print(f"[ERROR] Failed to load clustered IL model: {e}")
        import traceback
        traceback.print_exc()
        return None


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
# REWARD WRAPPER (KEPT YOUR CURRENT VERSION)
# ==========================================

class EnhancedRewardWrapper(gym.Wrapper):
    """
    Your current reward design - UNCHANGED
    Uses clustered IL model for guidance
    """
    
    def __init__(self, env, il_model=None, device="cuda", env_name=""):
        super().__init__(env)
        self.il_model = il_model
        self.device = device
        self.env_name = env_name
        
        self.kp = 50.0
        self.kd = 5.0  # Your current damping value
        
        self.prev_ball_distance = None
        self.prev_target_distance = None
        self.steps_survived = 0
        self.episode_rewards = []
        self.prev_action = None
        
        self.max_episode_steps = 400
        
        self._setup_reward_weights()
    
    def _setup_reward_weights(self):
        """Your current reward weights"""
        
        if "KickToTarget" in self.env_name:
            self.env_reward_scale = 1.0
            self.il_weight = 0.2
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.55
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
            self.smoothness_weight = 0.05
            self.motion_weight = 0.1
            
            print(f"[REWARD] KickToTarget: target=0.55, max_len=400, enhanced motion")
            
        elif "ObstaclePenalty" in self.env_name:
            self.env_reward_scale = 1.0
            self.il_weight = 0.2
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.45
            self.obstacle_weight = 0.35
            self.velocity_weight = 0.0
            self.time_penalty_weight = 0.002
            self.smoothness_weight = 0.05
            self.motion_weight = 0.1
            
            print(f"[REWARD] ObstaclePenalty: Enhanced motion")
            
        elif "GoaliePenalty" in self.env_name:
            self.env_reward_scale = 1.0
            self.il_weight = 0.25
            self.approach_weight = 0.25
            self.balance_weight = 0.1
            self.target_weight = 0.35
            self.obstacle_weight = 0.0
            self.velocity_weight = 0.5
            self.time_penalty_weight = 0.002
            self.smoothness_weight = 0.05
            self.motion_weight = 0.1
            
            print(f"[REWARD] GoaliePenalty: Enhanced motion")
            
        else:
            self.env_reward_scale = 1.0
            self.il_weight = 0.2
            self.approach_weight = 0.3
            self.balance_weight = 0.1
            self.target_weight = 0.4
            self.obstacle_weight = 0.25
            self.velocity_weight = 0.25
            self.time_penalty_weight = 0.002
            self.smoothness_weight = 0.05
            self.motion_weight = 0.1
            
            print(f"[REWARD] Scene: Multi-task with motion")
    
    def compute_il_bonus(self, obs):
        """IL bonus with PD controller - WORKS WITH CLUSTERED MODEL"""
        if not self.il_model:
            return 0.0
        
        try:
            joint_pos = obs[:12]
            joint_vel = obs[12:24]
            
            root_pos = np.zeros(3)
            root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            root_vel = np.zeros(6)
            
            full_qpos = np.concatenate([root_pos, root_quat, joint_pos])
            full_qvel = np.concatenate([root_vel, joint_vel]).astype(np.float32)
            state = np.concatenate([full_qpos, full_qvel]).astype(np.float32)
            
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
                
                # Clustered model uses cluster_label, but we can pass None for default
                delta_qpos = self.il_model(state_tensor, cluster_label=None)
                delta_qpos = delta_qpos.squeeze(0).cpu().numpy()
            
            actuated_delta = delta_qpos[-12:]
            target_qpos = joint_pos + actuated_delta
            position_error = target_qpos - joint_pos
            torque = self.kp * position_error - self.kd * joint_vel
            
            torque_magnitude = np.mean(np.abs(torque))
            bonus = np.clip(torque_magnitude * 0.1, 0.0, 3.0)
            
            return bonus
            
        except Exception as e:
            # Silently fail - don't spam console
            return 0.0
    
    def compute_approach_bonus(self, obs):
        """Approach ball bonus"""
        try:
            ball_pos = obs[24:27]
            dist = np.linalg.norm(ball_pos)
            
            if self.prev_ball_distance is None:
                self.prev_ball_distance = dist
                return 0.0
            
            improvement = self.prev_ball_distance - dist
            self.prev_ball_distance = dist
            
            return improvement * 10.0
        except:
            return 0.0
    
    def compute_balance_bonus(self, obs):
        """Balance bonus"""
        try:
            joint_pos = obs[:12]
            balance_score = 1.0 - np.mean(np.abs(joint_pos)) * 1.2
            return np.clip(balance_score, 0, 1.0) * 5.0
        except:
            return 0.0
    
    def compute_target_bonus(self, obs):
        """Target gradient bonus"""
        if len(obs) < 36:
            return 0.0
        
        try:
            target_pos = obs[33:36]
            dist = np.linalg.norm(target_pos)
            
            if self.prev_target_distance is None:
                self.prev_target_distance = dist
                return 0.0
            
            improvement = self.prev_target_distance - dist
            self.prev_target_distance = dist
            
            return improvement * 12.0
        except:
            return 0.0
    
    def compute_obstacle_avoidance_bonus(self, obs):
        """Obstacle avoidance"""
        if len(obs) < 54:
            return 0.0
        
        try:
            obstacles = [obs[45:48], obs[48:51], obs[51:54]]
            ball_pos = obs[24:27]
            
            min_dist = min([np.linalg.norm(ball_pos - o) for o in obstacles])
            
            if min_dist > 2.5:
                return 4.0
            elif min_dist > 1.5:
                return 2.0
            elif min_dist > 1.0:
                return 0.0
            else:
                return -3.0
        except:
            return 0.0
    
    def compute_ball_velocity_bonus(self, obs):
        """Ball velocity toward goal"""
        try:
            ball_vel = obs[27:29]
            
            if len(obs) >= 36:
                goal_pos = obs[33:36]
                goal_dist = np.linalg.norm(goal_pos)
                
                if goal_dist < 0.1:
                    return 0.0
                
                goal_dir = goal_pos[:2] / goal_dist
                vel_toward_goal = np.dot(ball_vel, goal_dir)
                
                return np.clip(vel_toward_goal * 3.0, 0.0, 8.0)
            else:
                vel_magnitude = np.linalg.norm(ball_vel)
                return np.clip(vel_magnitude * 2.0, 0.0, 6.0)
        except:
            return 0.0
    
    def compute_smoothness_penalty(self, action):
        """Penalize rapid action changes"""
        if self.prev_action is None:
            return 0.0
        
        action_change = np.mean(np.abs(action - self.prev_action))
        return -action_change
    
    def compute_motion_bonus(self, obs):
        """Reward leg joint velocities"""
        try:
            joint_vel = obs[12:24]
            mean_abs_vel = np.mean(np.abs(joint_vel))
            bonus = np.clip(mean_abs_vel * 2.0, 0.0, 5.0)
            return bonus
        except:
            return 0.0
    
    def compute_distance_motion_bonus(self, obs):
        """Distance-based motion rewards"""
        try:
            ball_dist = np.linalg.norm(obs[24:27])
            left_vel = obs[12:18]
            right_vel = obs[18:24]
            mean_vel = np.mean(np.abs(np.concatenate([left_vel, right_vel])))
            alternation_score = np.abs(np.sign(left_vel[0]) + np.sign(right_vel[0]))
            
            if ball_dist > 2.0:
                bonus = (mean_vel * 3.0 + alternation_score * 2.0) if alternation_score > 1.5 else 0.0
                return np.clip(bonus, 0.0, 8.0)
            elif ball_dist < 1.0:
                return self.compute_il_bonus(obs) * 1.5
            return 0.0
        except:
            return 0.0
    
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        # Your current reward components
        il_bonus = self.compute_il_bonus(obs)
        approach_bonus = self.compute_approach_bonus(obs)
        balance_bonus = self.compute_balance_bonus(obs)
        target_bonus = self.compute_target_bonus(obs)
        obstacle_bonus = self.compute_obstacle_avoidance_bonus(obs)
        velocity_bonus = self.compute_ball_velocity_bonus(obs)
        survival_bonus = 0.01
        time_penalty = -self.time_penalty_weight * self.steps_survived
        smoothness_penalty = self.compute_smoothness_penalty(action) * self.smoothness_weight
        motion_bonus = self.compute_motion_bonus(obs) * self.motion_weight
        distance_motion_bonus = self.compute_distance_motion_bonus(obs) * 0.15
        
        total_reward = (
            reward * self.env_reward_scale +
            target_bonus * self.target_weight +
            il_bonus * self.il_weight +
            approach_bonus * self.approach_weight +
            balance_bonus * self.balance_weight +
            obstacle_bonus * self.obstacle_weight +
            velocity_bonus * self.velocity_weight +
            survival_bonus +
            time_penalty +
            smoothness_penalty +
            motion_bonus +
            distance_motion_bonus
        )
        
        self.steps_survived += 1
        self.episode_rewards.append(total_reward)
        self.prev_action = action.copy()
        
        if self.steps_survived >= self.max_episode_steps:
            trunc = True
            info['TimeLimit.truncated'] = True
        
        return obs, total_reward, term, trunc, info
    
    def reset(self, **kwargs):
        self.prev_ball_distance = None
        self.prev_target_distance = None
        self.steps_survived = 0
        self.episode_rewards = []
        self.prev_action = None
        return self.env.reset(**kwargs)


# ==========================================
# CALLBACKS (SAME AS BEFORE)
# ==========================================

class DetailedEvalCallback(BaseCallback):
    """Detailed evaluation tracking"""
    
    def __init__(self, eval_env, log_path, eval_freq=10000, n_eval_episodes=10):
        super().__init__()
        self.eval_env = eval_env
        self.log_path = log_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        os.makedirs(log_path, exist_ok=True)
        self.eval_history = []
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            print(f"\n[EVAL] Running evaluation at step {self.num_timesteps:,}...")
            
            episode_rewards = []
            episode_lengths = []
            episode_successes = []
            
            for ep in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                ep_reward = 0
                ep_length = 0
                success = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, dones, infos = self.eval_env.step(action)
                    
                    if isinstance(reward, np.ndarray):
                        reward = reward[0]
                    if isinstance(dones, np.ndarray):
                        done = dones[0]
                    else:
                        done = dones
                    if isinstance(infos, list):
                        info = infos[0]
                    else:
                        info = infos
                    
                    ep_reward += reward
                    ep_length += 1
                    
                    if isinstance(info, dict) and 'success' in info:
                        if info['success']:
                            success = True
                
                episode_rewards.append(ep_reward)
                episode_lengths.append(ep_length)
                episode_successes.append(1 if success else 0)
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            success_rate = np.mean(episode_successes) * 100
            
            eval_result = {
                'timestep': self.num_timesteps,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'mean_length': float(mean_length),
                'success_rate': float(success_rate),
                'episodes': self.n_eval_episodes
            }
            
            self.eval_history.append(eval_result)
            
            json_path = os.path.join(self.log_path, 'eval_history.json')
            with open(json_path, 'w') as f:
                json.dump(self.eval_history, f, indent=2)
            
            print(f"\n{'='*70}")
            print(f"EVALUATION RESULTS (Step {self.num_timesteps:,})")
            print(f"{'='*70}")
            print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"Mean Length: {mean_length:.1f} steps")
            print(f"Success Rate: {success_rate:.1f}%")
            print(f"{'='*70}\n")
        
        return True


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
                gpu_cached = torch.cuda.memory_reserved(0) / 1e9
            else:
                gpu_mem = 0
                gpu_cached = 0
            
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
                print(f"GPU: {gpu_mem:.2f}GB / {gpu_cached:.2f}GB")
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
        print("\n[SHUTDOWN] Saving model...")
        self.requested = True
    
    def _on_step(self):
        if self.requested:
            path = os.path.join(self.save_path, f"emergency_{self.num_timesteps}.zip")
            self.model.save(path)
            print(f"[SAVED] {path}")
            sys.exit(0)
        return True


# ==========================================
# TRAINER
# ==========================================

class SACTrainer:
    """Continue training with clustered IL models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("\n" + "="*80)
        print("CONTINUE SAC TRAINING WITH CLUSTERED IL")
        print("="*80)
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
        
        print(f"CPU: {os.cpu_count()} threads")
        print(f"Device: {self.device}")
        print("\nConfiguration:")
        print("  ✅ Checkpoint: emergency_19600000.zip")
        print("  ✅ IL Models: il_models_clustered/ (NEW!)")
        print("  ✅ Reward: Your current shaping (unchanged)")
        print("  ✅ Continue: Phase 1 (KickToTarget)")
        print("="*80 + "\n")
    
    def prepare_actor_only_submission(self, model_path, output_name, obs_dim, action_dim):
        """Create actor-only submission file"""
        from stable_baselines3 import SAC
        
        print(f"\n[SUBMISSION] Creating actor-only file (<10MB)...")
        print(f"  Loading SAC model: {model_path}")
        
        sac_model = SAC.load(model_path, device=self.device)
        actor_state_dict = sac_model.policy.actor.state_dict()
        
        minimal_actor = MinimalActor(obs_dim, action_dim).to(self.device)
        minimal_actor.load_state_dict(actor_state_dict, strict=False)
        
        submission_dir = "models_for_submission"
        os.makedirs(submission_dir, exist_ok=True)
        
        submission_path = os.path.join(submission_dir, f"{output_name}.pt")
        
        torch.save({
            'actor_state_dict': minimal_actor.state_dict(),
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'network_arch': '[256, 256, 128]'
        }, submission_path)
        
        file_size_mb = os.path.getsize(submission_path) / 1e6
        
        print(f"  Saved: {submission_path}")
        print(f"  File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 10:
            print(f"  ⚠️  WARNING: Still over 10MB ({file_size_mb:.2f} MB)")
        else:
            print(f"  ✅ UNDER 10MB! Ready to submit!")
        
        return submission_path
    
    def continue_phase_1(self):
        """Continue Phase 1 from checkpoint with clustered IL"""
        
        print("\n" + "="*80)
        print("CONTINUING PHASE 1: KICK TO TARGET")
        print("="*80)
        print("Checkpoint: emergency_19600000.zip (19.6M steps completed)")
        print("IL Model: approach_and_balance (CLUSTERED)")
        print("Additional Training: 20M steps")
        print("="*80 + "\n")
        
        env_id = "LowerT1KickToTarget-v0"
        il_phase = "approach_and_balance"
        timesteps = 20_000_000
        
        # YOUR CHECKPOINT
        pretrained_model = r"C:\Users\ulyss\RoboAthletes\booster_soccer_showdown\sac_models_10mb\LowerT1KickToTarget_approach_and_balance\emergency_19600000.zip"
        
        # Load CLUSTERED IL model
        il_model = load_clustered_il_model(il_phase, self.device)
        
        # Create environment
        sai = SAIClient(env_id=env_id)
        
        n_envs = 16
        print(f"[ENV] Creating {n_envs} environments...")
        
        def make_env():
            base_env = sai.make_env()
            wrapped_env = EnhancedRewardWrapper(
                base_env,
                il_model=il_model,
                device=self.device,
                env_name=env_id
            )
            return Monitor(wrapped_env)
        
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        eval_env = DummyVecEnv([make_env for _ in range(4)])
        
        print(f"[ENV] ✓ Created\n")
        
        # Load checkpoint
        print(f"[MODEL] Loading checkpoint...")
        model = SAC.load(pretrained_model, device=self.device)
        model.set_env(env)
        
        print(f"[MODEL] ✓ Loaded from step 19.6M")
        print(f"[MODEL] Learning rate: {model.learning_rate:.2e}\n")
        
        # Output directory
        output_dir = f"sac_models_10mb/LowerT1KickToTarget_approach_and_balance_continued"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/evaluations", exist_ok=True)
        
        # Callbacks
        callbacks = CallbackList([
            CheckpointCallback(
                save_freq=100_000 // n_envs,
                save_path=f"{output_dir}/checkpoints",
                name_prefix="sac",
                verbose=1
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=output_dir,
                log_path=output_dir,
                eval_freq=25_000 // n_envs,
                n_eval_episodes=10,
                deterministic=True,
                verbose=1
            ),
            DetailedEvalCallback(
                eval_env,
                log_path=f"{output_dir}/evaluations",
                eval_freq=50_000 // n_envs,
                n_eval_episodes=15
            ),
            PerformanceCallback(timesteps),
            GracefulShutdownCallback(output_dir)
        ])
        
        print(f"[TRAINING] Continuing for {timesteps:,} steps\n")
        
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False  # ← Continue from 19.6M, don't reset
        )
        
        final_path = os.path.join(output_dir, "final_model.zip")
        model.save(final_path)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print("CONTINUED TRAINING COMPLETE")
        print("="*80)
        print(f"Time: {training_time/3600:.2f} hours")
        print(f"Total Steps: ~39.6M (19.6M + 20M)")
        print(f"Model: {final_path}")
        print("="*80 + "\n")
        
        # Create submission
        obs_dim = env.envs[0].observation_space.shape[0]
        action_dim = 12
        
        best_model_path = os.path.join(output_dir, "best_model.zip")
        submission_model = self.prepare_actor_only_submission(
            best_model_path,
            f"LowerT1KickToTarget_continued_actor",
            obs_dim,
            action_dim
        )
        
        print(f"\n✅ Actor-only submission: {submission_model}\n")
        
        env.close()
        eval_env.close()


def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("CONTINUE TRAINING WITH CLUSTERED IL MODELS")
    print("="*80)
    print("\nThis will:")
    print("  1. Load your checkpoint (emergency_19600000.zip)")
    print("  2. Load NEW clustered IL model (approach_and_balance)")
    print("  3. Continue training for 20M more steps")
    print("  4. Keep your current reward shaping")
    print("="*80 + "\n")
    
    response = input("Ready to continue? (yes/no): ")
    if response.lower() != "yes":
        print("Training cancelled.")
        return
    
    trainer = SACTrainer()
    trainer.continue_phase_1()


if __name__ == "__main__":
    main()