import os
os.environ['SAI_API_KEY'] = 'sai_xaAEewt57gwAHfiP7vPG5MaxlXHFa3NG'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path

"""
================================================================================
CORRECTED CLUSTERED IL TRAINING - WITH PROPER LOCOMOTION
================================================================================

Dataset Location: 
C:/Users/ulyss/RoboAthletes/booster_soccer_showdown/booster_dataset/soccer/booster_lower_t1

CRITICAL FIX:
Phase 1 (approach_and_balance) NOW includes:
  ✅ walking.npz   (stable approach)
  ✅ jogging.npz   (medium-speed approach)
  ✅ running.npz   (fast approach) - OPTIONAL

This fixes the "stuttering/inching" problem by teaching actual locomotion!
================================================================================
"""


class TrajectoryCluster:
    """Clusters robot trajectories by motion characteristics"""
    
    def __init__(self, n_clusters=5, min_clusters=2):
        self.desired_n_clusters = n_clusters
        self.min_clusters = min_clusters
        self.actual_n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        
        self.cluster_names = {
            0: "Idle/Standing",
            1: "Walking Approach",
            2: "Execute Kick", 
            3: "Balance Recovery",
            4: "Jogging/Running"
        }
    
    def extract_trajectory_features(self, qpos, qvel):
        """Extract motion characteristics from a trajectory"""
        
        avg_velocity = np.mean(np.linalg.norm(qvel, axis=1))
        max_velocity = np.max(np.linalg.norm(qvel, axis=1))
        
        acceleration = np.diff(qvel, axis=0)
        avg_acceleration = np.mean(np.linalg.norm(acceleration, axis=1))
        
        jerk = np.diff(acceleration, axis=0)
        avg_jerk = np.mean(np.linalg.norm(jerk, axis=1))
        
        position_change = np.diff(qpos, axis=0)
        avg_position_change = np.mean(np.linalg.norm(position_change, axis=1))
        
        duration = len(qpos)
        velocity_variance = np.var(np.linalg.norm(qvel, axis=1))
        
        features = np.array([
            avg_velocity,
            max_velocity,
            avg_acceleration,
            avg_jerk,
            avg_position_change,
            duration,
            velocity_variance
        ])
        
        return features
    
    def cluster_trajectories(self, trajectory_list):
        """Cluster trajectories adaptively"""
        
        n_trajectories = len(trajectory_list)
        
        print(f"\n[CLUSTERING] Analyzing {n_trajectories} trajectories...")
        
        # Adaptive clustering
        self.actual_n_clusters = min(self.desired_n_clusters, n_trajectories)
        self.actual_n_clusters = max(self.min_clusters, self.actual_n_clusters)
        
        if self.actual_n_clusters != self.desired_n_clusters:
            print(f"[ADAPTIVE] Adjusting clusters: {self.desired_n_clusters} → {self.actual_n_clusters}")
            print(f"  Reason: Only {n_trajectories} trajectories available\n")
        
        self.kmeans = KMeans(n_clusters=self.actual_n_clusters, random_state=42, n_init=10)
        
        # Extract features
        features = []
        for qpos, qvel, _ in trajectory_list:
            traj_features = self.extract_trajectory_features(qpos, qvel)
            features.append(traj_features)
        
        features = np.array(features)
        features_scaled = self.scaler.fit_transform(features)
        
        # Cluster
        cluster_assignments = self.kmeans.fit_predict(features_scaled)
        
        # Statistics
        cluster_stats = {}
        for cluster_id in range(self.actual_n_clusters):
            mask = cluster_assignments == cluster_id
            count = np.sum(mask)
            cluster_features = features[mask]
            
            cluster_stats[cluster_id] = {
                'name': self.cluster_names.get(cluster_id, f"Motion Type {cluster_id}"),
                'count': count,
                'avg_velocity': np.mean(cluster_features[:, 0]),
                'avg_acceleration': np.mean(cluster_features[:, 2]),
                'avg_duration': np.mean(cluster_features[:, 5])
            }
        
        # Print summary
        print(f"{'='*70}")
        print(f"TRAJECTORY CLUSTERING RESULTS ({self.actual_n_clusters} clusters)")
        print(f"{'='*70}")
        for cluster_id, stats in cluster_stats.items():
            print(f"\nCluster {cluster_id}: {stats['name']}")
            print(f"  Count: {stats['count']} trajectories")
            print(f"  Avg Velocity: {stats['avg_velocity']:.3f}")
            print(f"  Avg Acceleration: {stats['avg_acceleration']:.3f}")
            print(f"  Avg Duration: {stats['avg_duration']:.1f} timesteps")
        print(f"{'='*70}\n")
        
        return cluster_assignments, cluster_stats


class ClusteredBoosterDataset(Dataset):
    """Dataset with trajectory clustering and chunking"""
    
    def __init__(self, npz_folder, file_list=None, chunk_length=32):
        self.trajectories = []
        self.cluster_labels = []
        self.chunk_length = chunk_length
        
        print(f"\n[LOADING] Clustered Booster Dataset")
        print(f"  Dataset Path: {npz_folder}")
        print(f"  Chunk Length: {chunk_length} timesteps\n")
        
        # Get NPZ files
        if file_list:
            npz_files = file_list
        else:
            npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
        
        # Load trajectories
        trajectory_list = []
        for npz_file in npz_files:
            # Handle both full paths and filenames
            if os.path.isabs(npz_file):
                filepath = npz_file
            else:
                filepath = os.path.join(npz_folder, npz_file)
            
            if not os.path.exists(filepath):
                print(f"  ⚠️  {os.path.basename(npz_file)}: NOT FOUND (skipping)")
                continue
            
            try:
                data = np.load(filepath, allow_pickle=True)
                qpos = data['qpos']
                qvel = data['qvel']
                
                trajectory_list.append((qpos, qvel, os.path.basename(npz_file)))
                print(f"  ✓ {os.path.basename(npz_file)}: {len(qpos)} timesteps")
                
            except Exception as e:
                print(f"  ❌ {os.path.basename(npz_file)}: ERROR - {e}")
        
        if len(trajectory_list) == 0:
            print("\n[ERROR] No trajectories loaded!")
            self.chunks = []
            self.chunk_clusters = []
            self.data = []
            self.labels = []
            return
        
        # Cluster trajectories
        clusterer = TrajectoryCluster(n_clusters=5)
        cluster_assignments, cluster_stats = clusterer.cluster_trajectories(trajectory_list)
        
        # Create chunked sequences
        self.chunks = []
        self.chunk_clusters = []
        
        for idx, (qpos, qvel, filename) in enumerate(trajectory_list):
            cluster_id = cluster_assignments[idx]
            
            # Split into overlapping chunks
            for start_idx in range(0, len(qpos) - chunk_length, chunk_length // 2):
                end_idx = start_idx + chunk_length
                
                if end_idx > len(qpos):
                    break
                
                chunk_qpos = qpos[start_idx:end_idx]
                chunk_qvel = qvel[start_idx:end_idx]
                
                chunk_data = []
                for t in range(len(chunk_qpos) - 1):
                    state = np.concatenate([chunk_qpos[t], chunk_qvel[t]])
                    action = chunk_qpos[t + 1] - chunk_qpos[t]
                    chunk_data.append((state, action))
                
                self.chunks.append(chunk_data)
                self.chunk_clusters.append(cluster_id)
        
        print(f"\n[CHUNKED] {len(self.chunks)} motion sequences created")
        print(f"[INFO] Each sequence: {chunk_length} timesteps")
        
        # Flatten for dataset
        self.data = []
        self.labels = []
        for chunk, cluster_id in zip(self.chunks, self.chunk_clusters):
            for state, action in chunk:
                self.data.append((state, action))
                self.labels.append(cluster_id)
        
        print(f"[TOTAL] {len(self.data):,} state-action pairs\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]
        cluster_label = self.labels[idx]
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.LongTensor([cluster_label])
        )
    
    def get_cluster_distribution(self):
        """Returns distribution of samples across clusters"""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


class ClusteredImitationPolicy(nn.Module):
    """Policy network with cluster-aware learning"""
    
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
        
    def forward(self, state, cluster_label):
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


class ClusteredILTrainer:
    """Trainer for cluster-aware imitation learning"""
    
    def __init__(self, dataset_path, output_dir="il_models_clustered_v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Hardware info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n[GPU] {gpu_name} ({gpu_mem:.1f}GB)")
        
        print(f"[CPU] {os.cpu_count()} threads")
        print(f"[DEVICE] {self.device}\n")
        
        print("="*80)
        print("CORRECTED CLUSTERED IL TRAINING - WITH LOCOMOTION")
        print("="*80)
        print("✅ Phase 1 NOW includes walking.npz + jogging.npz")
        print("✅ Fixes stuttering/inching behavior")
        print("✅ Teaches proper approach locomotion")
        print("="*80 + "\n")
    
    def train_phase(self, phase_name, file_list, epochs=150, batch_size=256, lr=3e-4):
        """Train clustered IL model"""
        
        print(f"\n{'='*80}")
        print(f"PHASE: {phase_name.upper()}")
        print(f"{'='*80}\n")
        
        phase_dir = os.path.join(self.output_dir, phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        
        # Load dataset
        dataset = ClusteredBoosterDataset(
            self.dataset_path,
            file_list=file_list,
            chunk_length=32
        )
        
        if len(dataset) == 0:
            print(f"[ERROR] No data loaded for {phase_name}")
            return None
        
        # Cluster distribution
        cluster_dist = dataset.get_cluster_distribution()
        print(f"[CLUSTERS] Sample distribution:")
        for cluster_id, count in cluster_dist.items():
            print(f"  Cluster {cluster_id}: {count:,} samples ({count/len(dataset)*100:.1f}%)")
        print()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Get dimensions
        sample_state, sample_action, _ = dataset[0]
        state_dim = len(sample_state)
        action_dim = len(sample_action)
        n_clusters = len(cluster_dist)
        
        print(f"[MODEL] State: {state_dim}, Action: {action_dim}, Clusters: {n_clusters}")
        print(f"[DATA] {len(dataset):,} samples in {len(dataloader):,} batches\n")
        
        # Create model
        model = ClusteredImitationPolicy(
            state_dim, action_dim, n_clusters=n_clusters
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        train_losses = []
        
        print(f"[TRAINING] {phase_name} for {epochs} epochs\n")
        
        start_time = datetime.now()
        
        try:
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                num_batches = 0
                
                for states, actions, cluster_labels in dataloader:
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    cluster_labels = cluster_labels.to(self.device)
                    
                    predicted_actions = model(states, cluster_labels)
                    loss = criterion(predicted_actions, actions)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                scheduler.step(avg_loss)
                
                if (epoch + 1) % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Loss: {avg_loss:.6f} (best: {best_loss:.6f})")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"  Time: {elapsed/60:.1f}m\n")
                
                # Save best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_path = os.path.join(phase_dir, f"{phase_name}_best.pt")
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss,
                        'state_dim': state_dim,
                        'action_dim': action_dim,
                        'n_clusters': n_clusters,
                        'phase': phase_name,
                        'files': file_list,
                        'cluster_distribution': cluster_dist
                    }, best_path)
                
                if (epoch + 1) % 50 == 0:
                    checkpoint_path = os.path.join(phase_dir, f"{phase_name}_epoch_{epoch+1}.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': avg_loss,
                        'state_dim': state_dim,
                        'action_dim': action_dim,
                        'n_clusters': n_clusters
                    }, checkpoint_path)
                    print(f"  [CHECKPOINT] Epoch {epoch+1}\n")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving...")
            emergency_path = os.path.join(phase_dir, f"{phase_name}_emergency.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': avg_loss,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'n_clusters': n_clusters
            }, emergency_path)
            return model
        
        # Save final
        final_path = os.path.join(phase_dir, f"{phase_name}_final.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'loss': avg_loss,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'n_clusters': n_clusters,
            'train_losses': train_losses
        }, final_path)
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"PHASE {phase_name.upper()} COMPLETE")
        print(f"{'='*80}")
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Best Model: {best_path}")
        print(f"{'='*80}\n")
        
        return model
    
    def train_all_phases(self):
        """Train all phases with CORRECTED mappings"""
        
        print("\n" + "="*80)
        print("CORRECTED ENVIRONMENT-OPTIMIZED IL TRAINING")
        print("="*80)
        print("✅ Fixed approach_and_balance to include locomotion!")
        print("="*80 + "\n")
        
        phases = [
            {
                'name': 'approach_and_balance',
                'target_env': 'LowerT1KickToTarget-v0',
                'description': 'Accurate kicks with proper approach',
                'requirements': 'LOCOMOTION, Precision, Balance, Controlled Power',
                'obs_dims': 39,
                'files': [
                    # ⭐⭐ LOCOMOTION (CRITICAL - NOW INCLUDED!)
                    'walking.npz',         # Stable, slow approach
                    'jogging.npz',         # Medium-speed approach
                    
                    # PRECISION KICKING
                    'pass_ball1.npz',      # Controlled passing (accuracy)
                    'kick_ball1.npz',      # Standard kick technique
                    'kick_ball2.npz',      # Kick variations
                    'kick_ball3.npz'       # Additional kick styles
                ]
            },
            {
                'name': 'obstacle_kicking',
                'target_env': 'LowerT1ObstaclePenaltyKick-v0',
                'description': 'Navigate 3 obstacles then score',
                'requirements': 'Maneuvering, Ball Control, Repositioning',
                'obs_dims': 54,
                'files': [
                    'soccer_drill_run.npz', # ⭐⭐ CRITICAL: Complex navigation
                    'kick_ball1.npz',       # Kick after positioning
                    'kick_ball2.npz',       # Varied kick angles
                    'kick_ball3.npz',       # More kick options
                    'jogging.npz',          # ⭐ Movement between obstacles
                    'walking.npz'           # Careful positioning
                ]
            },
            {
                'name': 'powerful_goalie_beater',
                'target_env': 'LowerT1GoaliePenaltyKick-v0',
                'description': 'Score against moving goalkeeper',
                'requirements': 'Maximum Power, Speed, Dynamic Timing',
                'obs_dims': 45,
                'files': [
                    'powerful_kick.npz',    # ⭐⭐ Maximum power kicks
                    'goal_kick.npz',        # ⭐⭐ Actual goal-scoring
                    'kick_ball1.npz',       # Standard reference
                    'running.npz'           # ⭐ Fast approach
                ]
            },
            {
                'name': 'complete_generalization',
                'target_env': 'Scene (All 3 Tasks)',
                'description': 'Master all soccer scenarios',
                'requirements': 'Full Soccer Skill Repertoire',
                'obs_dims': '39/45/54 + 3 (task one-hot)',
                'files': [
                    # Core kicking
                    'kick_ball1.npz',
                    'kick_ball2.npz',
                    'kick_ball3.npz',
                    'pass_ball1.npz',
                    
                    # Power skills
                    'powerful_kick.npz',
                    'goal_kick.npz',
                    
                    # Movement skills
                    'soccer_drill_run.npz',
                    'running.npz',
                    'jogging.npz',
                    'walking.npz'
                ]
            }
        ]
        
        trained_models = {}
        
        for i, phase in enumerate(phases, 1):
            print(f"\n{'='*80}")
            print(f"PHASE {i}/4: {phase['name'].upper()}")
            print(f"{'='*80}")
            print(f"Target Environment: {phase['target_env']}")
            print(f"Description: {phase['description']}")
            print(f"Requirements: {phase['requirements']}")
            print(f"Observation Space: {phase['obs_dims']} dims")
            print(f"\nNPZ Files ({len(phase['files'])}):")
            for file in phase['files']:
                print(f"  • {file}")
            print(f"{'='*80}\n")
            
            input(f"Press Enter to start Phase {i}...")
            
            model = self.train_phase(
                phase_name=phase['name'],
                file_list=phase['files'],
                epochs=150,
                batch_size=256,
                lr=3e-4
            )
            
            if model:
                trained_models[phase['name']] = os.path.join(
                    self.output_dir,
                    phase['name'],
                    f"{phase['name']}_best.pt"
                )
                print(f"\n✅ Phase {i}/4 complete!")
                print(f"   Trained on {len(phase['files'])} motion files")
                print(f"   Ready for {phase['target_env']}\n")
        
        print(f"\n{'='*80}")
        print("ALL IL PHASES COMPLETE!")
        print(f"{'='*80}")
        print("\nTrained Models by Environment:")
        print("-" * 80)
        for i, phase in enumerate(phases, 1):
            if phase['name'] in trained_models:
                print(f"\n{i}. {phase['target_env']}")
                print(f"   Phase: {phase['name']}")
                print(f"   Requirements: {phase['requirements']}")
                print(f"   Model: {trained_models[phase['name']]}")
        print(f"\n{'='*80}\n")
        
        print("🚀 Next Step: Run multi-task SAC training!")
        print("   Use: MULTI_TASK_SAC_WITH_CLUSTERED_IL.py\n")
        
        return trained_models


def main():
    """Main execution"""
    
    # Dataset path
    dataset_path = r"C:\Users\ulyss\RoboAthletes\booster_soccer_showdown\booster_dataset\soccer\booster_lower_t1"
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("\nPlease verify the path and try again.")
        return
    
    # List available files
    print("\n" + "="*80)
    print("DATASET VERIFICATION")
    print("="*80)
    print(f"Path: {dataset_path}\n")
    
    npz_files = [f for f in os.listdir(dataset_path) if f.endswith('.npz')]
    
    if len(npz_files) == 0:
        print("[ERROR] No NPZ files found in dataset!")
        return
    
    print(f"Found {len(npz_files)} NPZ files:")
    for file in sorted(npz_files):
        filepath = os.path.join(dataset_path, file)
        file_size = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {file:<25s} ({file_size:.1f} KB)")
    
    print("="*80 + "\n")
    
    # Create trainer (NOTE: Output to new directory!)
    trainer = ClusteredILTrainer(
        dataset_path, 
        output_dir="il_models_clustered_v2"  # ← NEW VERSION!
    )
    
    # Train all phases
    trainer.train_all_phases()


if __name__ == "__main__":
    main()