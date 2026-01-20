import os
os.environ['SAI_API_KEY'] = 'sai_xaAEewt57gwAHfiP7vPG5MaxlXHFa3NG'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import json
from pathlib import Path

"""
================================================================================
COMPLETE IMITATION LEARNING TRAINING
================================================================================
This script trains IL models from the Booster Dataset with proper NPZ mapping
for each training phase. Each phase learns specific soccer skills that will
transfer to the corresponding SAC RL environment.

NPZ File Mapping:
-----------------
Phase 1 (Precision Kick) → LowerT1KickToTarget-v0
    Files: kick_ball1.npz, kick_ball2.npz, kick_ball3.npz, pass_ball1.npz
    Skills: Accurate ball placement, controlled kicking

Phase 2 (All Kicks) → LowerT1ObstaclePenaltyKick-v0
    Files: All kick files + soccer_drill_run.npz, jogging.npz
    Skills: Maneuvering, obstacle avoidance, varied kicks

Phase 3 (Powerful Kick) → LowerT1GoaliePenaltyKick-v0
    Files: powerful_kick.npz, goal_kick.npz, running.npz
    Skills: Maximum power kicks, dynamic movement

Phase 4 (Complete) → Scene Generalization
    Files: ALL .npz files
    Skills: Full repertoire for multi-task generalization
================================================================================
"""

class BoosterDataset(Dataset):
    """Enhanced dataset loader with file filtering"""
    
    def __init__(self, npz_folder, file_list=None):
        """
        Args:
            npz_folder: Path to booster_dataset folder
            file_list: Optional list of specific .npz files to load
                      If None, loads all files
        """
        self.data = []
        self.npz_folder = npz_folder
        self.loaded_files = []
        
        print(f"\n[LOADING] Booster Dataset from: {npz_folder}")
        
        # Get NPZ files
        if file_list:
            npz_files = [f for f in file_list if f.endswith('.npz')]
            print(f"[FILTER] Loading specific files: {len(npz_files)} files")
        else:
            npz_files = [f for f in os.listdir(npz_folder) if f.endswith('.npz')]
            print(f"[ALL] Loading all files: {len(npz_files)} files")
        
        total_pairs = 0
        
        for npz_file in npz_files:
            filepath = os.path.join(npz_folder, npz_file)
            
            if not os.path.exists(filepath):
                print(f"[SKIP] File not found: {npz_file}")
                continue
            
            try:
                data = np.load(filepath, allow_pickle=True)
                
                qpos = data['qpos']  # Joint positions
                qvel = data['qvel']  # Joint velocities
                
                # Create state-action pairs
                # State: current positions and velocities
                # Action: position delta (velocity target)
                pairs_from_file = 0
                
                for i in range(len(qpos) - 1):
                    state = np.concatenate([qpos[i], qvel[i]])
                    action = qpos[i + 1] - qpos[i]
                    
                    self.data.append((state, action))
                    pairs_from_file += 1
                
                self.loaded_files.append(npz_file)
                total_pairs += pairs_from_file
                print(f"  ✓ {npz_file}: {pairs_from_file:,} pairs")
                
            except Exception as e:
                print(f"[ERROR] Failed to load {npz_file}: {e}")
        
        print(f"\n[LOADED] {total_pairs:,} state-action pairs from {len(self.loaded_files)} files")
        print(f"[FILES] {', '.join(self.loaded_files)}\n")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action = self.data[idx]
        return torch.FloatTensor(state), torch.FloatTensor(action)
    
    def get_loaded_files(self):
        return self.loaded_files


class ImitationPolicy(nn.Module):
    """
    Imitation Learning Policy Network
    
    Architecture: 4-layer feedforward with ReLU activations
    - Input: Robot state (qpos + qvel)
    - Hidden: [512, 512, 256, 256] for rich feature extraction
    - Output: Action deltas (Tanh normalized)
    
    This architecture is designed to:
    1. Capture complex motion patterns from expert demonstrations
    2. Generalize to similar but unseen situations
    3. Provide good initialization for RL fine-tuning
    """
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Prevent overfitting
            
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, action_dim),
            nn.Tanh()  # Normalized actions
        )
    
    def forward(self, state):
        return self.network(state)


class ILTrainer:
    """Complete Imitation Learning Trainer with Phase Support"""
    
    def __init__(self, dataset_path, output_dir="il_models_complete"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Hardware info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"\n[GPU] {gpu_name} ({gpu_mem:.1f}GB VRAM)")
        
        print(f"[CPU] {os.cpu_count()} threads available")
        print(f"[DEVICE] Training on: {self.device}\n")
    
    def train_phase(self, phase_name, file_list, epochs=150, batch_size=256, lr=3e-4):
        """
        Train IL model for a specific phase
        
        Args:
            phase_name: Name of the phase (e.g., "precision_kick")
            file_list: List of .npz files to use for this phase
            epochs: Number of training epochs
            batch_size: Batch size (optimized for RTX 3090 Ti)
            lr: Learning rate
        
        Returns:
            Trained model
        """
        
        print("\n" + "="*80)
        print(f"PHASE: {phase_name.upper()}")
        print("="*80)
        print(f"Files: {', '.join(file_list)}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Learning Rate: {lr}")
        print("="*80 + "\n")
        
        # Create phase output directory
        phase_dir = os.path.join(self.output_dir, phase_name)
        os.makedirs(phase_dir, exist_ok=True)
        
        # Load dataset for this phase
        dataset = BoosterDataset(self.dataset_path, file_list=file_list)
        
        if len(dataset) == 0:
            print(f"[ERROR] No data loaded for {phase_name}")
            return None
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True  # Faster GPU transfer
        )
        
        # Get dimensions
        sample_state, sample_action = dataset[0]
        state_dim = len(sample_state)
        action_dim = len(sample_action)
        
        print(f"[DIMS] State: {state_dim}, Action: {action_dim}")
        print(f"[DATA] {len(dataset):,} training samples")
        print(f"[DATA] {len(dataloader):,} batches per epoch\n")
        
        # Create model
        model = ImitationPolicy(state_dim, action_dim).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        criterion = nn.MSELoss()
        
        # Training metrics
        best_loss = float('inf')
        train_losses = []
        epoch_times = []
        
        print(f"[TRAINING] Starting {phase_name}...")
        print(f"  Target: Minimize action prediction error")
        print(f"  Press Ctrl+C to save and exit\n")
        
        start_time = datetime.now()
        
        try:
            for epoch in range(epochs):
                epoch_start = datetime.now()
                model.train()
                
                epoch_loss = 0
                num_batches = 0
                
                for states, actions in dataloader:
                    states = states.to(self.device)
                    actions = actions.to(self.device)
                    
                    # Forward
                    predicted_actions = model(states)
                    loss = criterion(predicted_actions, actions)
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_loss = epoch_loss / num_batches
                train_losses.append(avg_loss)
                
                # Update learning rate
                scheduler.step(avg_loss)
                
                # Timing
                epoch_time = (datetime.now() - epoch_start).total_seconds()
                epoch_times.append(epoch_time)
                avg_epoch_time = np.mean(epoch_times[-10:])
                
                # Print progress every 10 epochs
                if (epoch + 1) % 10 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    eta = avg_epoch_time * (epochs - epoch - 1)
                    
                    print(f"Epoch {epoch+1}/{epochs}")
                    print(f"  Loss: {avg_loss:.6f} (best: {best_loss:.6f})")
                    print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
                    print(f"  Time: {epoch_time:.1f}s (avg: {avg_epoch_time:.1f}s)")
                    print(f"  Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m\n")
                
                # Save best model
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
                        'phase': phase_name,
                        'files': dataset.get_loaded_files(),
                        'train_losses': train_losses
                    }, best_path)
                
                # Checkpoint every 50 epochs
                if (epoch + 1) % 50 == 0:
                    checkpoint_path = os.path.join(
                        phase_dir, 
                        f"{phase_name}_epoch_{epoch+1}.pt"
                    )
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_loss,
                        'state_dim': state_dim,
                        'action_dim': action_dim,
                        'phase': phase_name,
                        'files': dataset.get_loaded_files()
                    }, checkpoint_path)
                    print(f"  [CHECKPOINT] Saved at epoch {epoch+1}\n")
        
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving emergency checkpoint...")
            emergency_path = os.path.join(phase_dir, f"{phase_name}_emergency.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'state_dim': state_dim,
                'action_dim': action_dim,
                'phase': phase_name,
                'files': dataset.get_loaded_files()
            }, emergency_path)
            print(f"[SAVED] {emergency_path}")
            return model
        
        # Save final model
        final_path = os.path.join(phase_dir, f"{phase_name}_final.pt")
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'state_dim': state_dim,
            'action_dim': action_dim,
            'phase': phase_name,
            'files': dataset.get_loaded_files(),
            'train_losses': train_losses
        }, final_path)
        
        # Training summary
        total_time = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "="*80)
        print(f"PHASE {phase_name.upper()} COMPLETE")
        print("="*80)
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Final Loss: {avg_loss:.6f}")
        print(f"Training Time: {total_time/60:.1f} minutes")
        print(f"Best Model: {best_path}")
        print(f"Final Model: {final_path}")
        print("="*80 + "\n")
        
        return model
    
    def train_all_phases(self):
        """Train all phases with proper curriculum learning"""
        
        print("\n" + "="*80)
        print("CURRICULUM LEARNING: FUNDAMENTALS → ADVANCED SOCCER SKILLS")
        print("="*80)
        
        phases = [
            {
                'name': 'locomotion_foundation',
                'files': ['walking.npz', 'jogging.npz', 'running.npz'],
                'target_env': 'ALL (Foundation)',
                'description': 'Stable bipedal locomotion',
                'epochs': 200,
                'lr': 3e-4
            },
            {
                'name': 'ball_interaction',
                'files': ['soccer_drill_run.npz', 'pass_ball1.npz', 'walking.npz'],
                'target_env': 'LowerT1KickToTarget-v0',
                'description': 'Approach ball, controlled contact',
                'epochs': 150,
                'lr': 3e-4
            },
            {
                'name': 'varied_kicking',
                'files': ['kick_ball1.npz', 'kick_ball2.npz', 'kick_ball3.npz', 
                          'soccer_drill_run.npz', 'jogging.npz'],
                'target_env': 'LowerT1ObstaclePenaltyKick-v0',
                'description': 'Different kick angles for obstacles',
                'epochs': 150,
                'lr': 2e-4
            },
            {
                'name': 'powerful_striking',
                'files': ['powerful_kick.npz', 'goal_kick.npz', 'running.npz', 
                          'kick_ball1.npz'],
                'target_env': 'LowerT1GoaliePenaltyKick-v0',
                'description': 'Maximum power to beat goalie',
                'epochs': 150,
                'lr': 2e-4
            },
            {
                'name': 'complete_integration',
                'files': ['walking.npz', 'jogging.npz', 'running.npz',
                          'soccer_drill_run.npz', 'pass_ball1.npz',
                          'kick_ball1.npz', 'kick_ball2.npz', 'kick_ball3.npz',
                          'powerful_kick.npz', 'goal_kick.npz'],
                'target_env': 'Scene Generalization',
                'description': 'Full skill repertoire',
                'epochs': 200,
                'lr': 1e-4
            }
        ]
        
        trained_models = {}
        
        for i, phase in enumerate(phases, 1):
            print(f"\n{'='*80}")
            print(f"PHASE {i}/5: {phase['name'].upper()}")
            print(f"{'='*80}")
            print(f"Target: {phase['target_env']}")
            print(f"Description: {phase['description']}")
            print(f"Files: {', '.join(phase['files'])}")
            print(f"Epochs: {phase['epochs']}, LR: {phase['lr']}")
            print(f"{'='*80}\n")
            
            input(f"Press Enter to start Phase {i}/5...")
            
            model = self.train_phase(
                phase_name=phase['name'],
                file_list=phase['files'],
                epochs=phase['epochs'],
                batch_size=256,
                lr=phase['lr']
            )
            
            if model:
                trained_models[phase['name']] = os.path.join(
                    self.output_dir, 
                    phase['name'], 
                    f"{phase['name']}_best.pt"
                )
                print(f"\n✅ Phase {i}/5 complete!\n")
        
        print("\n" + "="*80)
        print("CURRICULUM LEARNING COMPLETE")
        print("="*80)
        for phase_name, model_path in trained_models.items():
            print(f"{phase_name:25s} → {model_path}")
        print("="*80 + "\n")
        
        return trained_models


def main():
    """Main execution"""
    
    # Dataset location
    dataset_path = r"C:\Users\ulyss\RoboAthletes\booster_soccer_showdown\booster_dataset\soccer\booster_t1"
    
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset not found at: {dataset_path}")
        print("Please check the path and try again.")
        return
    
    # Create trainer
    trainer = ILTrainer(dataset_path)
    
    # Train all phases
    trainer.train_all_phases()


if __name__ == "__main__":
    main()