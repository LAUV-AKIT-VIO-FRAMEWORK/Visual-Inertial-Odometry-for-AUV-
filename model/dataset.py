import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class VIOLogDataset(Dataset):
    """Dataset for VIO log data with ground truth for training only"""
    
    def __init__(self, log_path, window=10, mode='train', transform=None):
        """
        Args:
            log_path: Path to CSV log file
            window: Number of timesteps in context window
            mode: 'train', 'val', or 'test'
            transform: Optional transform to apply
        """
        self.df = pd.read_csv(log_path, on_bad_lines='skip')
        self.window = window
        self.mode = mode
        self.transform = transform
        
        # Clean data
        self._clean_data()
        
        # Compute targets using ground truth (only for training)
        self._compute_targets_from_gt()
        
        # Prepare features
        self._prepare_features()
        
        # Normalize features (fit on training data only)
        self.scalers = {}
        if mode == 'train':
            self._fit_normalizers()
            
    def _clean_data(self):
        """Clean and preprocess the raw data"""
        # Replace infinite values with NaN
        self.df = self.df.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill NaN values in feature columns
        feature_cols = ['ax_v', 'ay_v', 'az_v', 'vx_v', 'vy_v', 'vz_v',
                       'num_matches', 's_trace', 's_cond', 'h_norm', 
                       'accel_mag', 'accel_var', 'baseline_ratio']
        
        for col in feature_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(method='ffill').fillna(0)
        
        # Encode status
        status_map = {
            'IDLE': 0,
            'INSUFFICIENT_TRIANGULATION': 1,
            'SMALL_BASELINE': 2,
            'too_few_points': 3,
            'INVALID_FRAME_PAIR': 4
        }
        self.df['status_code'] = self.df['status'].map(status_map).fillna(0)
        
    def _compute_targets_from_gt(self):
        """Compute optimal noise targets using ground truth"""
        df = self.df
        
        # ----- Process Noise Target (Q) -----
        # Based on velocity prediction error
        if all(col in df.columns for col in ['vx_gt', 'vy_gt', 'vz_gt']):
            # Velocity error (ground truth vs estimate)
            vel_error = np.sqrt(
                (df['vx_v'] - df['vx_gt'])**2 + 
                (df['vy_v'] - df['vy_gt'])**2 + 
                (df['vz_v'] - df['vz_gt'])**2
            )
            
            # Acceleration magnitude as dynamics indicator
            accel_mag = np.sqrt(df['ax_v']**2 + df['ay_v']**2 + df['az_v']**2)
            
            # Combine: higher Q needed when velocity error is high during motion
            self.q_raw = vel_error * (1 + 0.5 * (accel_mag > 1.0).astype(float))
        else:
            # Fallback if GT velocities not available
            print("Warning: GT velocities not found, using acceleration-based Q")
            self.q_raw = np.sqrt(df['ax_v']**2 + df['ay_v']**2 + df['az_v']**2)
        
        # ----- Measurement Noise Target (R) -----
        # Based on position error when visual measurements are used
        if all(col in df.columns for col in ['px_gt', 'py_gt', 'pz_gt']):
            # Position error
            pos_error = np.sqrt(
                (df['px_v'] - df['px_gt'])**2 + 
                (df['py_v'] - df['py_gt'])**2 + 
                (df['pz_v'] - df['pz_gt'])**2
            )
            
            # Scale by visual quality: higher R when tracking is poor
            visual_quality = df['num_matches'].values / 50.0
            tracking_failure = (df['status_code'] > 0).astype(float)
            
            self.r_raw = pos_error * (1 + 5 * tracking_failure) * (2 - visual_quality)
        else:
            # Fallback using innovation norm if available
            print("Warning: GT positions not found, using innovation-based R")
            innov_norm = df['innov_norm'].values
            if innov_norm.max() > 0:
                self.r_raw = innov_norm
            else:
                self.r_raw = np.ones(len(df)) * 0.1
        
        # Log transform for better distribution
        self.q_target = np.log1p(self.q_raw)
        self.r_target = np.log1p(self.r_raw)
        
        # Store statistics for denormalization
        if self.mode == 'train':
            self.q_mean = self.q_target.mean()
            self.q_std = self.q_target.std() + 1e-6
            self.r_mean = self.r_target.mean()
            self.r_std = self.r_target.std() + 1e-6
            
            # Normalize targets
            self.q_target = (self.q_target - self.q_mean) / self.q_std
            self.r_target = (self.r_target - self.r_mean) / self.r_std
            
    def _prepare_features(self):
        """Prepare features that will be available during inference"""
        df = self.df
        
        # Temporal context features (available during inference)
        self.context_cols = [
            'ax_v', 'ay_v', 'az_v',  # acceleration
            'vx_v', 'vy_v', 'vz_v',   # velocity
            'moving_flag'              # derived from velocity
        ]
        
        # Create moving flag (no GT needed)
        speed = np.sqrt(df['vx_v']**2 + df['vy_v']**2 + df['vz_v']**2)
        df['moving_flag'] = (speed > 0.1).astype(float)
        
        # Define which set features are continuous (need normalization) vs binary
        self.set_continuous_cols = [
            'accel_mag',           # acceleration magnitude
            'accel_var',            # acceleration variance
            'num_matches_norm',     # normalized match count
            's_trace_log',          # log of covariance trace
            's_cond_log',           # log of condition number
            'h_norm_log',           # log of H matrix norm
            'baseline_ratio'        # baseline ratio
        ]
        
        self.set_binary_cols = [
            'insufficient_triang',  # binary status indicators
            'small_baseline',
            'too_few_points',
            'is_zupt'
        ]
        
        # Combined set features
        self.set_cols = self.set_continuous_cols + self.set_binary_cols
        
        # Create derived features
        df['num_matches_norm'] = df['num_matches'] / 50.0
        df['s_trace_log'] = np.log1p(df['s_trace'])
        df['s_cond_log'] = np.log1p(df['s_cond'])
        df['h_norm_log'] = np.log1p(df['h_norm'])
        
        # Binary status flags
        df['insufficient_triang'] = (df['status'] == 'INSUFFICIENT_TRIANGULATION').astype(float)
        df['small_baseline'] = (df['status'] == 'SMALL_BASELINE').astype(float)
        df['too_few_points'] = (df['status'] == 'too_few_points').astype(float)
        
    def _fit_normalizers(self):
        """Fit normalization parameters on training data"""
        df = self.df
        
        # Context normalizers (exclude moving_flag which is binary)
        self.ctx_scaler = StandardScaler()
        ctx_data = df[self.context_cols[:-1]].values  # exclude moving_flag
        self.ctx_scaler.fit(ctx_data)
        
        # Set normalizers for continuous features only
        self.set_scaler = StandardScaler()
        set_continuous_data = df[self.set_continuous_cols].values
        self.set_scaler.fit(set_continuous_data)
        
    def normalize_features(self, ctx_data, set_data):
        """Normalize features using fitted scalers
        
        Args:
            ctx_data: numpy array of shape [window_size, n_context_features]
            set_data: numpy array of shape [window_size, n_set_features]
        
        Returns:
            Normalized ctx_data and set_data
        """
        # Normalize context
        if hasattr(self, 'ctx_scaler'):
            # ctx_data shape: [window_size, features]
            # Separate moving_flag (last column) from other features
            moving_flag = ctx_data[:, -1:]  # Keep as 2D with shape [window_size, 1]
            ctx_features = ctx_data[:, :-1]  # Shape: [window_size, features-1]
            
            # Normalize features (excluding moving_flag)
            ctx_features_norm = self.ctx_scaler.transform(ctx_features)
            
            # Recombine with moving_flag
            ctx_data = np.concatenate([ctx_features_norm, moving_flag], axis=1)
        
        # Normalize set features - only continuous part
        if hasattr(self, 'set_scaler'):
            # Split set_data into continuous and binary parts
            n_continuous = len(self.set_continuous_cols)
            set_continuous = set_data[:, :n_continuous]
            set_binary = set_data[:, n_continuous:]
            
            # Normalize continuous features
            set_continuous_norm = self.set_scaler.transform(set_continuous)
            
            # Recombine with binary features
            set_data = np.concatenate([set_continuous_norm, set_binary], axis=1)
            
        return ctx_data, set_data
        
    def __len__(self):
        return max(0, len(self.df) - self.window)
    
    def __getitem__(self, idx):
        # Get window of data
        rows = self.df.iloc[idx:idx+self.window]
        
        # 1. Temporal Context (available during inference)
        ctx = rows[self.context_cols].values.astype(np.float32)  # Shape: [window, n_context_features]
        
        # 2. Set Features (available during inference)
        mset = rows[self.set_cols].values.astype(np.float32)  # Shape: [window, n_set_features]
        
        # Handle any remaining NaNs
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=0.0, neginf=0.0)
        mset = np.nan_to_num(mset, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if scalers are available
        if hasattr(self, 'ctx_scaler') or hasattr(self, 'set_scaler'):
            ctx, mset = self.normalize_features(ctx, mset)
        
        # 3. Targets (only used during training, from GT)
        targets = np.array([
            self.q_target[idx+self.window-1],
            self.r_target[idx+self.window-1]
        ], dtype=np.float32)
        
        # Convert to tensors
        ctx_tensor = torch.from_numpy(ctx)
        mset_tensor = torch.from_numpy(mset)
        targets_tensor = torch.from_numpy(targets)
        
        if self.transform:
            ctx_tensor = self.transform(ctx_tensor)
            mset_tensor = self.transform(mset_tensor)
            
        return ctx_tensor, mset_tensor, targets_tensor
    
    def get_normalization_stats(self):
        """Return normalization statistics for inference"""
        return {
            'q_mean': self.q_mean if hasattr(self, 'q_mean') else 0,
            'q_std': self.q_std if hasattr(self, 'q_std') else 1,
            'r_mean': self.r_mean if hasattr(self, 'r_mean') else 0,
            'r_std': self.r_std if hasattr(self, 'r_std') else 1
        }
