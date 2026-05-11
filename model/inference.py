import torch
import numpy as np
from .set_transformer import AKITTransformer

class AKITPredictor:
    """Wrapper for inference with trained AKIT model"""
    
    def __init__(self, model_path, device='cpu'):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get config from checkpoint
        if 'config' in checkpoint:
            config = checkpoint['config']
        else:
            # Default config if not saved
            config = {
                'hidden_dim': 128,
                'num_inducing': 32,
                'num_heads': 4
            }
        
        # Initialize model
        self.model = AKITTransformer(
            context_dim=7,      # [ax, ay, az, vx, vy, vz, moving_flag]
            set_dim=11,         # 11 set features
            hidden_dim=config['hidden_dim'],
            num_inducing=config['num_inducing'],
            num_heads=config['num_heads'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Store normalization stats if available
        self.stats = checkpoint.get('stats', {
            'q_mean': 0, 'q_std': 1, 'r_mean': 0, 'r_std': 1
        })
        
        print(f"✅ AKIT Model loaded from {model_path}")
        print(f"   Q stats - mean: {self.stats['q_mean']:.3f}, std: {self.stats['q_std']:.3f}")
        print(f"   R stats - mean: {self.stats['r_mean']:.3f}, std: {self.stats['r_std']:.3f}")
        
    def prepare_context(self, imu_data, window_size=10):
        """
        Prepare context window from IMU buffer
        
        Args:
            imu_data: List of IMU measurements, each [ax, ay, az, vx, vy, vz]
            window_size: Size of context window
        
        Returns:
            context: numpy array [window_size, 7]
        """
        if len(imu_data) < window_size:
            # Pad with zeros if not enough data
            pad_size = window_size - len(imu_data)
            padding = [imu_data[0]] * pad_size if imu_data else [np.zeros(6)] * pad_size
            imu_data = padding + imu_data
        
        # Take last window_size samples
        recent = imu_data[-window_size:]
        
        context = []
        for data in recent:
            # Compute moving flag (speed > 0.1 m/s)
            speed = np.linalg.norm(data[3:6])  # vx, vy, vz
            moving_flag = 1.0 if speed > 0.1 else 0.0
            
            context.append([
                data[0], data[1], data[2],  # ax, ay, az
                data[3], data[4], data[5],  # vx, vy, vz
                moving_flag
            ])
        
        return np.array(context, dtype=np.float32)
    
    def prepare_set(self, metrics):
        """
        Prepare set features from current metrics
        
        Args:
            metrics: Dictionary containing:
                - accel_mag: acceleration magnitude
                - accel_var: acceleration variance
                - num_matches: number of feature matches
                - s_trace: covariance trace
                - s_cond: covariance condition number
                - h_norm: H matrix norm
                - status: tracking status string
                - baseline_ratio: baseline ratio
                - is_zupt: zero velocity update flag
        
        Returns:
            mset: numpy array [1, 11] (single measurement)
        """
        # Continuous features
        num_matches_norm = metrics.get('num_matches', 0) / 50.0
        s_trace_log = np.log1p(metrics.get('s_trace', 0))
        s_cond_log = np.log1p(metrics.get('s_cond', 0))
        h_norm_log = np.log1p(metrics.get('h_norm', 0))
        
        # Binary status flags
        status = metrics.get('status', '')
        insufficient_triang = 1.0 if status == 'INSUFFICIENT_TRIANGULATION' else 0.0
        small_baseline = 1.0 if status == 'SMALL_BASELINE' else 0.0
        too_few_points = 1.0 if status == 'too_few_points' else 0.0
        
        mset = np.array([[
            metrics.get('accel_mag', 0),
            metrics.get('accel_var', 0),
            num_matches_norm,
            s_trace_log,
            s_cond_log,
            h_norm_log,
            insufficient_triang,
            small_baseline,
            too_few_points,
            metrics.get('baseline_ratio', 0),
            metrics.get('is_zupt', 0)
        ]], dtype=np.float32)
        
        return mset
    
    def predict(self, context, mset):
        """
        Predict noise scales
        
        Args:
            context: [window_size, 7] numpy array
            mset: [1, 11] numpy array
        
        Returns:
            q_scales: [3] array [gyro, accel, bias] scales
            r_scale: float measurement noise scale
        """
        # Add batch dimension if needed
        if len(context.shape) == 2:
            context = context[np.newaxis, :, :]
        if len(mset.shape) == 2:
            mset = mset[np.newaxis, :, :]
        
        # Convert to tensors
        ctx_tensor = torch.FloatTensor(context).to(self.device)
        mset_tensor = torch.FloatTensor(mset).to(self.device)
        
        with torch.no_grad():
            q_scales, r_scale = self.model(ctx_tensor, mset_tensor)
        
        # Convert to numpy
        q_scales = q_scales.squeeze().cpu().numpy()
        r_scale = r_scale.squeeze().cpu().numpy()
        
        # Clamp to reasonable ranges
        q_scales = np.clip(q_scales, 0.1, 10.0)
        r_scale = np.clip(r_scale, 0.1, 100.0)
        
        return q_scales, r_scale
    
    def predict_from_buffer(self, imu_buffer, current_metrics):
        """
        Convenience method to predict from buffers
        
        Args:
            imu_buffer: List of IMU measurements
            current_metrics: Dictionary of current visual metrics
        
        Returns:
            q_scales, r_scale
        """
        context = self.prepare_context(imu_buffer)
        mset = self.prepare_set(current_metrics)
        return self.predict(context, mset)
