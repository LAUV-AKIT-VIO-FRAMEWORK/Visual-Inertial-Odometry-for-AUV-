import numpy as np
import torch
import cv2
import rospy
import os
from datetime import datetime
from sensor_msgs.msg import Imu, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R_tool
import time
from scipy.stats import chi2

from model.set_transformer.set_transformer import AKITTransformer
from model.set_transformer.inference import AKITPredictor
from HDF5_Logger.vio_hdf5_logger import VIOHDF5Logger
from model.visual_pipeline import VisualPipeline
from model.keypoint_extractor import HybridKeypointExtractor
from model.ekf.ekf_se3 import EKFSE3, State, NoiseParams
from utils.camera_intrinsics import load_camera_intrinsics
from utils.geometry import match_orb_enhanced, robust_triangulate, score_triangulated_point
from gazebo_msgs.msg import ModelStates


class GazeboVIONode:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
        np.random.seed(42)  # Fixed seed for reproducibility
        torch.manual_seed(42)
        # Camera intrinsics
        self.K_cam = np.array([
            [407.0646, 0.0, 384.5],
            [0.0, 407.0646, 246.5],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
 
        print(f"Camera intrinsics:")
        print(f"  fx={self.K_cam[0,0]}, fy={self.K_cam[1,1]}")
        print(f"  cx={self.K_cam[0,2]}, cy={self.K_cam[1,2]}")
       
        # Camera extrinsics - FIXED
        self.R_bc, self.t_bc = self.get_vio_extrinsics_correct()
        print("\n=== FINAL CAMERA CONFIGURATION ===")
        print(f"R_bc:\n{self.R_bc}")
        print(f"t_bc: {self.t_bc}")
       
        # Test forward vector
        test_forward = np.array([1, 0, 0])
        in_camera = self.R_bc @ test_forward
        print(f"Body forward -> Camera: {in_camera}")
        print(f"Camera Z (forward) component: {in_camera[2]:.3f}")
       
        self.verify_extrinsics_consistency()
        
        # ===== ADAPTIVE MODEL LOADING =====
        self.window_size = 10
        self.ctx_buf = []
        self.mset_buf = []
        self.scale_history = []
        self.y_error_history = []
        
        # Old Q model (keep for fallback)
        self.q_model = None
        self.use_ml_q = False
        
        # ===== NEW: AKIT Model =====
        self.akit_predictor = None
        self.use_akit = False
        
        # Try to load old Q model first
        '''if os.path.exists("q_model.pth"):
            try:
                from model.set_transformer.q_transformer import QTransformer
                self.q_model = QTransformer().to(self.device)
                self.q_model.load_state_dict(torch.load("q_model.pth", map_location=self.device))
                self.q_model.eval()
                self.use_ml_q = True
                rospy.loginfo("✅ Adaptive Q model loaded")
            except Exception as e:
                rospy.logwarn(f"Could not load Q model: {e}")'''
        
        # ===== Load AKIT model (new) =====
        # Update this path to your trained model
        akit_model_path = "runs/akit_experiment_20260314_121507/best_model.pth"
        
        if os.path.exists(akit_model_path):
            try:
                self.akit_predictor = AKITPredictor(akit_model_path, device=self.device)
                self.use_akit = True
                rospy.loginfo("✅ AKIT model loaded successfully")
            except Exception as e:
                rospy.logwarn(f"⚠️ Failed to load AKIT model: {e}")
                rospy.logwarn("Falling back to fixed noise parameters")
        else:
            rospy.logwarn(f"⚠️ AKIT model not found at {akit_model_path}")
            rospy.logwarn("Using fixed noise parameters or old Q model")
        
        # ===== END ADAPTIVE MODEL LOADING =====
       
        # Initialize buffers
        self.accel_buffer = []
        self.window_size = 10
        self.latest_frame = None
        self.last_update_p = None
        self.last_imu_t = 0
        self.current_metrics = {
            'innov': 0.0, 'trace': 0.0, 'cond': 0.0,
            'h_norm': 0.0, 'matches': 0, 'baseline': 0.0,
            'avg_error': np.nan,
            'status': 'IDLE',  # Add status for AKIT
            'accel_mag': 0.0,  # Add for AKIT
            'accel_var': 0.0,  # Add for AKIT
            'is_zupt': 0       # Add for AKIT
        }
        
        # Triangulation and point selection parameters
        self.triangulation_config = {
            'max_reprojection_error': 15.0,  # Pixels
            'min_baseline_depth_ratio': 0.02,  # Minimum baseline/depth ratio
            'max_point_distance': 150.0,  # Maximum distance from vehicle
            'min_point_distance': 1.5,  # Minimum distance from vehicle
            'score_top_percentage': 0.7,  # Keep top 70% of points by score
        }

        # Setup logging
        if not os.path.exists("logs"):
            os.makedirs("logs")
     
        header = ("ts,status,px_v,py_v,pz_v,px_gt,py_gt,pz_gt,"
              "vx_v,vy_v,vz_v,vx_gt,vy_gt,vz_gt,"
              "ax_v,ay_v,az_v,roll_v,pitch_v,yaw_v,roll_gt,pitch_gt,yaw_gt,"
              "innov_norm,s_trace,s_cond,h_norm,num_matches,baseline_ratio,"
              "accel_mag,accel_var,is_zupt,gyro_scale,acc_scale,bias_scale,r_scale\n")
        # With these (fixed filenames):
        self.debug_log = open(f"logs/vio_full.log", "w", buffering=1)
        self.debug_log.write(header)
        self.debug_log.flush()
        self.hdf5_logger = VIOHDF5Logger(f"logs/training_data.h5")

        try:
            self.visual_pipe = VisualPipeline(feat_dim=256).to(self.device).eval()
            self.orb_extractor = HybridKeypointExtractor(max_keypoints=2000)
            self.bridge = CvBridge()
            print("Visual pipeline initialized successfully")
        except Exception as e:
            print(f"ERROR initializing visual pipeline: {e}")
            raise
       
        x0 = State(
            R=np.eye(3),
            p=np.zeros(3),
            v=np.zeros(3),
            bg=np.zeros(3),
            ba=np.zeros(3),
            s=1.0,           # Scale factor
            Y_v=-100.0,      # Initial sway damping coefficient (negative)
            Y_r=10.0         # Initial yaw-sway coupling
        )

        # Create 17x17 covariance matrix
        P0 = np.eye(17)
        P0[0:3, 0:3] = np.eye(3) * 0.1      # Orientation (increase from 0.01)
        P0[3:6, 3:6] = np.eye(3) * 10.0      # Velocity (increase from 0.1) - CRITICAL!
        P0[6:9, 6:9] = np.eye(3) * 1.0       # Position (keep)
        P0[9:12, 9:12] = np.eye(3) * 0.1     # Gyro bias (increase from 0.01)
        P0[12:15, 12:15] = np.eye(3) * 1.0   # Accel bias (increase from 0.1)
        P0[15, 15] = 1.0                      # Scale factor (increase from 0.01) - CRITICAL!
        P0[16, 16] = 10.0                     # Damping coefficient (keep)
      
               
        self.ekf = EKFSE3(x0, P0, gravity=9.793)
        # 2. Add recovery thresholds to EKF
        self.ekf.recovery_threshold = 20.0    # 20m error triggers recovery
        self.ekf.max_orientation_jump = 10.0  # 10° max rotation per update

        # 3. Add innovation limits
        self.max_innov_norm = 500.0           # Reject innovations >500
        self.max_mahalanobis = 100.0          # Mahalanobis distance threshold

        # 4. Add tracking for orientation health
        self.orientation_failures = 0
        self.last_valid_orientation = self.ekf.x.R.copy()
        self.orientation_drift_counter = 0
        # Reduced noise parameters
        self.noise = NoiseParams()
        self.noise.gyro_noise = 0.0001
        self.noise.accel_noise = 0.001
        self.noise.gyro_bias_rw = 1e-7
        self.noise.accel_bias_rw = 1e-6
        
        # Store current AKIT scales for logging
        self.current_gyro_scale = 1.0
        self.current_acc_scale = 1.0
        self.current_bias_scale = 1.0
        self.current_r_scale = 1.0
        
        self.test_y_axis_sign()
        print(f"\n=== BUFFER INITIALIZATION ===")
        # Buffers
        self.imu_history = []
        self.kps_buf = []
        self.descs_buf = []
        self.ts_buf = []
        self.R_buf = []
        self.p_buf = []
        self.innov_history = []
        self.points_history = []
        self.gt_p = np.zeros(3)
        self.gt_v = np.zeros(3)
        self.gt_R = np.eye(3)
        self.latest_imu = np.zeros(6)
        self.origin_initialized = False
        self.frame_count = 0
        self.last_img_t = None
        self.last_innov_norm = 0.0
        self.last_s_trace = 0.0
        self.last_s_cond = 0.0
        self.last_h_norm = 0.0
        self.last_num_points = 0
        self.success_history = []
        # In __init__, add these after other initializations:
        self.last_valid_points = 0
        print("\n=== INITIAL PROJECTION TEST ===")
        self.test_projection_with_new_extrinsics()
        # IMU to body transform
        self.R_imu_to_body = np.eye(3)
        print(f"IMU to body transform:\n{self.R_imu_to_body}")
       
        # Store poses for vision updates
        self.buffer_max_size = 500
        self.ekf_poses_buffer = []  
        self.ekf_positions_buffer = []
       
        print(f"\n=== ROS PUBLISHER/SUBSCRIBER INITIALIZATION ===")
        # ROS publishers/subscribers
        self.odom_pub = rospy.Publisher("/vio/odom", Odometry, queue_size=10)
       
        # Add debug subscribers
        rospy.Subscriber("/lauv/imu", Imu, self.imu_callback_debug, queue_size=1000)
        rospy.Subscriber("/lauv/lauv/camerafront/camera_image", Image, self.image_callback_debug, queue_size=10)
       
        # Then the real ones
        self.imu_sub = rospy.Subscriber("/lauv/imu", Imu, self.imu_callback, queue_size=1000)
        self.img_sub = rospy.Subscriber("/lauv/lauv/camerafront/camera_image", Image, self.image_callback, queue_size=10)
        self.gt_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.gt_callback, queue_size=10)
       
        rospy.on_shutdown(self.shutdown_hook)
        self.wait_for_camera_ready()
        # Initialize attributes
        self.image_buffer = []
        self.first_frame_time = None
        self.initial_gravity_estimated = False
        self.velocity_history = []
        self.last_flow_magnitude = 0.0
        self.expected_flow = 0.0
        self.last_triangulation_depth = []
        self.last_vision_position_change = None
        self.last_imu_position_change = None
       
        # Observability monitoring
        self.observability_window = []          # store recent H matrices from successful updates
        self.max_obs_window = 50                 # maximum number of matrices to keep
        self.observability_last_analysis = 0     # timestamp of last analysis

        # Innovation consistency
        self.innovation_history = []              # list of (timestamp, nis, dof)

        # Health tracking and fallback
        self.vio_health = 1.0                     # health score 0-1
        self.fallback_mode = False                 # when True, use model‑based propagation
        self.last_fallback_attempt = 0             # when we last tried to exit fallback
        self.fallback_cooldown = 5.0               # seconds between fallback exit attempts

        print(f"\n=== GAZEBO VIO NODE INITIALIZATION COMPLETE ===")
        print(f"Waiting for data...")

        self.drift_analysis = {
            'positions': [],
            'gt_positions': [],
            'timestamps': [],
            'errors': [],
            'update_times': [],
            'innovation_history': [],
            'window_size': 100,
            'drift_rate': 0.0,
            'last_analysis_time': time.time()
        }
       
        # Add scale observer
        self.scale_observer = {
            'velocity_ratio': 1.0,
            'feature_flow': 1.0,
            'depth_consistency': 1.0,
            'fused_scale': 1.0,
            'confidence': 0.0,
            'history': [],
            'last_update': time.time()
        }
       
        # Add adaptive boost factors
        self.adaptive_boosts = {
            'scale': 1.0,
            'y_bias': 1.0,
            'rotation': 1.0
        }
    
    # ===== FIXED: Helper methods for AKIT =====
    def prepare_akit_metrics(self, status, acc_mag, acc_var, is_zupt):
        """Prepare metrics dictionary for AKIT model"""
        metrics = {
            'accel_mag': acc_mag,
            'accel_var': acc_var,
            'num_matches': self.current_metrics.get('matches', 0),
            's_trace': getattr(self, 'last_s_trace', 0.0),
            's_cond': getattr(self, 'last_s_cond', 0.0),
            'h_norm': getattr(self, 'last_h_norm', 0.0),
            'status': status,
            'baseline_ratio': self.current_metrics.get('baseline', 0.0),
            'is_zupt': is_zupt
        }
        return metrics
    
    def get_adaptive_R(self, base_R=30.0):
        """Get adaptive measurement noise from AKIT model"""
        if not self.use_akit or len(self.ctx_buf) < self.window_size:
            return base_R
        
        try:
            # FIXED: Prepare IMU buffer with ALL context data (including orientation)
            imu_buffer = []
            for ctx_entry in self.ctx_buf[-self.window_size:]:
                # ctx_entry: [ax, ay, az, roll, pitch, yaw, is_zupt]
                imu_buffer.append([
                    ctx_entry[0], ctx_entry[1], ctx_entry[2],  # ax, ay, az
                    ctx_entry[3], ctx_entry[4], ctx_entry[5],  # roll, pitch, yaw (FIXED!)
                    ctx_entry[6]  # is_zupt
                ])
            
            # Prepare metrics with current visual info
            metrics = self.prepare_akit_metrics(
                status=self.current_metrics.get('status', 'IDLE'),
                acc_mag=self.current_metrics.get('accel_mag', 0),
                acc_var=self.current_metrics.get('accel_var', 0),
                is_zupt=self.current_metrics.get('is_zupt', 0)
            )
            
            # Get predictions from AKIT
            q_scales, r_scale = self.akit_predictor.predict_from_buffer(imu_buffer, metrics)
            
            # Store current scales for logging
            self.current_gyro_scale = float(np.clip(q_scales[0], 0.1, 10.0))
            self.current_acc_scale = float(np.clip(q_scales[1], 0.1, 10.0))
            self.current_bias_scale = float(np.clip(q_scales[2], 0.1, 10.0))
            self.current_r_scale = float(np.clip(r_scale, 0.1, 10.0))
            
            # Scale base R (pixel sigma) by predicted R scale
            adaptive_R = base_R * self.current_r_scale
            
            # Clamp to reasonable range with wider bounds
            adaptive_R = np.clip(adaptive_R, base_R * 0.1, base_R * 10.0)
            
            if self.frame_count % 50 == 0:
                print(f"[AKIT-R] R_scale={self.current_r_scale:.2f} → adaptive_R={adaptive_R:.1f}px")
            
            return adaptive_R
            
        except Exception as e:
            rospy.logwarn_throttle(10.0, f"AKIT-R prediction failed: {e}")
            return base_R
    # ===== END FIXED METHODS =====

    def get_vio_extrinsics_correct(self):
        """
        Camera mounted 45° down, looking forward.
        Body: X forward, Y right, Z down
        Camera: Z forward, X right, Y down (with 45° pitch)
        """
        t_bc = np.array([1.2, 0.0, 0.0])  # Camera on right side
        
        pitch = 45.0 * np.pi / 180.0
        c = np.cos(pitch)  # 0.707
        s = np.sin(pitch)  # 0.707
        
        R_bc = np.zeros((3, 3))
        
        # Body X (forward) → Camera Y (s) and Camera Z (c)
        R_bc[:, 0] = [0, s, c]
        
        # Body Y (right) → Camera X (1)
        R_bc[:, 1] = [1, 0, 0]
        
        # Body Z (down) → Camera Y (c) and Camera Z (-s) 
        R_bc[:, 2] = [0, -c, s]  # Negative Y for down
        
        # ===== VERIFICATION =====
        print("\n🔧 EXTRINSICS VERIFICATION:")
        
        test_forward = np.array([1, 0, 0])
        in_camera = R_bc @ test_forward
        print(f"  Body forward in camera: {in_camera}")
        print(f"  Camera Z component: {in_camera[2]:.3f} (should be POSITIVE)")
        
        test_down = np.array([0, 0, 1])
        in_camera_down = R_bc @ test_down
        print(f"  Body down in camera: {in_camera_down}")
        print(f"  Camera Y component: {in_camera_down[1]:.3f} (should be NEGATIVE)")
        
        test_right = np.array([0, 1, 0])
        in_camera_right = R_bc @ test_right
        print(f"  Body right in camera: {in_camera_right}")
        print(f"  Camera X component: {in_camera_right[0]:.3f} (should be POSITIVE)")
        
        # Test with 10m forward
        print(f"\n  Forward 10m in camera: {R_bc @ [10, 0, 0]}")
        
        return R_bc, t_bc

    def get_adaptive_thresholds(self):
        """Get adaptive thresholds based on recent success history"""
        current_vel = np.linalg.norm(self.ekf.x.v)
        
        # Get bootstrap mode flag
        bootstrap = getattr(self, 'bootstrap_mode', True)
        update_count = getattr(self, 'vision_update_count', 0)
        
        # Base thresholds - start conservative but with good defaults
        config = {
            'min_baseline': 0.3,
            'max_baseline': 5.0,
            'min_points': 8,
            'pixel_sigma': 60.0,
            'max_reprojection': 200.0
        }
        
        # ===== BOOTSTRAP MODE: MUCH MORE PERMISSIVE =====
        if bootstrap or update_count < 10:
            config['min_points'] = 4
            config['pixel_sigma'] = 100.0
            config['max_reprojection'] = 500.0
            config['min_baseline'] = 0.1
            config['max_baseline'] = 8.0
            print(f"  [ADAPTIVE] 🚀 BOOTSTRAP: very permissive (min_points={config['min_points']})")
            return config
        
        # Adjust based on recent success rate
        if hasattr(self, 'success_history') and len(self.success_history) > 5:
            recent_successes = [s for s in self.success_history[-20:] if s.get('success', False)]
            success_rate = len(recent_successes) / 20 if len(self.success_history) >= 20 else 0.5
            recent_innovs = [s.get('innov', 300) for s in recent_successes if s.get('innov', 0) > 0]
            avg_innov = np.mean(recent_innovs) if recent_innovs else 300
            
            print(f"  [ADAPTIVE] Success rate: {success_rate:.2f}, Avg innov: {avg_innov:.1f}")
            
            # MUCH MORE AGGRESSIVE when success rate is low
            if success_rate < 0.2:  # Less than 20% success - very loose
                config['min_points'] = 5
                config['pixel_sigma'] = 90.0
                config['max_reprojection'] = 600.0
                config['min_baseline'] = 0.15
                config['max_baseline'] = 7.0
                print(f"  [ADAPTIVE] ⚠️ Very low success rate, very loose thresholds")
                
            elif success_rate < 0.3:  # 20-30% success - moderately loose
                config['min_points'] = 6
                config['pixel_sigma'] = 80.0
                config['max_reprojection'] = 400.0
                config['min_baseline'] = 0.2
                config['max_baseline'] = 6.0
                print(f"  [ADAPTIVE] ⚠️ Low success rate, loose thresholds")
                
            elif success_rate > 0.5:  # More than 50% success - tighten for accuracy
                config['min_points'] = 10
                config['pixel_sigma'] = 40.0
                config['max_reprojection'] = 250.0
                config['min_baseline'] = 0.4
                config['max_baseline'] = 5.0
                print(f"  [ADAPTIVE] ✅ Good success rate, tight thresholds")
            
            # If innovation is consistently high, loosen pixel sigma
            if avg_innov > 50 and success_rate < 0.4:
                config['pixel_sigma'] = min(100.0, config['pixel_sigma'] * 1.2)
                print(f"  [ADAPTIVE] 📊 High innovation, increasing pixel sigma to {config['pixel_sigma']:.1f}")
        
        # Velocity-based adjustments
        if current_vel < 0.2:  # Slow moving
            config['min_baseline'] = max(0.15, config['min_baseline'])
            config['pixel_sigma'] = min(40.0, config['pixel_sigma'])
            
        elif current_vel > 2.0:  # Fast moving
            # Need larger baseline for same angular resolution
            config['min_baseline'] = min(0.5, config['min_baseline'] + 0.1)
            # Allow larger max baseline when moving fast
            config['max_baseline'] = min(10.0, config['max_baseline'] + 2.0)
            # More pixel motion expected
            config['pixel_sigma'] = max(50.0, config['pixel_sigma'])
        
        # Safety bounds
        config['min_points'] = max(4, min(15, config['min_points']))
        config['pixel_sigma'] = np.clip(config['pixel_sigma'], 20.0, 120.0)
        config['max_reprojection'] = np.clip(config['max_reprojection'], 150.0, 800.0)
        config['min_baseline'] = np.clip(config['min_baseline'], 0.05, 0.5)
        config['max_baseline'] = np.clip(config['max_baseline'], 3.0, 12.0)
        
        return config

    def update_expected_flow(self, dt):
        """Calculate expected optical flow based on current motion"""
        if dt <= 0:
            return
       
        # Get average depth of tracked points (if available)
        avg_depth = 20.0  # Default assumption
        if hasattr(self, 'last_triangulation_depth') and len(self.last_triangulation_depth) > 0:
            # Use median to be robust to outliers
            avg_depth = np.median(self.last_triangulation_depth[-20:])
            # Clamp to reasonable range
            avg_depth = np.clip(avg_depth, 5.0, 100.0)
       
        # Expected flow = (velocity * dt * focal_length) / depth
        velocity = np.linalg.norm(self.ekf.x.v)
        self.expected_flow = (velocity * dt * self.K_cam[0,0]) / (avg_depth + 1e-6)
       
        if self.frame_count % 50 == 0:
            print(f"  🌊 Expected flow: {self.expected_flow:.2f}px (vel={velocity:.2f}m/s, depth={avg_depth:.1f}m)")

    def measure_optical_flow(self, pts1, pts2):
        """Measure actual optical flow magnitude"""
        if len(pts1) < 10:
            self.last_flow_magnitude = 0.0
            return 0.0
       
        displacements = np.linalg.norm(pts2 - pts1, axis=1)
        # Use median to be robust to outliers
        self.last_flow_magnitude = np.median(displacements)
        if hasattr(self, 'last_flow_magnitude') and self.last_flow_magnitude > 0:
            # Estimate depth from last triangulated points, or use a default (e.g., 20 m)
            avg_depth = 20.0
            if hasattr(self, 'last_triangulation_depth') and len(self.last_triangulation_depth) > 0:
                avg_depth = np.median(self.last_triangulation_depth[-10:])
                avg_depth = np.clip(avg_depth, 5.0, 100.0)
            
            # Convert flow to body velocity: v_body = flow * depth / fx
            flow_to_vel = avg_depth / self.K_cam[0,0]
            v_body_flow = self.last_flow_magnitude * flow_to_vel
            
            # Create a velocity measurement (only magnitude, direction from optical flow centroid?)
            # For simplicity, assume the dominant motion is forward (body x). You can improve by using average flow vector.
            y_flow = np.array([v_body_flow, 0, 0]) - self.ekf.x.v   # innovation
            H_flow = np.zeros((3, 17))
            H_flow[:, 3:6] = np.eye(3)  # measure velocity
            R_flow = np.eye(3) * (0.5 * v_body_flow)**2  # noise proportional to speed
            
            # Apply update (use update_generic if you have it)
            self.ekf.update_generic(y_flow, H_flow, R_flow, debug=False)
            
        if self.frame_count % 50 == 0:
            print(f"  🌊 Measured flow: {self.last_flow_magnitude:.2f}px")
       
        return self.last_flow_magnitude

    def _bound_covariance(self):
        """Bound covariance matrix to prevent numerical issues"""
        if hasattr(self.ekf, '_bound_covariance'):
            self.ekf._bound_covariance()
        else:
            # Simple bounding
            min_cov = 1e-9
            max_cov = 1e6
            for i in range(17):
                if self.ekf.P[i, i] < min_cov:
                    self.ekf.P[i, i] = min_cov
                elif self.ekf.P[i, i] > max_cov:
                    self.ekf.P[i, i] = max_cov

    def sanity_check_triangulation(self, pts3d_world, pts1, pts2, P1, P2):
        """Sanity check triangulated points"""
        print("\n=== TRIANGULATION SANITY CHECK ===")
       
        if len(pts3d_world) == 0:
            print("  No points to check")
            return False
       
        # Get camera centers
        C1 = -np.linalg.inv(P1[:, :3]) @ P1[:, 3]
        C2 = -np.linalg.inv(P2[:, :3]) @ P2[:, 3]
       
        # Vehicle position (current)
        vehicle_pos = self.ekf.x.p
       
        # Analyze points
        z_world = pts3d_world[:, 2]
        z_vehicle = vehicle_pos[2]
       
        print(f"  Vehicle depth: {z_vehicle:.1f}m")
        print(f"  Point Z in world: min={z_world.min():.1f}, max={z_world.max():.1f}, mean={z_world.mean():.1f}")
        print(f"  Points above vehicle: {np.sum(z_world > z_vehicle)}/{len(pts3d_world)}")
        print(f"  Points below vehicle: {np.sum(z_world < z_vehicle)}/{len(pts3d_world)}")
       
        # Check a few points in detail
        for i in range(min(3, len(pts3d_world))):
            pt = pts3d_world[i]
            print(f"\n  Point {i}:")
            print(f"    World: [{pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}]")
           
            # Reproject into both cameras
            pt_hom = np.append(pt, 1)
            p1 = P1 @ pt_hom
            p2 = P2 @ pt_hom
           
            if p1[2] > 0 and p2[2] > 0:
                u1, v1 = p1[0]/p1[2], p1[1]/p1[2]
                u2, v2 = p2[0]/p2[2], p2[1]/p2[2]
                print(f"    Reproj in cam1: [{u1:.1f}, {v1:.1f}] (meas: [{pts1[i,0]:.1f}, {pts1[i,1]:.1f}])")
                print(f"    Reproj in cam2: [{u2:.1f}, {v2:.1f}] (meas: [{pts2[i,0]:.1f}, {pts2[i,1]:.1f}])")
               
                error1 = np.sqrt((u1-pts1[i,0])**2 + (v1-pts1[i,1])**2)
                error2 = np.sqrt((u2-pts2[i,0])**2 + (v2-pts2[i,1])**2)
                print(f"    Errors: {error1:.1f}px, {error2:.1f}px")
            else:
                print(f"    Point behind camera in at least one view")
       
        # Return True if most points are below vehicle
        return np.sum(z_world < z_vehicle) > len(pts3d_world) * 0.5

    def get_adaptive_frame_search_params(self):
        """Dynamically adjust frame search based on observed frame rate"""
        
        # Calculate current frame rate from buffer
        if len(self.ts_buf) >= 10:
            # Average time between recent frames
            dt_sum = 0
            for i in range(1, min(10, len(self.ts_buf))):
                dt_sum += self.ts_buf[-i] - self.ts_buf[-i-1]
            avg_frame_dt = dt_sum / min(9, len(self.ts_buf)-1)
            
            # Update running average of frame rate
            if not hasattr(self, 'avg_frame_dt'):
                self.avg_frame_dt = avg_frame_dt
            else:
                self.avg_frame_dt = 0.9 * self.avg_frame_dt + 0.1 * avg_frame_dt
            
            # Adaptive parameters based on actual frame rate
            if self.avg_frame_dt > 2.0:  # Very slow (<0.5 Hz)
                min_baseline = 1.0  # Need larger baseline
                max_baseline = 8.0  # Allow larger gaps
                max_dt = 10.0        # Search up to 10 seconds back
                print(f"  [ADAPT] Very slow frames ({1/self.avg_frame_dt:.1f} Hz) - using larger windows")
            elif self.avg_frame_dt > 1.0:  # Slow (1 Hz)
                min_baseline = 0.7
                max_baseline = 5.0
                max_dt = 6.0
                print(f"  [ADAPT] Slow frames ({1/self.avg_frame_dt:.1f} Hz)")
            else:  # Faster (>1 Hz)
                min_baseline = 0.5
                max_baseline = 3.0
                max_dt = 4.0
        else:
            # Default values before enough data
            min_baseline = 0.5
            max_baseline = 3.0
            max_dt = 5.0
        
        return min_baseline, max_baseline, max_dt

    def get_frame_indices(self):
        """
        Get indices of frames - FORCE CONSECUTIVE FRAMES DURING BOOTSTRAP
        with baseline check to prevent scale explosion.
        """
        if len(self.ts_buf) < 2:
            print(f"  [FRAME-SEARCH] ❌ Buffer too small: {len(self.ts_buf)} frames")
            return None, None

        newest_idx = len(self.ts_buf) - 1
        current_t = self.ts_buf[newest_idx]

        # ===== BOOTSTRAP: ONLY USE CONSECUTIVE FRAMES =====
        bootstrap = getattr(self, 'bootstrap_mode', True)
        if bootstrap and self.vision_update_count < 20:
            if newest_idx >= 1:
                i = newest_idx - 1
                j = newest_idx
                dt = self.ts_buf[j] - self.ts_buf[i]

                # Check baseline if poses are available
                if (i < len(self.ekf_poses_buffer) and j < len(self.ekf_poses_buffer) and
                    i < len(self.ekf_positions_buffer) and j < len(self.ekf_positions_buffer)):

                    R_i = self.ekf_poses_buffer[i]
                    p_i = self.ekf_positions_buffer[i]
                    R_j = self.ekf_poses_buffer[j]
                    p_j = self.ekf_positions_buffer[j]

                    dp = R_i.T @ (p_j - p_i)
                    baseline = np.linalg.norm(dp)
                    lateral = abs(dp[1])

                    print("\n" + "="*80)
                    print(f"  🚀 BOOTSTRAP MODE - CONSECUTIVE FRAMES CHECK")
                    print("="*80)
                    print(f"  Frames {i} and {j}, dt={dt:.2f}s")
                    print(f"  Baseline: {baseline:.2f}m, Lateral: {lateral:.2f}m")
                    print(f"  ✅ Baseline OK, using frames {i} and {j}")
                    return i, j
                else:
                    # Fallback if poses not available
                    print(f"\n  🚀 BOOTSTRAP: using consecutive frames {i} and {j}, dt={dt:.2f}s")
                    return i, j
            return None, None

    

        min_baseline, max_baseline, max_dt = self.get_adaptive_frame_search_params_45deg()
 
        candidates = []
        best_score = -1
        best_pair = None

        # Estimate average feature depth
        avg_depth = 30.0
        if hasattr(self, 'last_triangulation_depth') and len(self.last_triangulation_depth) > 10:
            avg_depth = np.median(self.last_triangulation_depth[-20:])
            avg_depth = np.clip(avg_depth, 10.0, 100.0)

        focal_length = self.K_cam[0,0]

        search_start = max(0, len(self.ts_buf) - 20)
     
        for i in range(len(self.ts_buf)-2, search_start-1, -1):
            dt = current_t - self.ts_buf[i]
            if dt > max_dt:
                continue

            if i >= len(self.ekf_poses_buffer) or newest_idx >= len(self.ekf_poses_buffer):
                continue
            if i >= len(self.ekf_positions_buffer) or newest_idx >= len(self.ekf_positions_buffer):
                continue

            R_i = self.ekf_poses_buffer[i]
            p_i = self.ekf_positions_buffer[i]
            R_j = self.ekf_poses_buffer[newest_idx]
            p_j = self.ekf_positions_buffer[newest_idx]

            dR = R_i.T @ R_j
            dp = R_i.T @ (p_j - p_i)

            # Translation components
            forward_motion = dp[0]
            lateral_motion = dp[1]
            vertical_motion = dp[2]
            forward_dist = abs(forward_motion)
            lateral_dist = abs(lateral_motion)
            vertical_dist = abs(vertical_motion)
            total_translation = np.linalg.norm(dp)

            # ===== MAX BASELINE CHECK (NEW) =====
            if total_translation > max_baseline:
                continue   # skip frames that are too far apart

            # Rotation components (degrees)
            roll = np.arctan2(dR[2,1], dR[2,2]) * 180/np.pi
            pitch = np.arctan2(-dR[2,0], np.sqrt(dR[0,0]**2 + dR[1,0]**2)) * 180/np.pi
            yaw = np.arctan2(dR[1,0], dR[0,0]) * 180/np.pi
            total_rotation = np.linalg.norm([roll, pitch, yaw])

            # Pixel motion estimates
            lateral_pixels = focal_length * lateral_dist / avg_depth
            yaw_rad = abs(yaw * np.pi/180)
            yaw_pixels = focal_length * yaw_rad
            forward_pixels = focal_length * forward_dist * 0.707 / avg_depth
            horizontal_pixels = lateral_pixels + yaw_pixels

            # Feature overlap estimate
            if total_translation < 5.0 and dt < 2.0:
                estimated_overlap = 80.0 * (1.0 - total_translation/10.0) * (1.0 - dt/3.0)
            elif total_translation < 10.0 and dt < 4.0:
                estimated_overlap = 50.0 * (1.0 - total_translation/20.0) * (1.0 - dt/5.0)
            elif total_translation < 20.0 and dt < 6.0:
                estimated_overlap = 20.0 * (1.0 - total_translation/30.0) * (1.0 - dt/8.0)
            else:
                estimated_overlap = 0.0
            estimated_overlap = np.clip(estimated_overlap, 0.0, 100.0)

            # Quality score based on overlap
            if estimated_overlap < 10:
                overlap_factor = 0.1
                quality = "POOR (no overlap)"
            elif estimated_overlap < 30:
                overlap_factor = 0.5
                quality = "FAIR (low overlap)"
            elif estimated_overlap < 50:
                overlap_factor = 0.8
                quality = "GOOD (medium overlap)"
            else:
                overlap_factor = 1.0
                quality = "EXCELLENT (high overlap)"

            base_score = min(horizontal_pixels, 200.0)  # cap at 200px
            if forward_pixels > horizontal_pixels * 2 and horizontal_pixels > 0:
                forward_penalty = 0.6
            else:
                forward_penalty = 1.0

            score = base_score * overlap_factor * forward_penalty

            candidate_info = {
                'idx': i,
                'dt': dt,
                'forward_m': forward_dist,
                'lateral_m': lateral_dist,
                'total_motion': total_translation,
                'horizontal_px': horizontal_pixels,
                'estimated_overlap': estimated_overlap,
                'score': score,
                'quality': quality
            }
            candidates.append(candidate_info)

            if score > best_score:
                best_score = score
                best_pair = (i, newest_idx)

        # ===== SELECTION DECISION =====
        print("\n" + "-"*80)

        if not candidates:
            print(f"  ❌ No valid candidates found")
            if len(self.ts_buf) >= 2:
                print(f"  → ULTIMATE FALLBACK: consecutive frames")
                return len(self.ts_buf)-2, newest_idx
            return None, None

        candidates.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n  📊 TOP 3 CANDIDATES:")
        for j, cand in enumerate(candidates[:3]):
            print(f"     {j+1}. Frame {cand['idx']}: score={cand['score']:.2f}, "
                f"hor={cand['horizontal_px']:.1f}px, dt={cand['dt']:.2f}s, "
                f"overlap={cand['estimated_overlap']:.1f}% [{cand['quality']}]")

        if best_score > 20 and candidates[0]['estimated_overlap'] > 20:
            i, j = best_pair
            dt = self.ts_buf[j] - self.ts_buf[i]
            print(f"\n  ✅ SELECTED: frames {i} and {j}")
            print(f"     Score: {best_score:.2f}, Overlap: {candidates[0]['estimated_overlap']:.1f}%")
            return best_pair
        else:
            print(f"\n  ⚠️ No good candidates - using consecutive frames")
            return len(self.ts_buf)-2, newest_idx
        
    def estimate_feature_overlap(self, idx1, idx2):
        """
        Estimate actual feature overlap between two frames using ORB matching.
        Call this occasionally for debugging.
        """
        if idx1 >= len(self.kps_buf) or idx2 >= len(self.kps_buf):
            return 0.0
        
        # Use a simple matcher (BFMatcher) for quick estimation
        import cv2
        
        desc1 = self.descs_buf[idx1].astype(np.uint8)
        desc2 = self.descs_buf[idx2].astype(np.uint8)
        
        if len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        
        overlap_ratio = len(matches) / min(len(desc1), len(desc2))
        
        print(f"  [OVERLAP] Frame {idx1}-{idx2}: {len(matches)} matches, "
            f"ratio={overlap_ratio:.2f}")
        
        return overlap_ratio

    def get_adaptive_frame_search_params_45deg(self):
        """
        Adaptive parameters - DURING BOOTSTRAP, USE SMALL WINDOWS
        """
        # Default values
        min_baseline = 0.3
        max_baseline = 5.0
        max_dt = 6.0
        
        if len(self.ts_buf) < 10:
            return min_baseline, max_baseline, max_dt
        
        # Calculate average frame rate
        dt_sum = 0
        for i in range(1, min(10, len(self.ts_buf))):
            dt_sum += self.ts_buf[-i] - self.ts_buf[-i-1]
        avg_frame_dt = dt_sum / min(9, len(self.ts_buf)-1)
        frame_rate = 1.0 / avg_frame_dt if avg_frame_dt > 0 else 1.0
        
        # Check if we're in bootstrap mode
        bootstrap = getattr(self, 'bootstrap_mode', True)
        
        print(f"  [ADAPT-45°] frame_rate={frame_rate:.1f}Hz, bootstrap={bootstrap}")
        
        # ===== CRITICAL: DURING BOOTSTRAP, USE SMALL WINDOWS =====
        if bootstrap and self.vision_update_count < 20:
            min_baseline = 0.1   # Very small baseline
            max_baseline = 3.0   # Max 3 meters
            max_dt = 2.0          # Max 2 seconds
            print(f"  [ADAPT-45°] 🚀 BOOTSTRAP: using small windows (dt<{max_dt}s)")
            return min_baseline, max_baseline, max_dt
        
        # ===== AFTER BOOTSTRAP, USE ADAPTIVE LOGIC =====
        
        # Check recent success rate
        success_rate = 0.5
        if hasattr(self, 'vision_health'):
            total = self.vision_health.get('total_success', 0) + self.vision_health.get('total_failures', 0)
            if total > 10:
                success_rate = self.vision_health['total_success'] / total
        
        # Check scale health
        scale_healthy = True
        if hasattr(self.ekf, 'x') and hasattr(self.ekf.x, 's'):
            if abs(self.ekf.x.s - 1.0) > 0.1:
                scale_healthy = False
        
        if success_rate < 0.3:
            if frame_rate < 1.0:
                min_baseline = 0.4
                max_baseline = 5.0
                max_dt = 4.0
            else:
                min_baseline = 0.3
                max_baseline = 4.0
                max_dt = 3.0
            print(f"  [ADAPT-45°] Low success ({success_rate:.2f}) - reducing windows")
        
        elif not scale_healthy:
            min_baseline = 0.5
            max_baseline = 5.0
            max_dt = 4.0
            print(f"  [ADAPT-45°] Scale drifting - moderate windows")
        
        else:
            if frame_rate < 0.7:
                min_baseline = 0.5
                max_baseline = 5.0
                max_dt = 5.0
            elif frame_rate < 1.5:
                min_baseline = 0.4
                max_baseline = 4.0
                max_dt = 4.0
            else:
                min_baseline = 0.3
                max_baseline = 3.0
                max_dt = 3.0
        
        return min_baseline, max_baseline, max_dt

    def force_gravity_alignment(self):
        """
        Force roll and pitch to align with gravity vector.
        This corrects orientation drift which affects everything else.
        Should be called periodically (every 10-20 frames).
        """
        if not hasattr(self, 'latest_imu') or len(self.latest_imu) < 6:
            return False
        
        # Get accelerometer reading (low-pass filtered to reduce noise)
        accel_raw = self.latest_imu[3:6].copy()
        
        # Apply current bias estimate
        accel_corrected = accel_raw - self.ekf.x.ba
        
        # Normalize to get gravity direction in body frame
        accel_norm = np.linalg.norm(accel_corrected)
        if accel_norm < 8.0 or accel_norm > 11.0:  # Sanity check
            print(f"  ⚠️ Gravity alignment skipped - accelerometer magnitude {accel_norm:.2f} (should be ~9.8)")
            return False
        
        gravity_dir_body = accel_corrected / accel_norm
        
        # Expected gravity direction in body frame if orientation is correct
        # (gravity is [0,0,9.8] in world frame, so in body frame it's R.T @ [0,0,1])
        expected_gravity_body = self.ekf.x.R.T @ np.array([0, 0, 1])
        
        # Compute current alignment quality
        alignment_cosine = np.dot(gravity_dir_body, expected_gravity_body)
        alignment_angle = np.degrees(np.arccos(np.clip(alignment_cosine, -1, 1)))
        
        print(f"\n=== GRAVITY ALIGNMENT CHECK ===")
        print(f"  Current gravity in body: [{gravity_dir_body[0]:.3f}, {gravity_dir_body[1]:.3f}, {gravity_dir_body[2]:.3f}]")
        print(f"  Expected gravity:        [{expected_gravity_body[0]:.3f}, {expected_gravity_body[1]:.3f}, {expected_gravity_body[2]:.3f}]")
        print(f"  Alignment angle: {alignment_angle:.2f}°")
        print(f"  Z-axis purity: {abs(gravity_dir_body[2]):.1%}")
        
        # If already well aligned, no need for correction
        if alignment_angle < 2.0 and abs(gravity_dir_body[2]) > 0.98:
            print(f"  ✅ Gravity already well aligned")
            return True
        
        # ===== COMPUTE CORRECTION ROTATION =====
        # We want to rotate the body frame so that expected_gravity aligns with measured_gravity
        # This corrects roll and pitch errors
        
        # The rotation that takes expected_gravity to measured_gravity
        v = np.cross(expected_gravity_body, gravity_dir_body)
        s = np.linalg.norm(v)
        c = np.dot(expected_gravity_body, gravity_dir_body)
        
        if s < 1e-6:
            # Already aligned or 180° off (shouldn't happen)
            if c < 0:
                # 180° off - this is bad! Flip around perpendicular axis
                print(f"  ⚠️ Gravity completely inverted!")
                # Find a perpendicular axis
                perp = np.array([1, 0, 0]) if abs(expected_gravity_body[0]) < 0.9 else np.array([0, 1, 0])
                v = np.cross(expected_gravity_body, perp)
                v = v / np.linalg.norm(v) * np.pi
                R_correction = self.rotation_from_axis_angle(v, np.pi)
            else:
                # Already aligned
                R_correction = np.eye(3)
        else:
            # Rodrigues rotation formula
            vx = np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
            R_correction = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))
        
        # Extract roll/pitch correction (ignore yaw component)
        # Convert to Euler angles
        correction_euler = R_tool.from_matrix(R_correction).as_euler('xyz')
        
        # Zero out the yaw component - we don't want to correct yaw with gravity
        correction_euler[2] = 0
        
        # Reconstruct rotation matrix with only roll/pitch correction
        R_correction_rp = R_tool.from_euler('xyz', correction_euler).as_matrix()
        
        # Apply correction with damping (don't correct fully in one shot)
        alpha = 0.3  # Correction strength (30%)
        
        # Interpolate between identity and full correction
        R_alpha = self.slerp_rotation(np.eye(3), R_correction_rp, alpha)
        
        # Apply to current orientation
        self.ekf.x.R = R_alpha @ self.ekf.x.R
        
        # Re-orthogonalize to prevent drift
        U, _, Vt = np.linalg.svd(self.ekf.x.R)
        self.ekf.x.R = U @ Vt
        
        # Reset theta (small angle approximation)
        self.ekf.x.theta = np.zeros(3)
        
        # Boost covariance for roll/pitch to reflect correction
        self.ekf.P[0,0] *= 1.5  # Roll
        self.ekf.P[1,1] *= 1.5  # Pitch
        
        print(f"  ✅ Applied gravity alignment correction ({alpha*100:.0f}% of {alignment_angle:.1f}° error)")
        
        # Re-check after correction
        new_expected = self.ekf.x.R.T @ np.array([0, 0, 1])
        new_align = np.dot(gravity_dir_body, new_expected)
        new_angle = np.degrees(np.arccos(np.clip(new_align, -1, 1)))
        print(f"  → New alignment angle: {new_angle:.2f}°")
        
        return True

    def slerp_rotation(self, R1, R2, t):
        """
        Spherical linear interpolation between two rotation matrices.
        t=0 -> R1, t=1 -> R2
        """
        # Convert to quaternions
        q1 = R_tool.from_matrix(R1).as_quat()
        q2 = R_tool.from_matrix(R2).as_quat()
        
        # Ensure same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        # Slerp
        dot = np.clip(np.dot(q1, q2), -1, 1)
        theta = np.arccos(dot)
        
        if theta < 1e-6:
            q = q1
        else:
            q = (np.sin((1-t) * theta) * q1 + np.sin(t * theta) * q2) / np.sin(theta)
        
        # Normalize
        q = q / np.linalg.norm(q)
        
        return R_tool.from_quat(q).as_matrix()

    def rotation_from_axis_angle(self, axis, angle):
        """Create rotation matrix from axis-angle representation"""
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1 - c
        
        return np.array([
            [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
            [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
            [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
        ])

    def periodic_gravity_check(self):
        """
        Call this periodically (every 10-20 frames) to maintain gravity alignment.
        Add this to your main loop or image_callback.
        """
        # Only run every N frames
        if not hasattr(self, '_gravity_check_counter'):
            self._gravity_check_counter = 0
        
        self._gravity_check_counter += 1
        
        # Check every 20 frames
        if self._gravity_check_counter >= 20:
            self._gravity_check_counter = 0
            self.force_gravity_alignment()
            
    
    def monitor_yaw_convergence(self):
        """Monitor yaw convergence and adjust rotation trust using only internal state"""
    
        if not hasattr(self, 'yaw_history'):
            self.yaw_history = []
            self.velocity_directions = []
            self.yaw_rate_history = []
    
        # Get current yaw
        current_rpy = R_tool.from_matrix(self.ekf.x.R).as_euler('xyz', degrees=True)
        current_yaw = current_rpy[2]
    
        # Store yaw history
        self.yaw_history.append(current_yaw)
        if len(self.yaw_history) > 100:
            self.yaw_history.pop(0)
        
        # Store yaw rate for consistency checking
        if len(self.yaw_history) >= 2:
            yaw_rate = (self.yaw_history[-1] - self.yaw_history[-2])
            # Handle wrap-around
            if yaw_rate > 180:
                yaw_rate -= 360
            elif yaw_rate < -180:
                yaw_rate += 360
            self.yaw_rate_history.append(abs(yaw_rate))
            if len(self.yaw_rate_history) > 50:
                self.yaw_rate_history.pop(0)
    
        # ===== REMOVED GT DEPENDENCY =====
        # No longer using ground truth for yaw error calculation
    
        # ===== NEW: Detect yaw instability from internal metrics =====
        high_yaw_error = False
        reasons = []
    
        # 1. Check if yaw is changing rapidly (unstable)
        if len(self.yaw_rate_history) > 10:
            recent_yaw_rate = np.mean(self.yaw_rate_history[-10:])
            if recent_yaw_rate > 30:  # More than 30° per frame is suspicious
                high_yaw_error = True
                reasons.append(f"high yaw rate ({recent_yaw_rate:.1f}°/frame)")
    
        # 2. Check velocity direction consistency
        if len(self.velocity_directions) > 10:
            # Calculate variance of velocity directions
            vel_dirs = np.array(self.velocity_directions[-10:])
            mean_dir = np.mean(vel_dirs, axis=0)
            mean_dir = mean_dir / (np.linalg.norm(mean_dir) + 1e-6)
            
            # Compute angular spread
            dot_products = np.clip(vel_dirs @ mean_dir, -1, 1)
            angles = np.arccos(dot_products) * 180/np.pi
            angular_spread = np.std(angles)
            
            # If velocity direction is highly variable, may indicate yaw error
            if angular_spread > 30:  # More than 30° spread
                high_yaw_error = True
                reasons.append(f"unstable velocity direction (spread {angular_spread:.1f}°)")
            
            # Also check if mean direction makes physical sense
            # For AUV, forward motion should dominate
            forward_component = abs(mean_dir[0])  # X-axis forward
            if forward_component < 0.7 and np.linalg.norm(self.ekf.x.v) > 0.5:
                # Moving but not mostly forward - possible yaw misalignment
                high_yaw_error = True
                reasons.append(f"low forward component ({forward_component:.2f})")
            
            print(f"  🧭 Velocity stats - forward: {forward_component:.2f}, "
                f"spread: {angular_spread:.1f}°, dir: [{mean_dir[0]:.2f}, {mean_dir[1]:.2f}, {mean_dir[2]:.2f}]")
    
        # 3. Check innovation consistency if available
        if hasattr(self.ekf, 'last_innov_norm') and self.ekf.last_innov_norm > 50:
            # High innovation might indicate orientation problems
            if hasattr(self, 'innovation_history') and len(self.innovation_history) > 5:
                recent_innov = np.mean([i for _, i, _ in self.innovation_history[-5:]])
                if recent_innov > 100:
                    high_yaw_error = True
                    reasons.append(f"high innovation ({recent_innov:.1f})")
    
        # 4. During early operation, be more aggressive
        if self.ekf.update_count < 50:
            # In bootstrap, we want faster convergence
            high_yaw_error = True
            reasons.append("bootstrap mode")
    
        # Set the flag with reason
        self.high_yaw_error_mode = high_yaw_error
        if high_yaw_error and reasons:
            print(f"  ⚠️ High yaw error mode: {', '.join(reasons)}")
        elif not high_yaw_error and self.frame_count % 50 == 0:
            print(f"  ✅ Yaw appears stable")
    
        # Optional: Monitor yaw convergence trend
        if len(self.yaw_history) > 20:
            recent_yaws = np.array(self.yaw_history[-20:])
            # Unwrap to handle 360° boundary
            recent_yaws_unwrapped = np.unwrap(recent_yaws * np.pi/180) * 180/np.pi
            yaw_trend = np.polyfit(np.arange(len(recent_yaws_unwrapped)), 
                                recent_yaws_unwrapped, 1)[0]
            
            if abs(yaw_trend) > 5:  # Drifting more than 5° per frame
                print(f"  ⚠️ Yaw drifting: {yaw_trend:.2f}°/frame")
    
    def filter_points_by_location(self, pts3d, pts2d):
        """Filter points - accept any points in front of camera"""
        if len(pts3d) == 0 or len(pts2d) == 0:
            print(f"[FILTER] Empty input: pts3d={len(pts3d)}, pts2d={len(pts2d)}")
            return np.array([]), np.array([])
       
        vehicle_pos = self.ekf.x.p.copy()
       
        print(f"\n=== LOCATION FILTER DEBUG ===")
        print(f"Vehicle pos: [{vehicle_pos[0]:.1f}, {vehicle_pos[1]:.1f}, {vehicle_pos[2]:.1f}]")
        print(f"Input points: {len(pts3d)}")
       
        # Transform world → body
        pts_body = (self.ekf.x.R.T @ (pts3d - vehicle_pos).T).T
       
        # FIXED: Consistent body→camera transformation using R_bc.T
        # With your corrected extrinsics, this will now give correct camera Z
        pts_cam = (self.R_bc @ (pts_body - self.t_bc).T).T
       
        # Check each condition separately
        in_front = pts_cam[:, 2] > 0.5  # Points in front have POSITIVE Z
        print(f"  In front (Z>0.5): {np.sum(in_front)}/{len(pts3d)}")
       
        # Distance threshold - keep at 1000m for underwater
        distances = np.linalg.norm(pts3d - vehicle_pos, axis=1)
        not_too_far = distances < 1000.0
        print(f"  Not too far (<1000m): {np.sum(not_too_far)}/{len(pts3d)}")
       
        not_too_close = distances > 1.0
        print(f"  Not too close (>1m): {np.sum(not_too_close)}/{len(pts3d)}")
       
        # Print some sample camera Z values
        if len(pts_cam) > 0:
            print(f"  Camera Z range: [{np.min(pts_cam[:,2]):.1f}, {np.max(pts_cam[:,2]):.1f}]")
            print(f"  Sample camera Z values (first 5): {pts_cam[:5,2]}")
       
        valid = in_front & not_too_far & not_too_close
       
        if np.sum(valid) < 6:
            print(f"[FILTER] Too few points ({np.sum(valid)}), skipping")
            return np.array([]), np.array([])
       
        print(f"[FILTER] ✅ Keeping {np.sum(valid)} points in front of camera")
        return pts3d[valid], pts2d[valid]

    def get_proj(self, R_world_body, p_world_body):
        """Get projection matrix for a given body pose"""
        # Transformation chain: World → Body → Camera
        # Camera position in world frame
        p_world_camera = p_world_body + R_world_body @ self.t_bc
        
        # Camera orientation in world frame
        R_world_camera = R_world_body @ self.R_bc.T  # Note: R_bc.T maps camera→body
        
        # For projection matrix P = K [R|t], we need [R|t] that maps world→camera
        R_camera_world = R_world_camera.T
        t_camera_world = -R_camera_world @ p_world_camera
        
        P = self.K_cam @ np.hstack([R_camera_world, t_camera_world.reshape(3, 1)])
        
        return P

    def project_points(self, pts3d_world):
        """Project 3D world points to image plane using current pose"""
        if len(pts3d_world) == 0:
            return np.array([])
    
        # Transform world → body
        pts_body = (self.ekf.x.R.T @ (pts3d_world - self.ekf.x.p).T).T
    
        # FIXED: Consistent body→camera transformation using R_bc (NOT R_bc.T)
        pts_cam = (self.R_bc @ (pts_body - self.t_bc).T).T
    
        # Project to image - points in front have POSITIVE Z
        valid = pts_cam[:, 2] > 0.1
        if not np.any(valid):
            return np.array([])
    
        u = np.zeros(len(pts_cam))
        v = np.zeros(len(pts_cam))
        u[valid] = self.K_cam[0, 0] * pts_cam[valid, 0] / pts_cam[valid, 2] + self.K_cam[0, 2]
        v[valid] = self.K_cam[1, 1] * pts_cam[valid, 1] / pts_cam[valid, 2] + self.K_cam[1, 2]

        return np.column_stack([u, v])
    
    def verify_extrinsics_consistency(self):
        """Comprehensive test of extrinsics consistency across all functions"""
        print("\n" + "="*60)
        print("EXTRINSICS CONSISTENCY VERIFICATION")
        print("="*60)
        
        # Create a test point 10m ahead in body frame
        test_point_body = np.array([10, 0, 0])
        
        # Method 1: Direct transformation using R_bc
        in_camera_direct = self.R_bc @ test_point_body
        print(f"\n1. Direct R_bc @ body_forward:")
        print(f"   Result: {in_camera_direct}")
        print(f"   Camera Z: {in_camera_direct[2]:.3f} (should be POSITIVE)")
        
        # Method 2: Through get_proj and triangulation
        R_identity = np.eye(3)
        p_identity = np.array([0, 0, -50])  # 50m deep
        
        P = self.get_proj(R_identity, p_identity)
        
        # Transform point to world
        point_world = R_identity @ test_point_body + p_identity
        
        # Project using P
        point_hom = np.append(point_world, 1)
        proj = P @ point_hom
        
        if proj[2] > 0:
            u = proj[0] / proj[2]
            v = proj[1] / proj[2]
            print(f"\n2. Through projection matrix:")
            print(f"   Projects to pixel: ({u:.1f}, {v:.1f})")
            print(f"   Should be near image center: (384.5, 246.5)")
        
        # Method 3: Through filter_points_by_location logic
        pts_body = test_point_body.reshape(1, 3)
        pts_cam_filter = (self.R_bc @ (pts_body - self.t_bc).T).T
        print(f"\n3. Through filter_points_by_location:")
        print(f"   Result: {pts_cam_filter[0]}")
        print(f"   Camera Z: {pts_cam_filter[0,2]:.3f}")
        
        # Check if all methods agree
        print("\n✅ If all methods show positive Z and consistent values, extrinsics are correct!")
        print("="*60)
        
    def filter_outliers_ransac(self, pts3d, pts2d, max_trials=500):
        """RANSAC with adaptive thresholds based on point distances"""
        if len(pts3d) < 10:
            return pts3d, pts2d
       
        # Get vehicle position
        vehicle_pos = self.ekf.x.p.copy()
       
        # ===== FIXED: Use relative depth with tolerance =====
        # Points should be 5-100m BELOW the vehicle
        depth_diff = vehicle_pos[2] - pts3d[:, 2]  # Positive = point below
        valid_depth = (depth_diff > 5.0) & (depth_diff < 100.0)
       
        depth_valid_count = np.sum(valid_depth)
        print(f"  [RANSAC] Points 5-100m below vehicle: {depth_valid_count}/{len(pts3d)}")
        print(f"  [RANSAC] Depth differences: min={np.min(depth_diff):.1f}, max={np.max(depth_diff):.1f}")
       
        if depth_valid_count >= 6:
            pts3d = pts3d[valid_depth]
            pts2d = pts2d[valid_depth]
            print(f"  [RANSAC] Keeping {len(pts3d)} points with reasonable depth")
        else:
            print(f"  [RANSAC] Too few points with reasonable depth ({depth_valid_count}), using all but will likely fail")
       
        # Calculate point distances
        distances = np.linalg.norm(pts3d - vehicle_pos, axis=1)
       
        # Adaptive threshold: farther points can have larger errors
        base_threshold = 30.0  # pixels
        distance_factor = np.clip(distances / 50.0, 1.0, 3.0)
       
        try:
            # Project points
            pts2d_proj = self.project_points(pts3d)
           
            # Calculate errors
            errors = np.linalg.norm(pts2d - pts2d_proj, axis=1)
           
            # Adaptive thresholds per point
            inlier_mask = errors < (base_threshold * distance_factor)
           
            # Keep at least 70% of points, never fewer than 8
            min_points = max(8, int(len(pts3d) * 0.7))
            if np.sum(inlier_mask) < min_points:
                # Sort by error and take the best 70%
                sorted_idx = np.argsort(errors)
                inlier_mask = np.zeros(len(pts3d), dtype=bool)
                inlier_mask[sorted_idx[:min_points]] = True
                print(f"[RANSAC] Keeping top {min_points}/{len(pts3d)} points ({min_points/len(pts3d)*100:.0f}%)")
            else:
                print(f"[RANSAC] Keeping {np.sum(inlier_mask)}/{len(pts3d)} inliers ({np.sum(inlier_mask)/len(pts3d)*100:.0f}%)")
                
            return pts3d[inlier_mask], pts2d[inlier_mask]
           
        except Exception as e:
            print(f"[RANSAC] Error: {e}, using all points")
            return pts3d, pts2d
   
    def monitor_hydrodynamic_parameters(self):
        """Monitor and log hydrodynamic damping coefficients"""
        if hasattr(self.ekf.x, 'Y_v') and self.frame_count % 50 == 0:
            print(f"\n🌊 HYDRODYNAMIC PARAMETERS:")
            print(f"  Y_v (sway damping): {self.ekf.x.Y_v:.1f}")
           
            # Check if Y_v is converging to reasonable value
            if self.ekf.x.Y_v > -10:
                print(f"  ⚠️ Y_v too small (magnitude), should be more negative")
            elif self.ekf.x.Y_v < -500:
                print(f"  ⚠️ Y_v too large (magnitude), check estimation")
            else:
                print(f"  ✅ Y_v in reasonable range")
           
            # Also monitor Y_v variance
            if hasattr(self.ekf, 'P') and self.ekf.P.shape[0] > 16:
                Yv_var = self.ekf.P[16, 16]
                print(f"  Y_v variance: {Yv_var:.3f}")
               

    def check_and_apply_zupt(self):
        """Apply zero-velocity update when truly stationary"""
        if not self.origin_initialized:
            return False
       
        # Check if vehicle is stationary using multiple cues
        accel_raw = self.latest_imu[3:6]
        gyro_raw = self.latest_imu[0:3]
       
        # Remove estimated biases
        accel_corrected = accel_raw - self.ekf.x.ba
        gyro_corrected = gyro_raw - self.ekf.x.bg
       
        # Stationary conditions
        gravity_mag = 9.793
        accel_mag = np.linalg.norm(accel_corrected)
        gyro_mag = np.linalg.norm(gyro_corrected)
        vel_mag = np.linalg.norm(self.ekf.x.v)
       
        is_stationary = (
            abs(accel_mag - gravity_mag) < 0.2 and  # Close to gravity
            gyro_mag < 0.03 and  # No rotation
            vel_mag < 0.1  # Low velocity
        )
       
        if is_stationary:
            # Zero-velocity update
            # Measurement: velocity should be zero
            y = -self.ekf.x.v  # Innovation
           
            # Jacobian: H = [0 0 0 I 0 0 0 0] for 17-state
            H = np.zeros((3, 17))
            H[:, 3:6] = np.eye(3)  # Velocity part
           
            # Low noise for stationary
            R = np.eye(3) * 0.01
           
            # Apply update
            success, K, delta, _ = self.ekf.update_generic(y, H, R, debug=False)
           
            if success:
                print(f"[ZUPT] Applied correction, velocity from {vel_mag:.3f} to {np.linalg.norm(self.ekf.x.v):.3f} m/s")
                return True
       
        return False

    def test_extrinsics(self):
        """Test if extrinsics correctly map a point below vehicle to camera frame"""
        # Vehicle at origin, looking straight down
        R_wb = np.eye(3)
        p_wb = np.array([0, 0, -50])  # 50m deep
       
        # Point 10m below vehicle (at -60m depth)
        point_below = np.array([0, 0, -60])  # In world frame
       
        # Transform to body frame (should be [0, 0, -10] relative to vehicle)
        point_body = R_wb.T @ (point_below - p_wb)  # = [0, 0, -10]
       
        # ===== FIXED: Remove the .T =====
        point_camera = self.R_bc @ (point_body - self.t_bc)  # Use R_bc, not R_bc.T!
       
        print(f"\n=== EXTRINSICS TEST (FIXED) ===")
        print(f"Point 10m below vehicle in body frame: {point_body}")
        print(f"Point in camera frame: {point_camera}")
        print(f"Camera Z (should be positive for in front): {point_camera[2]:.3f}")
       
        # Project to image
        if point_camera[2] > 0:
            u = self.K_cam[0,0] * point_camera[0]/point_camera[2] + self.K_cam[0,2]
            v = self.K_cam[1,1] * point_camera[1]/point_camera[2] + self.K_cam[1,2]
            print(f"Projects to: ({u:.1f}, {v:.1f})")
           
    def force_orientation_correction(self, vision_delta=None):
        """Stronger gravity alignment for BOTH roll and pitch - FIXED"""
        corrected = False
       
        if len(self.latest_imu) == 6:
            acc_body = self.latest_imu[3:6] - self.ekf.x.ba
           
            # Normalize to get gravity direction in body frame
            gravity_dir_body = acc_body / (np.linalg.norm(acc_body) + 1e-6)
           
            # Expected gravity direction in body frame (should be R.T @ [0,0,1])
            expected_gravity_body = self.ekf.x.R.T @ np.array([0, 0, 1])
           
            # Compute individual axis errors
            roll_error = np.arctan2(gravity_dir_body[1], gravity_dir_body[2]) - \
                        np.arctan2(expected_gravity_body[1], expected_gravity_body[2])
            pitch_error = np.arctan2(-gravity_dir_body[0], gravity_dir_body[2]) - \
                        np.arctan2(-expected_gravity_body[0], expected_gravity_body[2])
           
            print(f"  📐 Roll error: {np.degrees(roll_error):.2f}°, Pitch error: {np.degrees(pitch_error):.2f}°")
           
            # Create rotation correction (first around X, then around Y)
            from scipy.spatial.transform import Rotation as R
           
            # Apply correction (30% of the error)
            alpha = 0.3
            R_correction = R.from_euler('xyz', [roll_error * alpha, pitch_error * alpha, 0]).as_matrix()
           
            self.ekf.x.R = R_correction @ self.ekf.x.R
            self.ekf.x.theta = np.zeros(3)
            print(f"  ✅ Applied {alpha*100:.0f}% gravity correction")
            corrected = True
   
        return corrected

    def test_y_axis_sign(self):
        """Test if Y-axis has correct sign"""
        print("\n=== Y-AXIS SIGN TEST ===")
       
        # Create a point that should be to the RIGHT of the vehicle in body frame
        # Body Y positive = right
        test_point_body_right = np.array([10, 5, -10])  # 10m forward, 5m right, 10m down
       
        # Transform to world using current pose
        test_point_world = self.ekf.x.R @ test_point_body_right + self.ekf.x.p
       
        # Now project this point into camera
        # If Y sign is wrong, this point will appear on LEFT side of image
        R_wc = self.ekf.x.R @ self.R_bc
        p_wc = self.ekf.x.p + self.ekf.x.R @ self.t_bc
        R_cw = R_wc.T
        t_cw = -R_cw @ p_wc
       
        p_c = R_cw @ test_point_world + t_cw
       
        if p_c[2] > 0:
            u = self.K_cam[0,0] * p_c[0]/p_c[2] + self.K_cam[0,2]
            v = self.K_cam[1,1] * p_c[1]/p_c[2] + self.K_cam[1,2]
           
            print(f"Point 5m right of vehicle projects to u={u:.1f}")
            print(f"Should be >384.5 (right side of image): {'✓' if u > 384.5 else '✗'}")
           
            if u < 384.5:
                print("🚨 Y-AXIS SIGN ERROR DETECTED!")
                print("A point on the RIGHT appears on the LEFT side of image")

    #=============observability================================
    def compute_observability_svd(self, H_window):
        """
        Compute SVD of stacked observability matrix over a window.
        Returns singular values and null space (right singular vectors for near-zero singular values).
        """
        if len(H_window) == 0:
            return None, None
        O = np.vstack(H_window)                # stack all H matrices
        U, s, Vh = np.linalg.svd(O, full_matrices=False)
        # Identify unobservable directions (singular values below threshold)
        threshold = 1e-3 * s[0]                 # relative to largest singular value
        unobservable_idx = np.where(s < threshold)[0]
        null_space = Vh[unobservable_idx, :] if len(unobservable_idx) > 0 else None
        return s, null_space

    def project_onto_nullspace(self, vec, null_space):
        """
        Project a vector (e.g., velocity error) onto the unobservable subspace.
        
        Args:
            vec: vector to project (should be full state vector of length 17, 
                or at least match null_space columns)
            null_space: matrix where rows are unobservable directions (k x n)
        
        Returns:
            norm of the projection
        """
        if null_space is None:
            return 0.0
        
        # null_space is (k, n) where k = number of unobservable directions, n = state size (17)
        # vec should be of length n (17)
        
        # If vec is not the right size, pad it with zeros
        if len(vec) < null_space.shape[1]:
            # Pad with zeros for missing dimensions
            vec_padded = np.zeros(null_space.shape[1])
            vec_padded[:len(vec)] = vec
            projection = null_space @ vec_padded
        elif len(vec) > null_space.shape[1]:
            # Truncate if too long
            projection = null_space @ vec[:null_space.shape[1]]
        else:
            # Perfect match
            projection = null_space @ vec
        
        return np.linalg.norm(projection)

    def compute_nis(self, y, S):
        """Normalized Innovation Squared (Mahalanobis distance)."""
        try:
            Sinv = np.linalg.inv(S)
            nis = y @ Sinv @ y
            return nis
        except:
            return np.inf

    def check_innovation_consistency(self, nis, dof):
        """Check if NIS is within chi-square confidence interval (95%)."""
        from scipy.stats import chi2
        threshold = chi2.ppf(0.95, dof)
        return nis < threshold

    def update_vio_health(self, success, innov_norm, num_points, nis=None):
        """
        Update health score (0-1) based on recent performance.
        Called after each update attempt (success or failure).
        """
        # Base health decays slowly over time
        self.vio_health *= 0.99

        if success:
            # Boost health based on quality
            quality = 0.0
            if innov_norm < 50:
                quality += 0.3
            if num_points >= 10:
                quality += 0.3
            if nis is not None and nis < 100:
                quality += 0.4
            self.vio_health = min(1.0, self.vio_health + quality * 0.1)
        else:
            # Penalty for failure
            self.vio_health *= 0.95

        # Clamp
        self.vio_health = np.clip(self.vio_health, 0.0, 1.0)

        # Switch fallback mode
        self.fallback_mode = self.vio_health < 0.3
        if self.fallback_mode:
            rospy.logwarn_throttle(5.0, "VIO health low, using model-based fallback")

    def model_based_propagation(self, dt):
        """
        Simple dynamic model: constant velocity with hydrodynamic damping.
        Called when fallback_mode is True – we skip the EKF update and just damp velocity.
        """
        # Use the damping coefficient from the EKF state if available, otherwise a small default
        damping = np.array([0.0, self.ekf.x.Y_v * 0.1, 0.0])   # example: only Y damped
        # Apply exponential decay to velocity
        self.ekf.x.v *= np.exp(damping * dt)
        # Update position using damped velocity
        self.ekf.x.p += self.ekf.x.v * dt
        # Increase covariance to reflect uncertainty (optional)
        self.ekf.P[3:6, 3:6] *= 1.05
        # Orientation remains unchanged (no gyro integration in fallback)
        
    #==========================================================
    def image_callback(self, msg):
        """Enhanced image callback with robust Y-axis bias handling and health monitoring"""
        if not self.origin_initialized:
            return

        t = msg.header.stamp.to_sec()

        # Initialize tracking variables at first frame
        if not hasattr(self, 'start_time'):
            self.start_time = t
            self.consecutive_stationary = 0
            self.motion_detected = False
            self.last_vision_update = t
            self.vision_update_count = 0
            self.consecutive_vision_failures = 0
            self.y_error_history = []
            self.scale_history = []
            self.scale_error_history = []
            self.bias_error_history = []
            self.scale_boost_factor = 1.0
            self.bias_correction_rate = 0.001
            self.last_scale_adjust = time.time()
            self.vision_health = {
                'consecutive_success': 0,
                'consecutive_failures': 0,
                'total_success': 0,
                'total_failures': 0
            }
            # Add bootstrap mode flag
            self.bootstrap_mode = True
            self.bootstrap_frame_count = 0

        if self.last_img_t is None:
            self.last_img_t = t
            return

        if len(self.ts_buf) > 1:
            dt = t - self.ts_buf[-2]
            if dt > 10.0:  # More than 10 seconds between frames
                print(f"  ⚠️ Large frame gap: {dt:.1f}s - features may have changed significantly")

        # ===== IMAGE PROCESSING =====
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return

        # Extract features
        img_t = torch.from_numpy(self.latest_frame).permute(2, 0, 1).float().unsqueeze(0).to(self.device)/255.0
        with torch.no_grad():
            enhanced, _ = self.visual_pipe(img_t)
            coords, desc, mask = self.orb_extractor(enhanced)
        if coords[0].shape[0] == 0:
            print(f"  ⚠️ No keypoints extracted - skipping frame")
            return
        current_kps = coords[0].cpu().numpy().astype(np.float32)
        current_desc = desc[0].cpu().numpy().astype(np.uint8)

        self.ts_buf.append(t)
        self.ekf_poses_buffer.append(self.ekf.x.R.copy())
        self.ekf_positions_buffer.append(self.ekf.x.p.copy())
        self.kps_buf.append(current_kps)
        self.descs_buf.append(current_desc)

        print(f"  [BUFFER] ts={len(self.ts_buf)}, kps={len(self.kps_buf)}, poses={len(self.ekf_poses_buffer)}")

        # Limit buffer size
        max_buffer = 50
        if len(self.kps_buf) > max_buffer:
            self.kps_buf.pop(0)
            self.descs_buf.pop(0)
            self.ekf_poses_buffer.pop(0)
            self.ekf_positions_buffer.pop(0)
            self.ts_buf.pop(0)

            if len(self.kps_buf) != len(self.ts_buf) or len(self.kps_buf) != len(self.ekf_poses_buffer):
                print(f"  🔥 BUFFER DESYNC DETECTED! Resetting all buffers")
                self.kps_buf = []
                self.descs_buf = []
                self.ekf_poses_buffer = []
                self.ekf_positions_buffer = []
                self.ts_buf = []

        # ===== MOTION STATE ESTIMATION =====
        current_vel = np.linalg.norm(self.ekf.x.v)
        accel_raw = np.linalg.norm(self.latest_imu[3:6]) if len(self.latest_imu) == 6 else 0

        is_moving = (
            current_vel > 0.1 or
            abs(accel_raw - 9.793) > 0.5 or
            (len(self.ts_buf) > 1 and (t - self.ts_buf[-2]) * current_vel > 0.05)
        )

        if is_moving:
            self.motion_detected = True
            self.consecutive_stationary = 0
        else:
            self.consecutive_stationary += 1
            if self.consecutive_stationary > 5:
                self.motion_detected = False

        acc_mag, acc_var, is_zupt = self.get_motion_features(self.latest_imu)
        
        # Update current metrics for AKIT
        self.current_metrics['accel_mag'] = acc_mag
        self.current_metrics['accel_var'] = acc_var
        self.current_metrics['is_zupt'] = is_zupt
        
        # Velocity sanity check
        MAX_VEL = 10.0  # m/s – adjust for your AUV
        if np.linalg.norm(self.ekf.x.v) > MAX_VEL:
            print(f"\n🚨 Velocity sanity triggered: {np.linalg.norm(self.ekf.x.v):.1f} m/s – resetting")
            self.ekf.x.v = np.zeros(3)
            self.ekf.P[3:6, 3:6] = np.eye(3) * 100.0   # large covariance to allow fast correction
            self.ekf.P[12:15, 12:15] = np.eye(3) * 10.0 # also increase bias uncertainty
            # Optionally, reset fallback timer to force a recovery attempt
            self.last_fallback_attempt = time.time()
            # Also check if orientation is completely wrong
            rpy = R_tool.from_matrix(self.ekf.x.R).as_euler('xyz', degrees=True)
            if abs(rpy[0]) > 45 or abs(rpy[1]) > 45:  # >45° tilt is extreme for AUV
                print(f"  🔄 Also resetting orientation (roll={rpy[0]:.1f}°, pitch={rpy[1]:.1f}°)")
                # Reset to identity orientation (or use gravity if available)
                if hasattr(self, 'latest_imu') and len(self.latest_imu) == 6:
                    # Simple gravity alignment
                    acc = self.latest_imu[3:6]
                    acc_norm = acc / np.linalg.norm(acc)
                    # This is a quick hack - a proper solution would compute orientation from gravity
                    self.ekf.x.R = np.eye(3)
                else:
                    self.ekf.x.R = np.eye(3)
            
            self.ekf.x.v = np.zeros(3)
            self.ekf.P[3:6, 3:6] = np.eye(3) * 100.0
            self.ekf.P[0:3, 0:3] = np.eye(3) * 10.0  # Also increase orientation uncertainty
            self.ekf.P[12:15, 12:15] = np.eye(3) * 10.0
            self.ekf.P[15, 15] = 1.0

        status = "IDLE"
        matches = 0
        pts3d_filt = np.array([])
        innov_norm = 0.0
        if len(self.kps_buf) >= 2:
            dt = self.ts_buf[-1] - self.ts_buf[-2]
            self.update_expected_flow(dt)

        # ===== FALLBACK MODE CHECK =====
        # If health is too low, use model‑based propagation and skip vision processing
        if self.fallback_mode:
            if time.time() - self.last_fallback_attempt > self.fallback_cooldown:
                # Try to recover
                self.last_fallback_attempt = time.time()
                # Let vision processing run
            else:
                # Stay in fallback
                dt_fb = self.ts_buf[-1] - self.ts_buf[-2] if len(self.ts_buf) >= 2 else 0.1
                self.model_based_propagation(dt_fb)
                status = "FALLBACK"
                self.update_vio_health(False, 999, 0)
                # Skip to post-processing
                matches = 0
                innov_norm = 0.0
                # Skip to after the vision processing block
                # We'll use a label or simply jump by placing the rest of the code after an if‑else.
                # For clarity, we'll structure the rest with an else that runs only when not in fallback.
                # However, to avoid deep indentation, we'll use a conditional block.
        if self.fallback_mode and time.time() - self.last_fallback_attempt > 30.0:
            # Force an attempt to recover
            self.last_fallback_attempt = time.time()

        # ===== VISION PROCESSING (only if not in fallback or when attempting recovery) =====
        if not self.fallback_mode or (time.time() - self.last_fallback_attempt <= self.fallback_cooldown):
            # The original vision processing block goes here (indented)
            if len(self.kps_buf) >= 2 and self.motion_detected:
                current_pos_var = np.max(np.diag(self.ekf.P)[6:9])
                if current_pos_var > 1000:
                    print(f"\n[COVARIANCE RESET] Position variance too large: {current_pos_var:.1f} m²")
                    self.ekf.reset_covariance()

                R_wc = self.ekf.x.R @ self.R_bc
                camera_forward_world = R_wc[2, :]
                cam_z_component = camera_forward_world[2]
                expected_z = self.R_bc[2, 2]

                if abs(cam_z_component - expected_z) > 0.5:
                    print(f"[VISION-SKIP] Camera orientation wrong: Z={cam_z_component:.3f} (expected {expected_z:.3f})")
                    status = "CAMERA_ORIENTATION_BAD"
                else:
                    thresholds = self.get_adaptive_thresholds()

                    idx_i, idx_j = self.get_frame_indices()
                    if idx_i is None or idx_j is None:
                        status = "INVALID_FRAME_PAIR"
                        print(f"  [SKIP] No valid frame pair found")
                    else:
                        if (idx_i < 0 or idx_i >= len(self.ts_buf) or idx_j < 0 or idx_j >= len(self.ts_buf) or
                            idx_i >= len(self.kps_buf) or idx_j >= len(self.kps_buf) or
                            idx_i >= len(self.ekf_poses_buffer) or idx_j >= len(self.ekf_poses_buffer)):
                            print(f"  [ERROR] Invalid indices: {idx_i}, {idx_j} (ts_buf={len(self.ts_buf)}, "
                                f"kps_buf={len(self.kps_buf)}, poses={len(self.ekf_poses_buffer)})")
                            status = "BUFFER_MISMATCH"
                        else:
                            dt = self.ts_buf[idx_j] - self.ts_buf[idx_i]
                            print(f"\n=== FRAME TIMING ===")
                            print(f"  Frame {idx_i}: {self.ts_buf[idx_i]:.3f}s")
                            print(f"  Frame {idx_j}: {self.ts_buf[idx_j]:.3f}s")
                            print(f"  dt: {dt:.3f}s")

                            # In your image_callback, where you call match_orb_enhanced:

                            # Determine quality threshold based on mode
                            if self.bootstrap_mode or self.vision_update_count < 20:
                                min_quality = 0.02  # Very permissive during bootstrap
                                print(f"  [MATCH] Bootstrap mode: using low quality threshold {min_quality}")
                            elif self.consecutive_vision_failures > 10:
                                min_quality = 0.03  # Desperate mode
                                print(f"  [MATCH] Desperate mode: using low quality threshold {min_quality}")
                            else:
                                min_quality = 0.05  # Normal operation

                            m_res = match_orb_enhanced(
                                self.kps_buf[idx_i], self.descs_buf[idx_i],
                                self.kps_buf[idx_j], self.descs_buf[idx_j],
                                dt=dt,
                                velocity=current_vel,
                                angular_velocity=np.linalg.norm(self.latest_imu[0:3]),
                                focal_length=self.K_cam[0,0],
                                min_quality_score=min_quality
                            )

                            if m_res is None:
                                status = "NO_MATCHES"
                                print(f"  [NO_MATCHES] No good matches found")
                            else:
                                pts1, pts2, _ = m_res
                                matches = len(pts1)
                                self.measure_optical_flow(pts1, pts2)
                                print(f"  [MATCH] Using {matches} high-quality matches")

                                if matches < thresholds['min_points']:
                                    status = "INSUFFICIENT_QUALITY_MATCHES"
                                    print(f"  [INSUFFICIENT_QUALITY] Only {matches} good matches, need {thresholds['min_points']}")
                                else:
                                    R_i = self.ekf_poses_buffer[idx_i]
                                    p_i = self.ekf_positions_buffer[idx_i]
                                    R_j = self.ekf_poses_buffer[idx_j]
                                    p_j = self.ekf_positions_buffer[idx_j]

                                    expected_baseline = current_vel * dt
                                    ekf_baseline = np.linalg.norm(p_j - p_i)

                                    print(f"\n=== BASELINE CHECK ===")
                                    print(f"  Expected baseline (vel*dt): {expected_baseline:.3f}m")
                                    print(f"  EKF baseline (pose diff): {ekf_baseline:.3f}m")
                                    print(f"  Ratio: {ekf_baseline/expected_baseline:.2f}x")

                                    if expected_baseline < thresholds['min_baseline']:
                                        status = "SMALL_BASELINE"
                                        print(f"  ⚠️ Baseline too small: {expected_baseline:.3f}m")
                                    else:
                                        P1 = self.get_proj(R_i, p_i)
                                        P2 = self.get_proj(R_j, p_j)

                                        tri_threshold = 30.0
                                        tri_min_angle = 1.0
                                        if self.vision_update_count > 10:
                                            tri_threshold = 15.0      # stricter reprojection error
                                            tri_min_angle = 2.0       # require larger parallax

                                        pts3d_world, pts1_clean, pts2_clean = robust_triangulate(
                                            pts1, pts2, P1, P2,
                                            threshold_px=tri_threshold,
                                            min_angle=tri_min_angle,
                                            vehicle_pos=self.ekf.x.p
                                        )

                                        print(f"\n[ROBUST TRIANGULATION]")
                                        print(f"  Original: {len(pts1)} → After robust: {len(pts3d_world)}")

                                        if len(pts3d_world) < 6:
                                            status = "INSUFFICIENT_TRIANGULATION"
                                            print(f"  [INSUFFICIENT] Only {len(pts3d_world)} points")
                                        else:
                                            vehicle_depth = self.ekf.x.p[2]
                                            depth_below = vehicle_depth - pts3d_world[:, 2]

                                            print(f"\n[DEPTH CHECK]")
                                            print(f"  Vehicle depth: {vehicle_depth:.1f}m")
                                            print(f"  Points below vehicle: {np.sum(pts3d_world[:,2] < vehicle_depth)}/{len(pts3d_world)}")

                                            scores = []
                                            for i in range(len(pts3d_world)):
                                                score, reasons = score_triangulated_point(
                                                    pts3d_world[i], pts1_clean[i], pts2_clean[i],
                                                    P1, P2, self.ekf.x.p
                                                )
                                                scores.append((score, i, reasons))

                                            scores.sort(reverse=True)
                                            top_count = max(12, int(len(scores) * 0.7))
                                            top_indices = [idx for _, idx, _ in scores[:top_count]]

                                            pts3d_selected = pts3d_world[top_indices]
                                            pts2_selected = pts2_clean[top_indices]

                                            print(f"\n[SCORING] Selected {len(pts3d_selected)}/{len(pts3d_world)} points")

                                            pts3d_filtered, pts2_filtered = self.filter_points_by_location(
                                                pts3d_selected, pts2_selected
                                            )

                                            min_required = thresholds['min_points']
                                            if len(pts3d_filtered) < min_required:
                                                # During bootstrap or when desperate, accept fewer points
                                                if (self.bootstrap_mode or self.vision_update_count < 20) and len(pts3d_filtered) >= 4:
                                                    print(f"  ⚠️ Only {len(pts3d_filtered)} points (need {min_required}), but trying anyway (bootstrap)")
                                                    # Continue processing - don't set status to fail
                                                else:
                                                    status = "LOCATION_FILTER_FAIL"
                                                    print(f"  [LOCATION_FILTER_FAIL] Only {len(pts3d_filtered)} points")
                                
                                            else:
                                                pts3d_ransac, pts2_ransac = self.filter_outliers_ransac(
                                                    pts3d_filtered, pts2_filtered
                                                )

                                                if len(pts3d_ransac) < 6:
                                                    status = "RANSAC_FAIL"
                                                    print(f"  [RANSAC_FAIL] Only {len(pts3d_ransac)} points")
                                                else:
                                                    print(f"\n[UPDATE] {len(pts3d_ransac)} points")
                                                    depths = np.linalg.norm(pts3d_ransac - self.ekf.x.p, axis=1)
                                                    if not hasattr(self, 'last_triangulation_depth'):
                                                        self.last_triangulation_depth = []
                                                    self.last_triangulation_depth.extend(depths.tolist())
                                                    if len(self.last_triangulation_depth) > 200:
                                                        self.last_triangulation_depth = self.last_triangulation_depth[-200:]

                                                    high_yaw_error = getattr(self, 'high_yaw_error_mode', False)
                                                    
                                                    # ===== MODIFIED: Use adaptive pixel sigma from AKIT =====
                                                    pixel_sigma = thresholds['pixel_sigma']
                                                    if self.use_akit:
                                                        pixel_sigma = self.get_adaptive_R(base_R=thresholds['pixel_sigma'])
                                                    # ===== END MODIFIED =====
                                                    
                                                    success, K, delta, metrics = self.ekf.update_from_reprojection(
                                                        pts3d_ransac, pts2_ransac,
                                                        (self.K_cam[0,0], self.K_cam[1,1], self.K_cam[0,2], self.K_cam[1,2]),
                                                        self.R_bc, self.t_bc, pixel_sigma,  # Use adaptive value
                                                        high_yaw_error=high_yaw_error
                                                    )

                                                    # <-- NEW: After update attempt, update health and possibly store observability data
                                                    if success: 
                                                        innov_norm = metrics.get('innov_norm', 0.0)
                                                        num_points = metrics.get('num_points', 0)
                                                        
                                                        # Store these in the EKF for the adaptive scaling to use
                                                        self.ekf.last_innov_norm = innov_norm
                                                        self.ekf.last_num_points = num_points
                                                        
                                                        # Also update history if your EKF has that method
                                                        if hasattr(self.ekf, 'update_filter_health_metrics'):
                                                            self.ekf.update_filter_health_metrics(innov_norm, num_points)

                                                        y = metrics.get('innovation')
                                                        S = metrics.get('innovation_cov')
                                                        H_final = metrics.get('jacobian')
                                                        
                                                        # Store Jacobian for observability analysis
                                                        if H_final is not None:
                                                            self.observability_window.append(H_final)
                                                            if len(self.observability_window) > self.max_obs_window:
                                                                self.observability_window.pop(0)

                                                        # Compute NIS (Normalized Innovation Squared)
                                                        nis = None
                                                        if y is not None and S is not None:
                                                            nis = self.compute_nis(y, S)
                                                            dof = H_final.shape[0] if H_final is not None else len(y)
                                                            self.innovation_history.append((time.time(), nis, dof))
                                                            if len(self.innovation_history) > 100:
                                                                self.innovation_history.pop(0)

   
                                                        # Update health with success
                                                        self.update_vio_health(True, metrics.get('innov_norm', 999), metrics.get('num_points', 0), nis)
                                                    else:
                                                        # Update health with failure
                                                        self.update_vio_health(False, 999, 0)

                                                    # End of success/failure handling
                                                    # (original code continues)
                                                    if success:
                                                        self.vision_health['consecutive_success'] += 1
                                                        self.vision_health['consecutive_failures'] = 0
                                                        self.vision_health['total_success'] += 1
                                                        self.consecutive_vision_failures = 0
                                                        self.vision_update_count += 1
                                                        self.last_valid_points = metrics.get('num_points', 0)
                                                        
                                                        # CRITICAL: Update these tracking variables!
                                                        self.last_innov_norm = metrics.get('innov_norm', 0.0)
                                                        self.last_s_trace = metrics.get('s_trace', 0.0)
                                                        self.last_s_cond = metrics.get('s_cond', 0.0)
                                                        self.last_h_norm = metrics.get('h_norm', 0.0)
                                                        self.last_num_points = metrics.get('num_points', 0)
                                                        
                                                        self.last_update_p = self.ekf.x.p.copy()
                                                        self.last_vision_update = t

                                                        print(f"\n📊 POST-UPDATE DIAGNOSTICS:")
                                                        print(f"  Points used: {metrics.get('num_points', 0)}")
                                                        print(f"  Innovation norm: {metrics.get('innov_norm', 0):.3f}")
                                                        print(f"  Rotation update: {metrics.get('rot_update_deg', 0):.2f}°")
                                                        print(f"  Position update: {metrics.get('pos_update', 0):.3f}m")
                                                        print(f"  Scale delta: {metrics.get('scale_update', 0):.5f}")
                                                        print(f"  New accel bias: {self.ekf.x.ba}")

                                                        status = "SUCCESS"
                                                        innov_norm = metrics.get('innov_norm', 0.0)
                                                        num_points_used = metrics.get('num_points', 0)
                                                        print(f"  ✓ SUCCESS! innov={innov_norm:.2f}, points={num_points_used}")

                                                        match_orb_enhanced.last_R = self.ekf.x.R.copy()
                                                    else:
                                                        status = metrics.get('status', 'REJECT')
                                                        print(f"  ✗ REJECT: {status}")
                                                        print(f"  📊 REJECTED UPDATE DIAGNOSTICS:")
                                                        print(f"    Status: {status}")
                                                        print(f"    Points attempted: {len(pts3d_ransac)}")
                                                        if 'invalid_reasons' in metrics:
                                                            print(f"    Invalid reasons: {metrics['invalid_reasons']}")

            # ===== PERIODIC OBSERVABILITY ANALYSIS =====
            if self.frame_count % 50 == 0 and len(self.observability_window) > 10:
                s, null_space = self.compute_observability_svd(self.observability_window)
                if null_space is not None and hasattr(self, 'gt_v'):
                    vel_error = self.ekf.x.v - self.gt_v
                    
                    # Create a full state error vector (17 elements) with zeros except for velocity
                    full_state_error = np.zeros(17)
                    full_state_error[3:6] = vel_error  # Velocity is at indices 3-5
                    
                    proj = self.project_onto_nullspace(full_state_error, null_space)
                    print(f"  [OBSERVABILITY] Velocity error projection onto null space: {proj:.3f}")
                    if proj > 1.0:
                        print(f"  ⚠️ Large velocity error in unobservable direction!")
        
        # ===== CONTINUE WITH EXISTING POST-PROCESSING =====
        # Store velocity history (do this every frame, regardless of success)
        if not hasattr(self, 'velocity_history'):
            self.velocity_history = []
        self.velocity_history.append((self.ekf.x.v[0], self.ekf.x.v[1], self.ekf.x.v[2]))
        if len(self.velocity_history) > 500:
            self.velocity_history.pop(0)

        # Store IMU position change for consistency check
        if hasattr(self, 'last_imu_position') and self.last_imu_t is not None:
            dt = t - self.last_imu_t
            if dt > 0 and dt < 0.1:  # Only for small dt
                self.last_imu_position_change = self.ekf.x.v * dt

        # ===== HANDLE VISION FAILURES =====
        if status != "SUCCESS":
            self.vision_health['consecutive_failures'] += 1
            self.vision_health['consecutive_success'] = 0
            self.vision_health['total_failures'] += 1
            self.consecutive_vision_failures += 1

            failures = self.consecutive_vision_failures
            if failures > 3 and not self.bootstrap_mode:
                if failures > 30:
                    damping_factor = 0.5
                elif failures > 20:
                    damping_factor = 0.7
                elif failures > 10:
                    damping_factor = 0.8
                elif failures > 5:
                    damping_factor = 0.9
                else:
                    damping_factor = 0.95

                self.ekf.x.v[1] *= damping_factor
                self.ekf.P[13, 13] *= (1.0 + 0.05 * failures)
                self.ekf.P[15, 15] *= (1.0 + 0.05 * failures)
                self.ekf.P[3:6, 3:6] *= (1.0 + 0.02 * failures)
                print(f"  🌊 VISION FAILURE DAMPING ({failures}): Y-vel reduced by {1-damping_factor:.0%}")

            '''if failures > 50 and not self.bootstrap_mode:
                print(f"\n🚨 EMERGENCY RECOVERY ACTIVATED!")
                if hasattr(self, 'gt_v') and np.linalg.norm(self.gt_v) > 0.1:
                    self.ekf.x.v = self.gt_v.copy() * 0.5
                    self.ekf.x.ba[1] = 0.0
                    self.ekf.P[13, 13] = 1.0
                    self.consecutive_vision_failures = 0
                    print(f"  🔄 Emergency reset complete")'''

        # ===== PERIODIC AUTO-TUNING =====
        if self.frame_count % 20 == 0:
            if len(self.innovation_history) >= 10:
                avg_innov = np.mean([i for _, i, _ in self.innovation_history[-10:]])
                if avg_innov > 100:
                    self.triangulation_config['pixel_sigma'] = min(100,
                        self.triangulation_config.get('pixel_sigma', 30) * 1.2)
                    print(f"  📊 Auto-tuning: Increased pixel sigma to {self.triangulation_config['pixel_sigma']:.1f}")

            if hasattr(self, 'gt_v') and np.linalg.norm(self.gt_v) > 0.1:
                v_ekf_norm = np.linalg.norm(self.ekf.x.v)
                v_gt_norm = np.linalg.norm(self.gt_v)
                actual_scale = v_ekf_norm / v_gt_norm if v_gt_norm > 0.1 else 1.0
                print(f"\n📊 AUTO-TUNING SUMMARY (Frame {self.frame_count}):")
                print(f"  Scale: EKF={self.ekf.x.s:.3f}, Actual={actual_scale:.3f}, Boost={self.scale_boost_factor:.1f}x")
                print(f"  Accel bias: {self.ekf.x.ba}")
                print(f"  Vision success rate: {self.vision_health['total_success']}/{(self.vision_health['total_success'] + self.vision_health['total_failures'])}")
                print(f"  AKIT Scales: gyro={self.current_gyro_scale:.2f}, acc={self.current_acc_scale:.2f}, bias={self.current_bias_scale:.2f}, R={self.current_r_scale:.2f}")

        # ===== UPDATE METRICS AND LOG =====
        if self.last_update_p is not None:
            dist_since_last = np.linalg.norm(self.ekf.x.p - self.last_update_p)
            vel_mag = np.linalg.norm(self.ekf.x.v)
            baseline_ratio = dist_since_last / (vel_mag + 1e-6) if vel_mag > 0.1 else 0.0
        else:
            dist_since_last = 0.0
            baseline_ratio = 0.0

        self.current_metrics.update({
            'baseline': float(dist_since_last), 
            'matches': int(matches),
            'status': status  # Update status for AKIT
        })

        innov_norm = getattr(self, 'last_innov_norm', 0.0)
        s_trace = getattr(self, 'last_s_trace', 0.0)
        s_cond = getattr(self, 'last_s_cond', 0.0)
        h_norm = getattr(self, 'last_h_norm', 0.0)
        num_points_used = getattr(self, 'last_num_points', 0)

        try:
            self.hdf5_logger.log(
                frame_id=self.frame_count,
                ts=t,
                ekf=self.ekf,
                imu_raw=self.latest_imu,
                accel_var=acc_var,
                is_zupt=is_zupt,
                num_matches=matches,
                num_triangulated=len(pts3d_filt) if pts3d_filt.size > 0 else 0,
                baseline=dist_since_last,
                vision_success=(status == "SUCCESS"),
                innov_norm=innov_norm,
                s_trace=s_trace,
                s_cond=s_cond,
                h_norm=h_norm,
                baseline_ratio=baseline_ratio,
                num_points_used=num_points_used,
                did_attempt_update=int(len(self.kps_buf) >= 2),
                did_apply_update=int(status == "SUCCESS")
            )
        except Exception as e:
            print(f"[WARN] Logging failure: {e}")

        self.log_all(t, status, acc_mag, acc_var, is_zupt)

        # ===== PUBLISH ODOMETRY =====
        self.publish_odom(msg.header.stamp)

        self.update_drift_data()
        self.analyze_drift()
        # Call periodically to maintain gravity alignment
        if self.frame_count % 20 == 0:
            self.periodic_gravity_check()
        # ===== STATUS PRINTING =====
        if self.frame_count % 5 == 0 or status == "SUCCESS":
            pos_error = np.linalg.norm(self.ekf.x.p - self.gt_p) if hasattr(self, 'gt_p') else 0.0
            print(f"\n[STATUS] Frame {self.frame_count}: {status}")
            print(f"  EKF Pos: ({self.ekf.x.p[0]:.1f}, {self.ekf.x.p[1]:.1f}, {self.ekf.x.p[2]:.1f})")
            if hasattr(self, 'gt_p'):
                print(f"  GT Pos:  ({self.gt_p[0]:.1f}, {self.gt_p[1]:.1f}, {self.gt_p[2]:.1f})")
                print(f"  Error: {pos_error:.2f}m")
            print(f"  Vel: {np.linalg.norm(self.ekf.x.v):.2f} m/s")
            print(f"  Accel bias: {self.ekf.x.ba}")
            print(f"  Scale: {self.ekf.x.s:.3f} (boost: {self.scale_boost_factor:.1f}x)")
            print(f"  Vision failures: {self.consecutive_vision_failures}")
            print(f"  Vision updates: {self.vision_update_count}")
            if self.bootstrap_mode:
                print(f"  🔧 BOOTSTRAP MODE: {self.vision_update_count}/20 updates")
            print(f"  VIO Health: {self.vio_health:.2f} (fallback={self.fallback_mode})")
            if self.use_akit:
                print(f"  AKIT: Active (gyro={self.current_gyro_scale:.2f}, acc={self.current_acc_scale:.2f}, bias={self.current_bias_scale:.2f}, R={self.current_r_scale:.2f})")

        self.last_img_t = t
        self.frame_count += 1
        
    def validate_update_geometry(self, delta_x, dt):
        """
        Validate that the proposed vision update is geometrically consistent
        with IMU-predicted motion - ADAPTIVE VERSION
        """
        # Extract proposed changes
        dtheta = delta_x[0:3]
        dv = delta_x[3:6]
        dp = delta_x[6:9]
       
        rot_deg = np.linalg.norm(dtheta) * 180/np.pi
        pos_change = np.linalg.norm(dp)
       
        # ===== QUALITY METRICS =====
        # Check if we have any valid points
        if not hasattr(self, 'last_valid_points') or self.last_valid_points < 6:
            print(f"  [QUALITY] Too few points ({self.last_valid_points}) - skipping")
            return False, "TOO_FEW_POINTS"
       
        # Check innovation magnitude (from last update)
        if hasattr(self, 'last_innov_norm') and self.last_innov_norm > 200:
            print(f"  [QUALITY] Innovation too high ({self.last_innov_norm:.1f}) - skipping")
            return False, "HIGH_INNOVATION"
       
        # ===== ADAPTIVE THRESHOLDS =====
        # During early convergence, be more permissive but not completely blind
        if self.ekf.update_count < 50:
            # First 50 updates: accept if not insane
            max_rot = 15.0  # 15° max rotation
            max_pos = 5.0   # 5m max position change
           
            if rot_deg > max_rot:
                return False, f"EXCESSIVE_ROTATION ({rot_deg:.1f}°)"
            if pos_change > max_pos:
                return False, f"EXCESSIVE_POSITION ({pos_change:.1f}m)"
               
            print(f"  [EARLY] Accepting moderate update: rot={rot_deg:.1f}°, pos={pos_change:.2f}m")
            return True, "VALID (early)"
           
        elif self.ekf.update_count < 200:
            # Next 150 updates: stricter
            max_rot = 8.0   # 8° max rotation
            max_pos = 3.0   # 3m max position change
           
            if rot_deg > max_rot:
                return False, f"EXCESSIVE_ROTATION ({rot_deg:.1f}°)"
            if pos_change > max_pos:
                return False, f"EXCESSIVE_POSITION ({pos_change:.1f}m)"
               
            print(f"  [MID] Accepting reasonable update: rot={rot_deg:.1f}°, pos={pos_change:.2f}m")
            return True, "VALID (mid)"
       
        else:
            # After convergence: tight thresholds
            max_rot = 3.0    # 3° max rotation
            max_pos = 1.0    # 1m max position change
           
            if rot_deg > max_rot:
                return False, f"EXCESSIVE_ROTATION ({rot_deg:.1f}°)"
            if pos_change > max_pos:
                return False, f"EXCESSIVE_POSITION ({pos_change:.1f}m)"
               
            # Also check direction consistency
            expected_motion = self.ekf.x.v * dt
            if np.linalg.norm(expected_motion) > 0.5:
                cos_angle = np.dot(expected_motion, dp) / (np.linalg.norm(expected_motion) * np.linalg.norm(dp) + 1e-6)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle_error = np.degrees(np.arccos(cos_angle))
               
                if angle_error > 45:
                    return False, f"WRONG_DIRECTION ({angle_error:.1f}°)"
           
            return True, "VALID"
   
    def get_motion_features(self, imu_data, zupt_enabled=False):
        """
        Improved motion features with better ZUPT detection
        """
        if not self.origin_initialized or len(imu_data) < 6:
            return 0.0, 0.0, 0
       
        omega = imu_data[0:3]
        acc = imu_data[3:6]
       
        # Remove bias
        acc_corrected = acc - self.ekf.x.ba
        omega_corrected = omega - self.ekf.x.bg
       
        # Get world acceleration
        acc_world = self.ekf.x.R @ acc_corrected
       
        # Gravity subtracted
        acc_motion = acc_world - self.ekf.g
        mag = np.linalg.norm(acc_motion)
       
        # High-frequency acceleration (for motion detection)
        if not hasattr(self, 'acc_lpf'):
            self.acc_lpf = mag
        else:
            # Simple low-pass filter
            self.acc_lpf = 0.9 * self.acc_lpf + 0.1 * mag
       
        acc_high_freq = abs(mag - self.acc_lpf)
       
        # Moving average for variance
        self.accel_buffer.append(mag)
        if len(self.accel_buffer) > self.window_size:
            self.accel_buffer.pop(0)
       
        var = np.var(self.accel_buffer) if len(self.accel_buffer) > 1 else 0.0
       
        # ===== IMPROVED ZUPT DETECTION =====
        # Only enable if explicitly allowed and conditions met
        is_zupt = 0
        if zupt_enabled:
            # Multiple conditions for ZUPT
            gyro_static = np.linalg.norm(omega_corrected) < 0.03
            acc_static = mag < 0.2
            vel_static = np.linalg.norm(self.ekf.x.v) < 0.05
            high_freq_static = acc_high_freq < 0.05
           
            if gyro_static and acc_static and vel_static and high_freq_static:
                is_zupt = 1
                # When ZUPT active, we can also reduce velocity
                if self.frame_count % 5 == 0:
                    # Gentle velocity damping
                    self.ekf.x.v *= 0.95
       
        return mag, var, is_zupt

    def imu_callback(self, msg):
        t_now = msg.header.stamp.to_sec()
    
        # Transform IMU to Body Frame
        omega_imu = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        acc_imu = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
    
        omega_body = self.R_imu_to_body @ omega_imu
        acc_body = self.R_imu_to_body @ acc_imu
        self.latest_imu = np.concatenate([omega_body, acc_body])
        
        if self.frame_count % 1000 == 0:
            print(f"Raw acc: {acc_imu}")
            print(f"Acc body: {acc_body}")
            print(f"Acc world: {self.ekf.x.R @ acc_body}")
            print(f"Velocity EKF: {self.ekf.x.v}")
            print(f"Velocity GT: {self.gt_v}")

        if self.frame_count % 10 == 0:
            self.check_gravity_consistency()

        if self.origin_initialized:
            # Handle first IMU message
            if self.last_imu_t is None:
                self.last_imu_t = t_now
                return

            dt = t_now - self.last_imu_t
        
            # Safety guard: clip dt to reasonable bounds instead of rejecting
            dt = np.clip(dt, 1e-3, 0.05)  # Never reject IMU, just bound it
        
            # Break large dt into smaller steps for better integration accuracy
            num_steps = max(1, int(dt / 0.01))
            small_dt = dt / num_steps
        
            for step in range(num_steps):
                # ---------- ZUPT (world frame) ----------
                acc_world = self.ekf.x.R @ acc_body
                is_zupt = 1.0 if np.linalg.norm(acc_world - np.array([0, 0, 9.793])) < 0.25 else 0.0

                # ---------- Build context ----------
                rpy = R_tool.from_matrix(self.ekf.x.R).as_euler('xyz')
                ctx = np.array([
                    acc_body[0], acc_body[1], acc_body[2],
                    rpy[0], rpy[1], rpy[2],
                    is_zupt
                ], dtype=np.float32)

                # ---------- Motion features ----------
                acc_mag = np.linalg.norm(acc_body)
                ang_vel_mag = np.linalg.norm(omega_body)
                speed = np.linalg.norm(self.ekf.x.v)

                mset = np.array([
                    acc_mag/15.0,
                    0.0,
                    ang_vel_mag/5.0,
                    speed/3.0,
                    0.0,
                    (ang_vel_mag**2)/10.0,
                    0.0,
                    0.0,
                    is_zupt
                ], dtype=np.float32)

                self.ctx_buf.append(ctx)
                self.mset_buf.append(mset)

                if len(self.ctx_buf) > self.window_size:
                    self.ctx_buf.pop(0)
                    self.mset_buf.pop(0)

                # ---------- Adaptive Q ----------
                Q_use = self.noise

                # ===== FIXED: Try AKIT first, fall back to old model =====
                if self.use_akit and len(self.ctx_buf) == self.window_size:
                    try:
                        # FIXED: Prepare IMU buffer with ALL context data
                        imu_buffer = []
                        for ctx_entry in self.ctx_buf:
                            imu_buffer.append([
                                ctx_entry[0], ctx_entry[1], ctx_entry[2],  # ax, ay, az
                                ctx_entry[3], ctx_entry[4], ctx_entry[5],  # roll, pitch, yaw (FIXED!)
                                ctx_entry[6]  # is_zupt
                            ])
                        
                        # Prepare metrics
                        metrics = self.prepare_akit_metrics(
                            status=self.current_metrics.get('status', 'IDLE'),
                            acc_mag=acc_mag,
                            acc_var=np.var([c[0:3] for c in self.ctx_buf]) if len(self.ctx_buf) > 1 else 0,
                            is_zupt=is_zupt
                        )
                        
                        # Get predictions
                        q_scales, r_scale = self.akit_predictor.predict_from_buffer(imu_buffer, metrics)
                        
                        # FIXED: Use ALL three scales
                        self.current_gyro_scale = float(np.clip(q_scales[0], 0.1, 10.0))
                        self.current_acc_scale = float(np.clip(q_scales[1], 0.1, 10.0))
                        self.current_bias_scale = float(np.clip(q_scales[2], 0.1, 10.0))
                        self.current_r_scale = float(np.clip(r_scale, 0.1, 10.0))
                        
                        # Create adaptive noise parameters with ALL scales
                        Q_use = NoiseParams(
                            gyro_noise=self.noise.gyro_noise * self.current_gyro_scale,
                            accel_noise=self.noise.accel_noise * self.current_acc_scale,
                            gyro_bias_rw=self.noise.gyro_bias_rw * self.current_bias_scale,  # FIXED: Use bias scale
                            accel_bias_rw=self.noise.accel_bias_rw * self.current_bias_scale   # FIXED: Use bias scale
                        )
                        
                        if self.frame_count % 20 == 0:
                            print(f"[AKIT] gyro_scale={self.current_gyro_scale:.2f} acc_scale={self.current_acc_scale:.2f} "
                                  f"bias_scale={self.current_bias_scale:.2f} R_scale={self.current_r_scale:.2f}")
                            
                    except Exception as e:
                        # Fall back to old model if AKIT fails
                        rospy.logwarn_throttle(10.0, f"AKIT prediction failed: {e}, using fallback")
                        
                        if self.use_ml_q and len(self.ctx_buf) == self.window_size:
                            ctx_t = torch.tensor([self.ctx_buf]).float().to(self.device)
                            mset_t = torch.tensor([self.mset_buf]).float().to(self.device)
                            
                            with torch.no_grad():
                                scale = self.q_model(ctx_t, mset_t)[0].cpu().numpy()
                            
                            gyro_scale = float(np.clip(scale[0], 0.5, 8.0))
                            acc_scale = float(np.clip(scale[1], 0.5, 8.0))
                            
                            Q_use = NoiseParams(
                                gyro_noise=self.noise.gyro_noise * gyro_scale,
                                accel_noise=self.noise.accel_noise * acc_scale,
                                gyro_bias_rw=self.noise.gyro_bias_rw,
                                accel_bias_rw=self.noise.accel_bias_rw
                            )
                            
                            if self.frame_count % 20 == 0:
                                print(f"[ML-Q] gyro_scale={gyro_scale:.2f} acc_scale={acc_scale:.2f}")
                
                elif self.use_ml_q and len(self.ctx_buf) == self.window_size:
                    # Use old model directly
                    ctx_t = torch.tensor([self.ctx_buf]).float().to(self.device)
                    mset_t = torch.tensor([self.mset_buf]).float().to(self.device)
                    
                    with torch.no_grad():
                        scale = self.q_model(ctx_t, mset_t)[0].cpu().numpy()
                    
                    gyro_scale = float(np.clip(scale[0], 0.5, 8.0))
                    acc_scale = float(np.clip(scale[1], 0.5, 8.0))
                    
                    Q_use = NoiseParams(
                        gyro_noise=self.noise.gyro_noise * gyro_scale,
                        accel_noise=self.noise.accel_noise * acc_scale,
                        gyro_bias_rw=self.noise.gyro_bias_rw,
                        accel_bias_rw=self.noise.accel_bias_rw
                    )
                    
                    if self.frame_count % 20 == 0:
                        print(f"[ML-Q] gyro_scale={gyro_scale:.2f} acc_scale={acc_scale:.2f}")
                # ===== END FIXED =====

                # Perform prediction step
                self.ekf.predict(
                    omega=omega_body,
                    acc=acc_body,
                    dt=small_dt,
                    Q=Q_use,
                    debug=False
                )
                self.ekf.apply_gravity_alignment(small_dt)
                # ===== DRIFT ERROR PRINTING =====
                if self.ekf.predict_count % 1000 == 0:
                    if hasattr(self, 'gt_p') and np.linalg.norm(self.gt_p) > 0:
                        pos_error = np.linalg.norm(self.ekf.x.p - self.gt_p)
                        vel_error = np.linalg.norm(self.ekf.x.v - self.gt_v)
                        
                        # Get orientation error
                        R_error = self.gt_R.T @ self.ekf.x.R
                        rot_error_deg = np.linalg.norm(R_tool.from_matrix(R_error).as_rotvec()) * 180/np.pi
                        
                        print(f"\n📊 DRIFT REPORT (Prediction {self.ekf.predict_count}):")
                        print(f"  Position error: {pos_error:.3f} m")
                        print(f"  Velocity error: {vel_error:.3f} m/s")
                        print(f"  Orientation error: {rot_error_deg:.3f}°")
                        print(f"  Scale factor: {self.ekf.x.s:.3f}")
                        print(f"  Accel bias: {self.ekf.x.ba}")
                        
                        # Z-axis specific
                        z_vel_error = self.ekf.x.v[2] - self.gt_v[2]
                        print(f"  Z-velocity error: {z_vel_error:.3f} m/s (EKF: {self.ekf.x.v[2]:.3f}, GT: {self.gt_v[2]:.3f})")
                        
                        # Track drift rate over last 1000 predictions
                        if not hasattr(self, 'last_drift_pos'):
                            self.last_drift_pos = self.ekf.x.p.copy()
                            self.last_drift_count = self.ekf.predict_count
                        else:
                            dt_drift = (self.ekf.predict_count - self.last_drift_count) * small_dt * num_steps
                            if dt_drift > 0:
                                drift_rate = pos_error / dt_drift
                                print(f"  Drift rate: {drift_rate:.4f} m/s")
                            self.last_drift_pos = self.ekf.x.p.copy()
                            self.last_drift_count = self.ekf.predict_count
        
        self.last_imu_t = t_now
        self.imu_history.append((t_now, self.latest_imu))
        if len(self.imu_history) > 100:
            self.imu_history.pop(0)
    
        self.frame_count += 1

    def imu_callback_debug(self, msg):
        """Debug IMU callback to check if data is arriving"""
        if not hasattr(self, 'imu_debug_count'):
            self.imu_debug_count = 0
        self.imu_debug_count += 1
       
        if self.imu_debug_count <= 5:
            print(f"[IMU-DEBUG {self.imu_debug_count}] Received IMU message")
            print(f"  Angular velocity: [{msg.angular_velocity.x:.3f}, "
                f"{msg.angular_velocity.y:.3f}, {msg.angular_velocity.z:.3f}]")
            print(f"  Linear acceleration: [{msg.linear_acceleration.x:.3f}, "
                f"{msg.linear_acceleration.y:.3f}, {msg.linear_acceleration.z:.3f}]")
            print(f"  Timestamp: {msg.header.stamp.to_sec()}")

    def image_callback_debug(self, msg):
        """Debug image callback to check if images are arriving"""
        if not hasattr(self, 'image_debug_count'):
            self.image_debug_count = 0
        self.image_debug_count += 1
       
        if self.image_debug_count <= 3:
            print(f"[IMAGE-DEBUG {self.image_debug_count}] Received image")
            print(f"  Width: {msg.width}, Height: {msg.height}")
            print(f"  Encoding: {msg.encoding}")
            print(f"  Timestamp: {msg.header.stamp.to_sec()}")
           
            # Try to convert to check for errors
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                print(f"  Successfully converted to OpenCV image: {cv_image.shape}")
            except Exception as e:
                print(f"  ERROR converting image: {e}")
   
    def test_projection_with_new_extrinsics(self):
        """Test projection with corrected extrinsics"""
        print(f"\n=== PROJECTION TEST WITH CORRECTED EXTRINSICS ===")
       
        # Test with realistic depths
        R_wb = np.eye(3)
        p_wb = np.array([0, 0, -50])  # 50m underwater
       
        P = self.get_proj(R_wb, p_wb)
       
        # Point 10m in front of body
        point_body = np.array([10, 0, 0])
        point_world = R_wb @ point_body + p_wb
       
        point_hom = np.append(point_world, 1.0)
        proj = P @ point_hom
       
        if abs(proj[2]) > 1e-6:
            pixel = proj[:2] / proj[2]
            print(f"Point 10m in front of body:")
            print(f"  World coordinates: {point_world}")
            print(f"  Projects to pixel: [{pixel[0]:.1f}, {pixel[1]:.1f}]")
            print(f"  Should be near image center: [384.5, 246.5]")
       
        # Test 2: Point on seafloor (Z = -95m)
        seafloor_point = np.array([20, 5, -95])  # 20m forward, 5m right, on seafloor
        seafloor_hom = np.append(seafloor_point, 1.0)
        seafloor_proj = P @ seafloor_hom
       
        if abs(seafloor_proj[2]) > 1e-6:
            seafloor_pixel = seafloor_proj[:2] / seafloor_proj[2]
            print(f"\nPoint on seafloor (Z=-95m):")
            print(f"  World coordinates: {seafloor_point}")
            print(f"  Projects to pixel: [{seafloor_pixel[0]:.1f}, {seafloor_pixel[1]:.1f}]")
       
        print(f"===\n")

    def gt_callback(self, msg: ModelStates):
        """Process ground truth data - FOR EVALUATION ONLY, no resets"""
        try:
            idx = msg.name.index("lauv")
        except ValueError:
            return
       
        pose = msg.pose[idx]
        twist = msg.twist[idx]
       
        # Store ground truth for evaluation only
        self.gt_p = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        self.gt_R = R_tool.from_quat(q).as_matrix()
        self.gt_v = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
       
        # Pass to EKF for recovery monitoring (not for correction!)
        self.ekf.update_ground_truth(self.gt_p, self.gt_R, self.gt_v)
       
        # Initialize EKF only once at start
        if not self.origin_initialized:
            print(f"\n=== INITIALIZING EKF WITH GROUND TRUTH ===")
            print(f"  ⚠️ This is for evaluation only - never in production!")
            self.ekf.x.p = self.gt_p.copy()
            self.ekf.x.R = self.gt_R.copy()
            self.ekf.x.v = self.gt_v.copy()
            self.origin_initialized = True
            print(f"EKF initialized at: [{self.gt_p[0]:.2f}, {self.gt_p[1]:.2f}, {self.gt_p[2]:.2f}]")
       
    def update_drift_data(self):
        """Store current state for drift analysis"""
        # Use rospy time or time.time() for numeric timestamps
        current_time = rospy.Time.now().to_sec()  # or time.time()
       
        self.drift_analysis['positions'].append(self.ekf.x.p.copy())
        self.drift_analysis['gt_positions'].append(self.gt_p.copy())
        self.drift_analysis['timestamps'].append(current_time)  # Store as float
       
        # Calculate current error
        error = np.linalg.norm(self.ekf.x.p - self.gt_p)
        self.drift_analysis['errors'].append(error)
       
        # Keep only recent history
        max_history = 1000
        for key in ['positions', 'gt_positions', 'timestamps', 'errors']:
            if len(self.drift_analysis[key]) > max_history:
                self.drift_analysis[key] = self.drift_analysis[key][-max_history:]

    def analyze_drift(self):
        """Analyze drift patterns over time - CORRECTED version"""
        if len(self.drift_analysis['positions']) < 10:
            return
        
        # Get ALL data, not just last 100 frames
        all_positions = np.array(self.drift_analysis['positions'])
        all_gt = np.array(self.drift_analysis['gt_positions'])
        all_times = np.array(self.drift_analysis['timestamps'], dtype=np.float64)
        
        # Use recent data for rate calculations, but full history for cumulative distance
        if len(all_positions) > 100:
            recent_pos = all_positions[-100:]
            recent_gt = all_gt[-100:]
            recent_times = all_times[-100:]
        else:
            recent_pos = all_positions
            recent_gt = all_gt
            recent_times = all_times
        
        # Calculate current error
        current_error = np.linalg.norm(all_positions[-1] - all_gt[-1])
        
        # ===== FIXED: Calculate TOTAL distance traveled (not just last 100 frames) =====
        # Use all available positions for cumulative distance
        if len(all_positions) > 1:
            position_diffs = np.diff(all_positions, axis=0)
            total_distance = np.sum(np.linalg.norm(position_diffs, axis=1))
        else:
            total_distance = 0.0
        
        # Calculate recent drift rate
        if len(recent_times) > 10 and recent_times[-1] > recent_times[0]:
            time_span = recent_times[-1] - recent_times[0]
            
            # Calculate error change over recent window
            recent_errors = np.linalg.norm(recent_pos - recent_gt, axis=1)
            error_change = recent_errors[-1] - recent_errors[0]
            drift_rate = error_change / time_span if time_span > 0 else 0
            
            # Calculate drift as percentage of TOTAL distance traveled
            drift_percentage = (current_error / total_distance * 100) if total_distance > 0 else 0
            
            print(f"\n📊 DRIFT ANALYSIS:")
            print(f"  Current error: {current_error:.3f} m")
            print(f"  Error range: {recent_errors.min():.3f} - {recent_errors.max():.3f} m")
            print(f"  Drift rate: {drift_rate:.4f} m/s (over last {time_span:.1f}s)")
            print(f"  Total distance traveled: {total_distance:.2f} m")
            print(f"  Drift % of distance: {drift_percentage:.2f}%")
            
            # Store for trend analysis
            self.drift_analysis['drift_rate'] = drift_rate
            self.drift_analysis['total_distance'] = total_distance
            
            # Alert based on drift rate (this is correct now)
            if drift_rate > 0.1:  # More than 10cm/s drift
                print(f"  ⚠️ HIGH DRIFT RATE: {drift_rate:.4f} m/s")
            elif drift_rate < 0.01:  # Less than 1cm/s drift
                print(f"  ✅ LOW DRIFT RATE: {drift_rate:.4f} m/s")
            
            # Also check if drift percentage is concerning
            if drift_percentage > 10.0:  # More than 10% drift over total distance
                print(f"  ⚠️ HIGH DRIFT PERCENTAGE: {drift_percentage:.2f}%")
            elif drift_percentage < 1.0:
                print(f"  ✅ LOW DRIFT PERCENTAGE: {drift_percentage:.2f}%")
                
    def log_all(self, t, status, acc_mag, acc_var, is_zupt):
        """Log all data to file"""
        # Get EKF orientation
        ev = R_tool.from_matrix(self.ekf.x.R).as_euler('xyz', degrees=True)
        eg = R_tool.from_matrix(self.gt_R).as_euler('xyz', degrees=True)
       
        # Get acceleration in world frame
        if len(self.latest_imu) == 6:
            av = self.ekf.x.R @ (self.latest_imu[3:6] - self.ekf.x.ba)
        else:
            av = np.zeros(3)
       
        # Get metrics from stored values (with defaults)
        innov_norm = getattr(self, 'last_innov_norm', 0.0)
        s_trace = getattr(self, 'last_s_trace', 0.0)
        s_cond = getattr(self, 'last_s_cond', 0.0)
        h_norm = getattr(self, 'last_h_norm', 0.0)
       
        # Prepare data - include ALL metrics including AKIT scales
        base_data = [
            t, status,
            self.ekf.x.p[0], self.ekf.x.p[1], self.ekf.x.p[2],
            self.gt_p[0], self.gt_p[1], self.gt_p[2],
            self.ekf.x.v[0], self.ekf.x.v[1], self.ekf.x.v[2],
            self.gt_v[0], self.gt_v[1], self.gt_v[2],
            av[0], av[1], av[2],
            ev[0], ev[1], ev[2],
            eg[0], eg[1], eg[2]
        ]
       
        ml_data = [
            innov_norm,           # innov_norm (was 'innov')
            s_trace,              # s_trace (was 'trace')
            s_cond,               # s_cond (was 'cond')
            h_norm,               # h_norm
            self.current_metrics.get('matches', 0),
            self.current_metrics.get('baseline', 0),
            acc_mag,
            acc_var,
            is_zupt,
            self.current_gyro_scale,  # gyro_scale from AKIT
            self.current_acc_scale,     # acc_scale from AKIT
            self.current_bias_scale,    # bias_scale from AKIT
            self.current_r_scale         # r_scale from AKIT
        ]
       
        # Write to log file
        line = ",".join(map(str, base_data + ml_data)) + "\n"
        self.debug_log.write(line)
        self.debug_log.flush()

    def publish_odom(self, stamp):
        """Publish odometry message"""
        o = Odometry()
        o.header.stamp = stamp
        o.header.frame_id = "world"
       
        # Position
        o.pose.pose.position.x = self.ekf.x.p[0]
        o.pose.pose.position.y = self.ekf.x.p[1]
        o.pose.pose.position.z = self.ekf.x.p[2]
       
        # Orientation
        q = R_tool.from_matrix(self.ekf.x.R).as_quat()
        o.pose.pose.orientation.x = q[0]
        o.pose.pose.orientation.y = q[1]
        o.pose.pose.orientation.z = q[2]
        o.pose.pose.orientation.w = q[3]
       
        # Velocity
        o.twist.twist.linear.x = self.ekf.x.v[0]
        o.twist.twist.linear.y = self.ekf.x.v[1]
        o.twist.twist.linear.z = self.ekf.x.v[2]
       
        # Publish
        self.odom_pub.publish(o)

    def shutdown_hook(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'hdf5_logger'):
            self.hdf5_logger.close()
        if hasattr(self, 'debug_log'):
            self.debug_log.close()
        print("[SHUTDOWN] Cleanup complete")

    def check_gravity_consistency(self):
        """Check if gravity vector in body frame is reasonable"""
        # Expected gravity direction in body frame (should be near [0,0,9.793])
        g_body = self.ekf.x.R.T @ np.array([0, 0, 9.793])
        
        # The Z component should dominate
        z_ratio = abs(g_body[2]) / np.linalg.norm(g_body)
        
        if z_ratio < 0.9:  # Less than 90% of gravity in Z
            print(f"  ⚠️ Gravity inconsistency: g_body={g_body}")
            print(f"    Only {z_ratio*100:.1f}% of gravity in Z-axis")
            
            # Don't reset, just warn and maybe boost orientation covariance
            self.ekf.P[0:3, 0:3] *= 2.0
            return False
        return True

    # After setting up subscribers (around line where you create self.img_sub)
    def wait_for_camera_ready(self):
        """Wait for camera publisher to be ready"""
        rospy.loginfo("Waiting for camera publisher...")
        start_time = rospy.Time.now()
        timeout = start_time + rospy.Duration(10.0)
        
        camera_topic = "/lauv/lauv/camerafront/camera_image"
        
        # Wait for camera publisher to appear
        while rospy.Time.now() < timeout:
            # Get list of all published topics
            topics = rospy.get_published_topics()
            topic_names = [t[0] for t in topics]
            
            if camera_topic in topic_names:
                rospy.loginfo(f"Camera topic '{camera_topic}' found!")
                break
                
            elapsed = (rospy.Time.now() - start_time).to_sec()
            if elapsed > 1.0 and elapsed % 2.0 < 0.1:
                rospy.loginfo(f"Still waiting for camera... ({elapsed:.1f}s)")
            rospy.sleep(0.1)
        
        if rospy.Time.now() >= timeout:
            rospy.logwarn(f"Camera topic '{camera_topic}' didn't appear within timeout")
        
        # Also wait for first image to actually arrive
        rospy.loginfo("Waiting for first image...")
        img_timeout = rospy.Time.now() + rospy.Duration(5.0)
        while self.last_img_t is None and rospy.Time.now() < img_timeout:
            rospy.sleep(0.1)
        
        if self.last_img_t is not None:
            rospy.loginfo(f"First image received at t={self.last_img_t:.3f}s!")
        else:
            rospy.logwarn("No image received within timeout")

if __name__ == '__main__':
    rospy.init_node('vio_node')
    node = GazeboVIONode()
    rospy.spin()
