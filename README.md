A-KIT BASED VISUAL-INERTIAL ODOMETRY FRAMEWORK FOR AUTONOMOUS UNDERWATER VEHICLE POSITIONING

Achieving accurate positioning of Autonomous Underwater Vehicles
(AUVs) in GPS-denied environments using Visual-Inertial Odometry
remains challenging due to dynamic underwater conditions and
sensor uncertainties.
---------------------------------------------------------------------
We propose A-KIT VIO: an adaptive visual-inertial odometry framework
Combines:
Camera (visual features)
IMU (motion estimation)

A-KIT VIO is a closed-loop state estimation framework for underwater positioning
Uses Extended Kalman Filter (EKF) for sensor fusion
Fuses:

High-rate IMU data (100–200 Hz)
Monocular camera data (1–10 Hz)

Core Innovation: Introduces a learning-based adaptive module (LSTM + Set Transformer) that dynamically adjusts process and measurement noise.

Ensures stable localization under dynamic underwater conditions.
<img width="1397" height="585" alt="image" src="https://github.com/user-attachments/assets/e8e67013-3a1f-486a-9d3e-72b00fff9b6d" />

-------------------------------------------------------------------------

EXPERIMENTAL SETUP
<img width="1312" height="697" alt="image" src="https://github.com/user-attachments/assets/1c221835-7d13-4509-8315-d166c76a73eb" />

Simulation Environment :
Evaluated in a Gazebo(version-11)-based underwater simulation with ROS(neotic)
AUV equipped with:
Monocular camera (1– 10Hz)
IMU (100–200 Hz)

Ground-truth pose used for evaluation

System Implementation : 
Real-time ROS-based framework
IMU → state prediction
Camera → feature extraction and updates
---------------------------------------------------------------------------
Motivation: Covariance Mismatch
EKF performance depends on accurate
process and measurement noise
Fixed covariance assumptions fail in
underwater environments due to:

Non-stationary motion (currents,
disturbances)
Varying visual quality (turbidity,
illumination)

This leads to covariance mismatch,
causing:
Overconfidence → drift
Instability → noisy estimates

----------------------------------------------------------------------------
Proposed Innovation
Introduce a learning-based adaptive noise
model
Dynamically updates both:
Process noise covariance (Qk) based on
motion dynamics.
Measurement noise covariance (Rk) based
on visual quality.
Instead of fixed noise values, the system
uses learned scaling factors to adjust
uncertainty in real time.

Qk=scaled noise based on (γgyro,γaccel,γbias )

Rk=yr.Rnominal

-------------------------------------------------------------------------------
Conclusion

Proposed A-KIT VIO, an adaptive visual–inertial odometry framework for GPS-denied
underwater environments.

Integrated EKF with a hybrid LSTM-Transformer adaptive noise scaling to handle complex motion and environmental conditions.

Overcomes limitations of fixed-noise VIO by adapting process noise online.

Achieves stable state estimation and bounded drift even with sparse visual updates.

Enables real-time performance with minimal computational overhead.

----------------------------------------------------------------------------
Future Work :
Testing with 30fps+ camera pipeline.

Closed loop innovation calculation.

Validation using real-world underwater datasets and field experiments.

-----------------------------------------------------------------------------
## Research Note & Citation
This repository contains the official implementation of the A-KIT VIO framework.

Please Note: The current version of this code includes advanced modules—specifically Induced Set Attention Blocks (ISAB) and Pooling by Multihead Attention (PMA)—which are part of a follow-up publication currently under submission.

If you use this code, the hybrid architecture, or the A-KIT framework in your research, please cite our foundational work:

Gauri P Nair, Vinaya V, Dona Sebastian, and Kavitha K V, "A-KIT Based Visual-Inertial Odometry Framework for Autonomous Underwater Vehicle Positioning," 11th International Conference on Information and Communication Technology for Intelligent Systems (ICTIS 2026), Springer Nature Lecture Notes in Networks and Systems (LNNS), 2026.
