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
Introduces a learning-based adaptive module (Set Transformer)Dynamically
adjusts process noise

Ensures stable localization under dynamic underwater conditions

-------------------------------------------------------------------------

EXPERIMENTAL SETUP

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
motion dynamics
Measurement noise covariance (Rk) based
on visual quality
Instead of fixed noise values, the system
uses learned scaling factors to adjust
uncertainty in real time

Qk=scaled noise based on (γgyro,γaccel,γbias )
Rk=γr⋅Rnominal
-------------------------------------------------------------------------------
Conclusion

Proposed A-KIT VIO, an adaptive visual–inertial odometry framework for GPS-denied
underwater environments
Integrated EKF with transformer-based adaptive noise scaling to handle dynamic
motion conditions
Overcomes limitations of fixed-noise VIO by adapting process noise online
Achieves stable state estimation and bounded drift even with sparse visual updates
Enables real-time performance with minimal computational overhead

----------------------------------------------------------------------------
Future Work :
testing with 30fps+ camera pipeline
closed loop innovation calculation
Validation using real-world underwater datasets and field experiments
