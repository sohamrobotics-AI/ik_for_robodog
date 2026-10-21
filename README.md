Quadruped Leg Kinematics and Trot Gait
A self-contained Python simulation of a 3-DOF quadruped leg and a four-leg trot gait. The script implements inverse and forward kinematics, generates a touchdown-matched Bézier swing trajectory, audits joint-angle and joint-speed limits, and exports diagnostic plots plus animated GIFs.
> **Scope:** This is a kinematic gait-analysis tool. It does not model rigid-body dynamics, contact forces, actuator torque, or a physical robot controller.
Features
3-DOF leg inverse kinematics (IK): Computes hip abduction, hip pitch, and knee angles for a target foot position.
Forward kinematics (FK): Reconstructs knee and foot positions from joint angles.
Four-leg trot coordination: Runs diagonal pairs in phase (`FL` + `RR`, then `FR` + `RL`).
Foot trajectory generation: Uses a straight stance segment and a Bézier swing curve designed to match stance velocity at lift-off and touchdown.
Reachability handling: Raises `IKError` for unreachable targets by default, or returns `NaN` values when requested.
Limit audit: Flags joint angles beyond ±90° and estimated joint speeds above 6 rad/s.
Visual diagnostics: Saves joint-angle plots, foot trajectory plots, feasibility envelope maps, and trot animations.
IK/FK sanity check: Tests round-trip accuracy on random target points for both left and right legs.
Requirements
Python 3.9 or newer (the script uses modern type annotations and `dataclasses`)
NumPy
Matplotlib
Pillow (used by Matplotlib's GIF writer)
Install dependencies:
```bash
python -m pip install numpy matplotlib pillow
```
Run
Place `quadruped_leg.py` in a directory where you want the generated outputs, then run:
```bash
python quadruped_leg.py
```
The script writes its output files to the same directory as `quadruped_leg.py` and prints:
IK-to-FK round-trip errors for left and right legs.
Joint-angle and joint-speed limit reports for nominal and aggressive gait settings.
The output directory for the generated figures and GIFs.
Generated files
File	Description
`joints_nominal.png`	Joint angles over one nominal gait cycle, with stance regions and limit violations marked.
`joints_aggressive.png`	Joint-angle diagnostics for a faster, longer-step gait.
`trot_nominal.gif`	Animation of the nominal trot. Legs turn red when a configured limit is violated.
`trot_aggressive.gif`	Animation of the aggressive trot.
`foot_trajectory.png`	Foot path and foot-velocity plots for the nominal gait.
`envelope_maps.png`	Feasibility maps showing the effect of stand/step height and cycle time/step length on joint limits.
The script overwrites files with these names each time it runs.
Leg coordinate frame and joint conventions
Each leg uses a local coordinate frame with its origin at the hip-abduction axis:
`+x`: forward
`+y`: left
`+z`: up
Negative `z`: below the hip
The three joint angles are returned in this order:
```text
[q1, q2, q3] = [hip abduction, hip pitch, knee]
```
Hip abduction (`q1`): Rotation about `+x`. Positive rotation swings a hanging foot toward `+y`.
Hip pitch (`q2`): Positive angles move the foot forward.
Knee (`q3`): Positive angles use the backward-bending knee convention described in the source comments.
The leg plane is offset from the abduction axis by `l0`. The `side` argument identifies the leg side: `+1` for left and `-1` for right.
Default geometry and gait
All distances are in metres, times in seconds, angles in radians internally, and angular velocities in rad/s.
Geometry
Parameter	Default	Meaning
`L1`	0.22 m	Thigh length
`L2`	0.22 m	Shank length
`l0`	0.06 m	Lateral offset from the abduction axis to the leg plane
Gait parameters
Parameter	Default	Meaning
`step_length`	0.12 m	Peak-to-peak foot travel in the body frame
`step_height`	0.05 m	Swing clearance above the stance line
`cycle_time`	0.60 s	Duration of one complete gait cycle
`duty`	0.50	Fraction of the cycle spent in stance
`stand_height`	0.37 m	Hip height above the ground
`tuck`	0.012 m	Lateral foot tuck at the swing apex
The script evaluates two configurations:
Nominal: the default `GaitParams`.
Aggressive: `step_length=0.20 m`, `step_height=0.08 m`, and `cycle_time=0.35 s`.
The nominal body-speed estimate is calculated as:
```text
body speed = step_length / (duty * cycle_time)
```
This is a gait-derived kinematic estimate, not a dynamically validated speed for a physical robot.
Main functions and classes
Symbol	Purpose
`LegGeometry`	Stores the thigh, shank, and lateral-offset dimensions.
`GaitParams`	Stores gait timing and foot-trajectory parameters.
`leg_ik(foot, side=+1, geom=GEOM, on_unreachable="raise")`	Computes joint angles for one or multiple foot targets. Accepts arrays shaped `(..., 3)`.
`leg_points(q, side=+1, geom=GEOM)`	Returns the knee and foot positions from joint angles.
`leg_fk(q, side=+1, geom=GEOM)`	Returns the foot position from joint angles.
`bezier(ctrl, s)`	Evaluates a Bézier curve from control points.
`foot_xz(phase, p)`	Generates the sagittal-plane foot trajectory for gait phase values.
`foot_path_leg_frame(phase, side, p, geom=GEOM)`	Adds lateral tuck and stand height to produce 3D foot targets.
`run_trot(p=GaitParams(), geom=GEOM, n=720)`	Simulates one gait cycle and returns joint positions, speed estimates, stance states, and limit flags.
`report(result, title="")`	Formats a per-leg, per-joint limit report.
`plot_joints(...)`, `plot_trajectory(...)`, `plot_envelopes(...)`	Generate diagnostic plots.
`make_gif(...)`	Creates an animated trot visualization.
Example: solve one leg target
```python
import numpy as np
from quadruped_leg import leg_ik, leg_fk, IKError

# Target in the leg frame: 5 cm forward, 6 cm to the left,
# and 35 cm below the hip.
target = np.array([0.05, 0.06, -0.35])

try:
    q = leg_ik(target, side=+1)
    reconstructed = leg_fk(q, side=+1)
    print("Joint angles [rad]:", q)
    print("Joint angles [deg]:", np.rad2deg(q))
    print("Reconstructed foot [m]:", reconstructed)
except IKError as exc:
    print("Target is outside the leg workspace:", exc)
```
For a right leg, use `side=-1`. Ensure the target coordinates are expressed in that leg's local frame.
Example: run a gait analysis
```python
from quadruped_leg import GaitParams, run_trot, report

params = GaitParams()
result = run_trot(params, n=720)

print(report(result, "My gait"))
print("Limit audit passed:", result.ok)
print("Estimated body speed [m/s]:", result.body_speed)
```
`result.ok` is `True` only when the sampled gait has no flagged joint-angle or joint-speed violations. Unreachable IK targets are also flagged. The audit checks sampled kinematic values against the limits configured in the script; it is not a hardware safety certification.
How the gait works
Stance: The foot moves along a straight line at constant trajectory speed.
Swing: A Bézier curve lifts the foot, carries it forward, and returns it to the stance line.
Touchdown matching: The swing control points are selected to match the stance trajectory's horizontal velocity at the phase boundaries, with zero vertical slope at lift-off and touchdown.
Trot phasing: Front-left and rear-right legs share a phase; front-right and rear-left legs are offset by half a cycle.
IK and audit: Each sampled foot target is converted to joint angles. Joint velocities are estimated using periodic central differences, then compared with the configured angle and speed limits.
Important limitations
The simulation is kinematic; it does not establish dynamic stability or realistic ground contact.
The joint-angle limit is configured as ±90° and the speed limit as 6 rad/s for this audit. Actual actuator limits may differ.
Joint speeds are estimated numerically from sampled joint angles. Results can depend on the sample count (`n`).
The code assumes the stated coordinate and sign conventions. A real robot may require frame transforms, offsets, sign changes, calibration, and joint-limit updates.
The animation is a visualization, not a physics simulation.
The feasibility maps are sampled approximations of the parameter space, not formal proofs of feasibility.
Troubleshooting
`ModuleNotFoundError: No module named 'numpy'` (or `matplotlib` / `PIL`)
Install the dependencies using the command in Requirements.
A target raises `IKError`
Check that the target is expressed in the correct leg frame, that `side` matches the leg, and that the point lies inside the reachable workspace. For batch analysis, `on_unreachable="nan"` marks unreachable targets with `NaN` values instead of raising an exception.
The report shows limit violations
Review the flagged leg, joint, and time interval. Consider adjusting step length, step height, stand height, or cycle time, then rerun the audit. Do not treat a passing audit as proof that a physical robot will operate safely.
License
No license is specified in this repository/file. Add a license before redistributing or reusing the code if you intend to make the project's reuse terms explicit.
