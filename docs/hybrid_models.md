# Hybrid Models Documentation

This document defines the two implemented physics-guided models and the
notation used in `src/models/hybrid_linear_beam.py` and
`src/models/hybrid_nonlinear_cam.py`.

## Notation

- `theta`: angular position (rad)
- `theta_dot`: angular velocity (rad/s)
- `theta_ddot`: angular acceleration (rad/s^2)
- `V`: input voltage (V)
- `i`: motor current (A)
- `u`: input voltage signal used in code (same role as `V`)
- `J`: inertia (kg.m^2)
- `R`: viscous damping coefficient (N.m.s/rad)
- `K`: stiffness (N.m/rad)
- `delta`: preload-equivalent displacement
- `tau` or `k_t`: torque constant (N.m/A or N.m/V-equivalent depending on model)

Cam-bar geometry constants:

- `R`: cam radius
- `r`: roller radius
- `e`: cam eccentricity
- `L`: lever geometry parameter
- `I`: bar inertia
- `k`: spring constant

Motor electrical constants:

- `R_M`: motor winding resistance
- `L_M`: motor inductance
- `k_b`: back-EMF constant

## Model 1: Linear Beam Hybrid

Equation:

$$
J\ddot{\theta} + R\dot{\theta} + K(\theta + \delta) = \tau V
$$

Rearranged for identification:

$$
\ddot{\theta} = -\frac{R}{J}\dot{\theta} - \frac{K}{J}\theta + \frac{\tau}{J}V - \frac{K}{J}\delta
$$

Implementation details:

- `theta_dot` and `theta_ddot` are estimated from sampled `y` using finite differences.
- A linear regression estimates:
  - `a1 = R/J`
  - `a0 = K/J`
  - `b0 = tau/J`
  - `bias = -(K/J)delta`
- Physical parameters are recovered as:
  - `J = tau / b0`
  - `R = a1 * J`
  - `K = a0 * J`
  - `delta = -bias / a0`

Prediction:

- `OSA`: uses measured `theta(k)` and finite-difference `theta_dot(k)` to predict
  `theta(k+1)`.
- `FR`: simulates recursively from two initial output samples.

## Model 2: Nonlinear Cam-Bar-Motor Hybrid

### Geometry

$$
S(\theta)=\sqrt{(R+r)^2-e^2\sin^2\theta}
$$
$$
y(\theta)=S(\theta)-e\cos\theta-(R+r-e)
$$
$$
\sin\phi(\theta)=\frac{e\sin\theta}{R+r}
$$
$$
\cos\phi(\theta)=\sqrt{1-\sin^2\phi(\theta)}
$$

Define:

$$
A(\theta)=\frac{dy}{d\theta}
=-\frac{e^2\sin\theta\cos\theta}{S(\theta)}+e\sin\theta
$$
$$
B(\theta)=\frac{d^2y}{d\theta^2}
=e\cos\theta
-\frac{e^2(\cos^2\theta-\sin^2\theta)}{S(\theta)}
-\frac{e^4\sin^2\theta\cos^2\theta}{S(\theta)^3}
$$

### Coupled dynamics

$$
\left(J+\frac{4I}{L^2\cos\phi}A\right)\ddot{\theta}
+\frac{4I}{L^2\cos\phi}B\dot{\theta}^2
+\frac{2k}{L\cos\phi}y
=k_t i-\frac{2k}{L\cos\phi}\delta
$$

Equivalent acceleration form:

$$
\ddot{\theta}
=\frac{
k_t i
-\frac{2k}{L\cos\phi}(y+\delta)
-\frac{4I}{L^2\cos\phi}B\dot{\theta}^2
}{
J+\frac{4I}{L^2\cos\phi}A
}
$$

Electrical subsystem:

$$
\dot{i}= -\frac{R_M}{L_M}i + \frac{k_b}{L_M}\dot{\theta} + \frac{u}{L_M}
$$

### Implemented assumptions

- Numerical integration uses explicit Euler with sample time `dt`.
- Output is modeled as `theta` (same signal used for fitting/prediction API).
- Parameters listed in `trainable_params` are optimized by minimizing free-run
  MSE with Adam (PyTorch).
- Positive physical parameters are constrained with softplus.
- Small epsilon guards avoid divisions near singular geometry points.

## Practical Notes

- For reliable fitting, resample and trim the dataset first.
- Start Model 2 with physically plausible initial parameters and a small
  trainable subset (`J`, `k`, `delta`, `k_t`) before adding more parameters.
- If training diverges, reduce learning rate and/or epochs and inspect
  `training_loss_`.
