# Angles: [roll, pitch, yaw] in radians. C_AB maps B to A.

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

results_dir = Path.cwd() / "results"
results_dir.mkdir(parents=True, exist_ok=True)


def EA2DCM(angles):
    phi, theta, psi = np.moveaxis(np.asarray(angles), -1, 0)

    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    sy, cy = np.sin(psi), np.cos(psi)

    return np.stack(
        [
            ct * cy,
            ct * sy,
            -st,
            sp * st * cy - cp * sy,
            sp * st * sy + cp * cy,
            sp * ct,
            cp * st * cy + sp * sy,
            cp * st * sy - sp * cy,
            cp * ct,
        ],
        axis=-1,
    ).reshape(np.shape(phi) + (3, 3))


def DCM2EA(C):
    C = np.asarray(C)

    theta = np.arcsin(np.clip(-C[..., 0, 2], -1.0, 1.0))
    phi = np.arctan2(C[..., 1, 2], C[..., 2, 2])
    psi = np.arctan2(C[..., 0, 1], C[..., 0, 0])

    # Set yaw to zero at gimbal lock.
    singular = np.abs(np.cos(theta)) < 1e-10
    phi = np.where(
        singular,
        np.arctan2(-C[..., 2, 1], C[..., 1, 1]),
        phi,
    )
    psi = np.where(singular, 0.0, psi)

    return np.stack([phi, theta, psi], axis=-1)


def euler_rates(t, angles, omega_amplitude):
    # Body frame angular velocity
    phi, theta, _ = angles

    p = omega_amplitude * np.sin(0.01 * t)
    q = 0.001
    r = omega_amplitude * np.cos(0.01 * t)

    if abs(np.cos(theta)) < 1e-12:
        raise RuntimeError("Euler-angle singularity at pitch = +/-90 degrees.")

    common = q * np.sin(phi) + r * np.cos(phi)

    return [
        p + np.tan(theta) * common,
        q * np.cos(phi) - r * np.sin(phi),
        common / np.cos(theta),
    ]


def plot_angles(time, angles, title, filename):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ["Roll, φ (deg)", "Pitch, θ (deg)", "Yaw, ψ (deg)"]

    for i, ax in enumerate(axes):
        ax.plot(time, np.degrees(angles[:, i]), linewidth=0.9)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.35)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(results_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


# Part c

time = np.linspace(0.0, 3600.0, 36001)  # Δt = 0.1 s

# Initial angles from part (b).
initial_target = np.radians([0.0, 0.0, 165.31264932710607])
initial_chaser = np.radians(
    [0.05378235584345724, -0.04249815016514536, 165.3226280164177]
)

target_solution = solve_ivp(
    euler_rates,
    (time[0], time[-1]),
    initial_target,
    args=(0.05,),
    t_eval=time,
    method="DOP853",
    max_step=0.1,
    rtol=1e-11,
    atol=1e-12,
)

chaser_solution = solve_ivp(
    euler_rates,
    (time[0], time[-1]),
    initial_chaser,
    args=(0.04,),
    t_eval=time,
    method="DOP853",
    max_step=0.1,
    rtol=1e-11,
    atol=1e-12,
)

if not target_solution.success:
    raise RuntimeError(target_solution.message)
if not chaser_solution.success:
    raise RuntimeError(chaser_solution.message)

target_angles = target_solution.y.T
chaser_angles = chaser_solution.y.T

plot_angles(
    time, target_angles, "Target Euler Angles Relative to Inertial Frame",
    "target_euler_angles.png",
)
plot_angles(
    time, chaser_angles, "Chaser Euler Angles Relative to Inertial Frame",
    "chaser_euler_angles.png",
)


# Part d

C_TN = EA2DCM(target_angles)
C_CN = EA2DCM(chaser_angles)

C_CT = C_CN @ np.swapaxes(C_TN, -1, -2)

orthogonality_error = np.linalg.norm(
    C_CT @ np.swapaxes(C_CT, -1, -2) - np.eye(3),
    ord="fro",
    axis=(1, 2),
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.semilogy(time, np.maximum(orthogonality_error, 1e-18), linewidth=0.7)
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\|C_{CT}C_{CT}^{T}-I\|_F$")
ax.set_title("Relative DCM Orthogonality Error")
ax.grid(True, which="both", alpha=0.35)
fig.tight_layout()
fig.savefig(results_dir / "relative_dcm_orthogonality_error.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Maximum orthogonality error: {orthogonality_error.max():.3e}")


# Part e

relative_angles = DCM2EA(C_CT)

# Roll/yaw jumps at +/-180 degrees are angle wrapping.
plot_angles(
    time,
    relative_angles,
    "Chaser Euler Angles Relative to Target Frame",
    "relative_euler_angles.png",
)

print(f"Figures saved to: {results_dir}")
