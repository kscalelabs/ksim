import mediapy as media
import mujoco
import numpy as np
import scipy.linalg

# Define the cost coefficients and parameters
BALANCE_COST = 1000  # Balancing.
BALANCE_JOINT_COST = 3  # Joints required for balancing.
OTHER_JOINT_COST = 0.3  # Other joints.
DURATION = 12  # seconds
FRAMERATE = 60  # Hz
TOTAL_ROTATION = 15  # degrees
CTRL_RATE = 0.8  # seconds
BALANCE_STD = 0.01  # actuator units
OTHER_STD = 0.08  # actuator units


model = mujoco.MjModel.from_xml_path("stompy.xml")
data = mujoco.MjData(model)

# load keyframe "default"
qpos0 = model.keyframe("default").qpos
ctrl0 = np.zeros(model.nu)
dq = np.zeros(model.nv)

nu = model.nu  # number of actuators.
R = np.eye(nu)
nv = model.nv  # number of DoFs.

# Get the Jacobian for the root body (torso) CoM.
mujoco.mj_resetData(model, data)
data.qpos = qpos0
mujoco.mj_forward(model, data)
jac_com = np.zeros((3, nv))
mujoco.mj_jacSubtreeCom(model, data, jac_com, model.body("torso").id)

# Get the Jacobian for the left foot.
jac_foot = np.zeros((3, nv))
mujoco.mj_jacBodyCom(model, data, jac_foot, None, model.body("link_legs_1_left_leg_1_ankle_pulley_48t_1").id)

jac_diff = jac_com - jac_foot
Qbalance = jac_diff.T @ jac_diff

# Get all joint names.
joint_names = [model.joint(i).name for i in range(model.njnt)]

# Get indices into relevant sets of joints.
root_dofs = range(6)
body_dofs = range(6, nv)
abdomen_dofs = [model.joint(name).dofadr[0] for name in joint_names if "abdomen" in name and not "z" in name]
left_leg_dofs = [
    model.joint(name).dofadr[0]
    for name in joint_names
    if "left" in name and ("hip" in name or "knee" in name or "ankle" in name) and not "z" in name
]
balance_dofs = abdomen_dofs + left_leg_dofs
other_dofs = np.setdiff1d(body_dofs, balance_dofs)

# Construct the Qjoint matrix.
Qjoint = np.eye(nv)
Qjoint[root_dofs, root_dofs] *= 0  # Don't penalize free joint directly.
Qjoint[balance_dofs, balance_dofs] *= BALANCE_JOINT_COST
Qjoint[other_dofs, other_dofs] *= OTHER_JOINT_COST

# Construct the Q matrix for position DoFs.
Qpos = BALANCE_COST * Qbalance + Qjoint

# No explicit penalty for velocities.
Q = np.block([[Qpos, np.zeros((nv, nv))], [np.zeros((nv, 2 * nv))]])

# Set the initial state and control.
mujoco.mj_resetData(model, data)
data.ctrl = ctrl0
data.qpos = qpos0

# Allocate the A and B matrices, compute them.
A = np.zeros((2 * nv, 2 * nv))
B = np.zeros((2 * nv, nu))
epsilon = 1e-6
flg_centered = True
mujoco.mjd_transitionFD(model, data, epsilon, flg_centered, A, B, None, None)


# Solve discrete Riccati equation.
P = scipy.linalg.solve_discrete_are(A, B, Q, R)
# Compute the feedback gain matrix K.
K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A

# Make new camera, set distance.
camera = mujoco.MjvCamera()
mujoco.mjv_defaultFreeCamera(model, camera)
camera.distance = 2.3

# Enable contact force visualisation.
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Set the scale of visualized contact forces to 1cm/N.
model.vis.map.force = 0.01


# Define smooth orbiting function.
def unit_smooth(normalised_time: float) -> float:
    return 1 - np.cos(normalised_time * 2 * np.pi)


def azimuth(time: float) -> float:
    return 100 + unit_smooth(data.time / DURATION) * TOTAL_ROTATION


# Precompute some noise.
np.random.seed(1)
nsteps = int(np.ceil(DURATION / model.opt.timestep))
perturb = np.random.randn(nsteps, nu)

# Scaling vector with different STD for "balance" and "other"
CTRL_STD = np.empty(nu)
for i in range(nu):
    joint = model.actuator(i).trnid[0]
    dof = model.joint(joint).dofadr[0]
    CTRL_STD[i] = BALANCE_STD if dof in balance_dofs else OTHER_STD

# Smooth the noise.
width = int(nsteps * CTRL_RATE / DURATION)
kernel = np.exp(-0.5 * np.linspace(-3, 3, width) ** 2)
kernel /= np.linalg.norm(kernel)
for i in range(nu):
    perturb[:, i] = np.convolve(perturb[:, i], kernel, mode="same")

# Reset data, set initial pose.
mujoco.mj_resetData(model, data)
data.qpos = qpos0

# New renderer instance with higher resolution.
renderer = mujoco.Renderer(model, width=1280, height=720)

frames = []
step = 0
while data.time < DURATION:
    # Get state difference dx.
    mujoco.mj_differentiatePos(model, dq, 1, qpos0, data.qpos)
    dx = np.hstack((dq, data.qvel)).T

    # LQR control law.
    data.ctrl = ctrl0 - K @ dx

    # Add perturbation, increment step.
    data.ctrl += CTRL_STD * perturb[step]
    step += 1

    # Step the simulation.
    mujoco.mj_step(model, data)

    # Render and save frames.
    if len(frames) < data.time * FRAMERATE:
        camera.azimuth = azimuth(data.time)
        renderer.update_scene(data, camera, scene_option)
        pixels = renderer.render()
        frames.append(pixels)

media.write_video("video.mp4", frames, fps=30)
