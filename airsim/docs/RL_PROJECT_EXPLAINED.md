# RL Drone Target-Finding Project: Interview Guide

> **Historical deep dive.** This walkthrough was written for interview
> preparation and reflects the environment at the time of the AirSim
> experiments. The authoritative interface is [`../README.md`](../README.md);
> where the two disagree, the source code wins.

This project is a reinforcement learning (RL) simulation where a drone learns to explore a 3D environment, avoid obstacles, and find a target object (called Dwight). It combines:

- AirSim for the drone simulator
- Gymnasium for the RL environment interface
- Stable Baselines 3 for the DQN learning algorithm
- Python for the control and training logic

The project is designed to teach an agent to behave like a drone pilot that can search, move, and avoid risky situations without being manually coded for every action.

---

## 1. Project goal in one sentence

The drone receives sensor information from the environment, chooses actions such as move forward, rise, descend, or yaw, and receives rewards or penalties depending on whether it gets closer to the target or crashes into obstacles.

This is a classic reinforcement learning loop:

- State = what the drone sees and senses
- Action = what the drone does next
- Reward = success or failure signal
- Policy = the model learns which actions are good over time

---

## 2. High-level architecture

The workflow is:

1. The environment creates a drone in AirSim.
2. The drone gets observations:
   - camera depth image
   - 8-direction LiDAR distances
   - target direction vector
3. The DQN model chooses an action.
4. The action is executed in AirSim.
5. The environment returns reward + next observation.
6. The model trains on many episodes and improves its policy.

Files involved:

- [../rl_drone_env.py](../rl_drone_env.py) — environment definition, sensor logic, rewards, step/reset logic
- [../rl_train.py](../rl_train.py) — training loop, DQN setup, model saving/resume
- [../rl_play.py](../rl_play.py) — runs the trained model in demo mode

---

## 3. Why this project is important

This project is a strong example of applied RL because it combines:

- simulated robotics
- sensor processing
- decision-making under uncertainty
- reward engineering
- safe training (collision avoidance, altitude limits)

In an interview, this is a good project to describe as: "I built a reinforcement learning drone navigation system in AirSim where the agent learns to search for a target using camera and LiDAR data, while avoiding obstacles and learning from reward shaping."

---

## 4. Deep dive into the environment file

### 4.1 File purpose

The main file [../rl_drone_env.py](../rl_drone_env.py) defines the Gymnasium environment for the drone. In RL, an environment is the simulation world the agent interacts with.

The environment defines:

- what observations the agent sees
- what actions it can take
- how rewards are assigned
- when an episode ends

---

### 4.2 Important constants and settings

These constants tell the agent how the world works.

```python
TARGET_MESH = "PersonTarget*"
TARGET_LABEL = "Dwight"
VEHICLE = "Drone1"
CAMERA = "0"
LIDAR_SURR = "SurroundLidar"
```

These lines tell the program which drone, camera, and target object to use. The project is not generic by default; it is specifically built around the AirSim drone named Drone1 and a target object named Dwight.

```python
STEP_LENGTH = 5.0
VERT_STEP = 2.0
YAW_STEP = 45.0
MAX_STEPS = 200
IMAGE_SHAPE = (84, 84, 1)
```

These parameters define how the drone moves:

- forward step is 5 meters
- up/down step is 2 meters
- yaw step is 45 degrees
- max episode length is 200 steps
- image size is 84x84 depth image

These are important because they control the complexity and training difficulty. A larger step or more complex observation increases learning difficulty.

```python
SPAWN_MIN_XY = -15.0
SPAWN_MAX_XY =  15.0
CONE_MIN_DIST = 20.0
CONE_MAX_DIST = 40.0
```

This randomizes the drone starting position and target position, so the agent learns in varied conditions instead of memorizing one fixed map.

This is called domain randomization. It makes the agent more robust.

---

### 4.3 Action space

```python
self.action_space = spaces.Discrete(5)
```

The action space has 5 discrete actions:

0 = Forward
1 = Up
2 = Down
3 = Yaw Left
4 = Yaw Right

This is simpler than continuous control. It is easier for DQN to learn because the model only chooses among 5 actions instead of continuous steering values.

---

### 4.4 Observation space

```python
self.observation_space = spaces.Dict({
    "image": spaces.Box(0, 255, shape=IMAGE_SHAPE, dtype=np.uint8),
    "lidar": spaces.Box(0, 100, shape=(8,), dtype=np.float32),
    "target_dir": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
})
```

This defines the agent’s observation.

- image: a depth image from the drone camera, resized to 84x84 and single channel
- lidar: 8 values corresponding to distance in 8 directions
- target_dir: a 3D vector showing direction toward the target

This means the model is not blind. It sees the environment in multiple ways.

---

### 4.5 Environment reset and randomization

```python
def _randomize_scene(self):
```

This function creates a new training scenario for each episode.

It does three main things:

1. Randomly places the drone at a new location
2. Randomly places the target at a reasonable distance
3. Randomly positions obstacle cubes away from the drone and target

Why this matters:

- the agent learns general behavior, not one specific map
- the training becomes harder and more realistic
- the drone can learn to navigate in many layouts

---

### 4.6 Flight setup and hover fix

```python
def _setup_flight(self):
```

This is one of the most important functions in the project.

It does the following in order:

- reset the simulator
- enable API control
- arm the drone
- take off
- pause physics
- teleport to starting pose
- command the drone to stay in place
- unpause and let it settle

This is specifically designed to fix a major RL problem: the drone falling or not hovering correctly after reset.

The comments in the code mention:

- “FIXED — DRONE ACTUALLY HOVERS NOW”
- “hover bug”

This is critical in real RL training because if the drone state is unstable, learning becomes noisy and unreliable.

---

### 4.7 Sensor logic

#### Depth image

```python
def _get_depth_image(self):
```

This function requests a depth image from AirSim, converts it, and resizes it to 84x84. The depth image becomes the visual input to the policy.

This is important because the agent uses visual information to understand the scene and estimate where obstacles or the target may be.

#### LiDAR

```python
def _get_lidar_sectors(self):
```

LiDAR is used to scan the environment in 8 sectors. Each sector contains the shortest distance to an obstacle in that direction.

This helps the agent avoid collisions. If an obstacle is too close, the reward is reduced and the drone can change direction.

#### Target direction vector

```python
def _get_target_direction(self):
```

This computes the direction vector from the drone to the target. It normalizes the vector to unit length so the model gets a consistent signal.

This is one of the most important inputs because it tells the agent where the target is relative to the current pose.

---

### 4.8 Action execution

```python
def _do_action(self, action):
```

This function executes one of the five actions.

For action 0 (Forward):

- compute forward vector based on drone yaw
- move 5 m ahead

For action 1 and 2:

- move upward or downward by 2 m

For action 3 and 4:

- rotate left or right by 45 degrees using `rotateToYawAsync`

Important idea: the model does not directly command continuous velocity. It chooses a high-level action, and the environment translates it into simulator motion.

This is typical in RL for robotics and game environments.

---

### 4.9 Reward engineering

```python
def _compute_reward(self, action):
```

This is the heart of the learning problem.

The reward is not just “+1 if target found.” It is carefully shaped.

The key logic is:

- if ground contact occurs → huge penalty
- if collision occurs → huge penalty
- if target is found → huge reward
- if out of bounds → penalty
- if the agent moves closer to target → positive reward
- if it faces target while moving forward → extra bonus
- if it spins too much → penalty
- if it is at healthy altitude → bonus
- if obstacle is near → penalty
- if it explores new space → small bonus

This is called reward shaping.

Why it matters: without good rewards, the model may not learn efficient behavior. The shaped rewards guide the drone toward safe and promising trajectories.

Examples:

```python
if ground_contact:
    reward = -200.0
    done = True
```

This trains the drone to avoid death.

```python
if target_found and dist < TARGET_FOUND_DIST:
    reward = 200.0
    done = True
```

This tells the agent that finding the target is the main objective.

```python
reward += delta * 3.0
```

This rewards progress toward the target. If the drone gets closer, the reward increases.

```python
if self._consecutive_yaws >= 3:
    reward -= 2.0
```

This discourages useless spinning in place.

This is a very important idea in RL interviews: the reward function is often more important than the model itself.

---

### 4.10 Step and reset functions

```python
def step(self, action):
```

This is the standard Gymnasium step method.

The environment:

1. executes the action
2. increments the step counter
3. captures updated observation
4. computes reward and terminal condition
5. returns `obs, reward, done, truncated, info`

This is the exact interface used by RL libraries like Stable Baselines 3.

```python
def reset(self, seed=None, options=None):
```

This resets the episode by:

- scanning scene objects
- randomizing the environment
- placing the drone in a start state
- returning the first observation

This ensures each episode starts in a valid and fresh state.

---

## 5. Training file: DQN logic

The file [../rl_train.py](../rl_train.py) is where the model is actually trained.

### 5.1 Why DQN?

DQN stands for Deep Q-Network. It is a classic RL method that approximates Q-values using a neural network.

It learns to estimate:

"If I take action a in state s, how much total future reward will I get?"

This suits the discrete 5-action environment well.

---

### 5.2 Environment wrapper

```python
env = DummyVecEnv([make_env])
env = VecTransposeImage(env)
```

This wraps the environment to allow vectorized training and correct image format handling.

Why it matters:

- RL frameworks often train with multiple environments in parallel
- image tensors need a correct channel layout
- `VecTransposeImage` handles common observation shape mismatches

---

### 5.3 Hyperparameters

```python
TOTAL_TIMESTEPS = 50_000
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
BUFFER_SIZE = 30_000
GAMMA = 0.99
```

These are important tuning parameters.

- learning rate: affects how quickly the model updates
- batch size: number of transitions used in each optimization step
- replay buffer: past experiences stored for learning
- gamma: discount factor for future rewards

The model learns from old transitions, not only the most recent ones. This helps stability.

---

### 5.4 Resume logic

```python
def find_latest_checkpoint():
```

This function looks for the newest saved model so training can continue after interruption.

This is very useful in long RL training because:

- training can take a long time
- Ctrl+C or crashes are common in simulations
- the project supports resumable training

This is a good technical feature to mention in interviews: the training loop is robust and able to continue from checkpoints.

---

### 5.5 DQN initialization

```python
model = DQN(
    "MultiInputPolicy",
    env,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    buffer_size=BUFFER_SIZE,
    gamma=GAMMA,
    ...
)
```

This creates the actual Deep Q-Network. It uses a `MultiInputPolicy` because the observation includes multiple inputs: image, LiDAR, and target vector.

This is exactly why the environment is designed as a dictionary of different observation types.

---

### 5.6 Evaluation callback

```python
eval_callback = EvalCallback(...)
```

During training, the model is periodically evaluated on a separate environment to check if it is improving.

This helps avoid overfitting to only a single training scenario.

---

### 5.7 Checkpoint callback

```python
checkpoint_callback = CheckpointCallback(...)
```

This saves intermediate models so you can continue later without losing progress.

This is essential in real-world RL since training can be expensive and time-consuming.

---

### 5.8 Training loop

```python
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    ...
)
```

This is the main learning command.

The model repeatedly:

- acts in the environment
- stores experience
- samples minibatches
- updates Q-values
- improves policy

---

## 6. Demo file: inference and live rendering

The file [../rl_play.py](../rl_play.py) loads a trained model and shows the drone acting autonomously.

This is used for:

- verifying trained behavior
- debugging model mistakes
- showing a demo to a user or interviewer

Key parts:

- `RLDemoPlayer.__init__()` loads the saved model
- `play()` loops through episodes
- `self.model.predict(obs, deterministic=True)` selects the action
- the environment executes the action and updates the visual HUD
- minimap is displayed for the drone path and obstacles

This file turns the trained policy into real behavior in the simulator.

---

## 7. Why this is a strong interview project

You can explain it in a professional interview like this:

> I built a reinforcement learning drone navigation project in AirSim. The drone uses depth camera and LiDAR observations to decide among five actions: forward, ascend, descend, rotate left, and rotate right. I defined a custom Gymnasium environment with randomized spawn positions and a reward function that encourages progress toward the target while penalizing collisions, falling, spin loops, and unsafe altitude. I trained a DQN agent in Stable Baselines 3 and added checkpointing, evaluation, and resume support so training can continue after interruption. The project demonstrates practical RL, reward shaping, robotics simulation, and model validation in a realistic environment.

That summary is strong because it includes:

- project problem
- environment design
- reward function
- algorithm choice
- training pipeline
- evaluation and robustness

---

## 8. Important technical concepts to learn before the interview

### Reinforcement learning

The agent learns by trial and error through rewards.

### Gymnasium environment

The environment provides the standard RL API: `reset()` and `step()`.

### Reward shaping

The reward is designed to push the drone toward useful behavior.

### DQN

A neural network approximates the optimal action-value function.

### Domain randomization

The agent sees varied conditions, which improves generalization.

### Safety constraints

The environment prevents dangerous behavior through penalties and altitude checks.

---

## 9. Your interview answer in 5 points

1. This project solves autonomous drone navigation using RL.
2. The environment is custom-built using AirSim and Gymnasium.
3. Observations include image, LiDAR, and target direction.
4. The model is trained with DQN using shaped rewards and randomized scenarios.
5. The project shows robotics, simulation, reward design, and practical ML engineering in one pipeline.

---

## 10. Final takeaway

This is not just “a drone flying around.” It is a complete RL system that teaches a robot to act intelligently under uncertainty.

The strongest parts of the project are:

- realistic simulation environment
- careful reward shaping
- safe training constraints
- modular RL pipeline
- checkpointing and resume support
- demo / validation workflow

If you can explain these clearly in an interview, you will sound very strong.
