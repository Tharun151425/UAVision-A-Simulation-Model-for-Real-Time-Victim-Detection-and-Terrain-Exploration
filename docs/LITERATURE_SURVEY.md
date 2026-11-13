# Literature Survey

This is a focused narrative review of the literature that shaped UAVision. It
is not a PRISMA-style systematic review, and it does not claim that the project
invents any of the underlying algorithms. The goal is to position the
engineering choices: why two simulators, why a curriculum, why recurrence, and
why the evaluation is deliberately conservative.

---

## 1. Search and rescue with aerial robots

Search and rescue (SAR) is a time-critical problem in which the search area,
target appearance, and environment are usually uncertain. Classical search
theory provides the mathematical foundation for allocating search effort under
uncertainty [1]. Robotic SAR adds mobility, sensing, and communication
constraints; Murphy et al. survey the field and emphasise that robots are most
useful when they extend human perception into unsafe or inaccessible space [2].

Small UAVs are attractive for wilderness and disaster response because they can
cover ground quickly and carry cameras into areas that are difficult to reach on
foot. Goodrich et al. demonstrated a camera-equipped mini-UAV supporting
wilderness search, including the human-robot interaction needed to make aerial
video useful to operators [3]. Waharte and Trigoni studied UAV search strategies
and highlighted the tension between area coverage and target re-observation [4].
Coverage-path-planning surveys extend this to systematic sweep patterns,
altitude selection, and energy constraints [5], [6].

Two lessons carry into UAVision:

1. **Coverage is not the same as search.** A policy can cover a map and still
   miss a target if its sensor footprint or revisit behaviour is wrong.
2. **The human loop matters.** Detection output is only useful if an operator
   can interpret location, confidence, and context. The AirSim HUD and
   segmentation overlay are a small nod to that requirement.

The project does not attempt a full operational SAR system. It isolates the
search policy and the perception contract in simulation so those pieces can be
studied before adding field constraints.

---

## 2. Exploration, coverage, and frontiers

Frontier-based exploration selects boundaries between known and unknown space
and has been a durable baseline since Yamauchi's work [7]. Coverage-path
planning surveys provide a complementary view: for a known region, geometric
patterns can be more efficient than reactive exploration [5], [6]. In unknown
regions, frontier or information-gain methods are usually stronger.

UAVision uses a lightweight version of the same idea. The fog-of-war map stores
unknown versus visited cells, and the GRU observation includes a local 16x16
patch. The policy is not given an explicit frontier planner; it must learn to
use the patch. This is intentional: the project studies whether a recurrent
policy can learn search behaviour from a compact spatial memory, not whether a
classical planner can be reimplemented.

The trade-off is sample complexity. A hand-written frontier planner would be
more sample-efficient and easier to certify. A learned policy is more flexible
and can integrate perception and control, but it needs a curriculum and careful
evaluation. The repository keeps both possibilities open: the map utilities are
separate from the policy, so a frontier planner can be added as a baseline.

---

## 3. Victim detection and perception

Real-time detection has been transformed by deep convolutional detectors.
YOLO framed detection as a single regression problem and made real-time
inference practical [8]. Subsequent work improved accuracy and speed, including
YOLOv4 [9] and anchor-free / focal-loss approaches [10]. Segmentation models
such as U-Net provide pixel-level masks for cases where the target boundary
matters [11]. Modern YOLO implementations package training, augmentation, and
deployment tooling [12].

UAVision's AirSim path uses **simulator segmentation** to identify the target.
That is a deliberate simplification: it removes detector error from the RL
loop so the search policy can be studied independently. It is not a claim that
segmentation is available on real hardware. A production system would need:

- a detector trained or fine-tuned on real imagery;
- temporal tracking to reject false positives;
- calibrated depth or geo-localisation;
- a human-in-the-loop confirmation step.

The PyBullet path goes further in the opposite direction: it replaces imagery
with first-hit ray codes. This is even more abstract, but it makes occlusion and
geometry explicit and cheap to train on. The two paths bracket the perception
problem: one is high-fidelity but scene-specific, the other is low-fidelity but
scalable.

---

## 4. Deep reinforcement learning for control

The modern deep RL toolkit begins with DQN, which combined Q-learning with a
deep network, experience replay, and a target network to learn Atari policies
from pixels [13]. Double DQN reduced overestimation bias [14], dueling
architectures improved value estimation [15], and prioritized experience replay
improved sample efficiency [16]. These ideas inform the AirSim DQN baseline.

Policy-gradient methods are a better fit for stochastic environments and
continuous or discrete control. TRPO introduced a trust-region update [17], and
PPO simplified the objective to a clipped surrogate that is easy to implement
and stable in practice [18]. PPO is the default for the PyBullet curriculum
because it tolerates dense reward shaping and runs well on CPU for small
networks. Soft actor-critic [19] is a natural alternative for continuous
control but is not used here.

The project uses Stable-Baselines3 for both DQN and PPO [20]. This is a
deliberate reproducibility choice: the algorithms are widely used, the
implementations are tested, and the training scripts are short enough to audit.
The contribution is the environment and curriculum, not a new optimizer.

---

## 5. Partial observability and recurrent policies

A SAR drone rarely has the full state. The target can be occluded, the map is
unknown, and sensor range is finite. The standard formal framing is a
partially observable Markov decision process (POMDP) [21]. Recurrent policies
address POMDPs by carrying a hidden state across time.

Deep recurrent Q-learning (DRQN) showed that an LSTM can integrate observations
in partially observable domains [22]. Heess et al. studied memory-based control
with recurrent networks and showed benefits on tasks that require integration
over time [23]. Recurrent PPO extends the PPO objective with a recurrent policy
and is available in `sb3-contrib`; UAVision uses it for the GRU phases.

The curriculum is designed around the distinction between **tracking** and
**searching**:

- In Phase 2A the target is visible at spawn, so memory is not required.
- In Phase 2B the target can disappear mid-flight, so the policy must remember
  its last bearing and re-acquire it.
- In Phase 2C the target can be hidden from the start, so the policy must
  explore and remember where it has been.

The fog patch is a second, explicit memory. Recurrence alone can represent
visited space, but a local map makes coverage observable and gives the policy a
stable feature for revisit avoidance. The project treats the LSTM and the fog
map as complementary, not redundant.

---

## 6. Curriculum learning and reward design

Curriculum learning trains on easier instances first and increases difficulty
as competence improves [24]. Narvekar et al. provide a framework and survey for
curriculum learning in RL, including task sequencing and transfer [25]. The
Phase-2 environments use a simple fixed-window gate: promote after 100 episodes
if success is high enough and crash/timeout rates are low enough.

This is a pragmatic design, not a theoretically optimal curriculum. It is easy
to inspect, easy to log, and easy to reproduce. The risk is noisy promotion:
a lucky window can move the agent to a harder level too early. The evaluation
guide therefore recommends a separate fixed-seed evaluation instead of trusting
training-time promotion.

Reward shaping is the other half of the curriculum. Ng et al. showed that
potential-based shaping preserves the optimal policy [26], but many practical
shaping terms do not satisfy that condition. UAVision uses non-potential terms
and documents their failure modes. For example, an early Phase 2A coverage
bonus paid for rays hitting obstacles, which made spinning next to a wall more
profitable than finding the survivor. The fix was to remove the coverage term,
gate distance shaping on visibility, and add an escalating anti-spin penalty.

This is a small case study in reward hacking, a broader failure mode in
learning systems [27]. The repository keeps the reward docstrings because they
are more useful to a future contributor than the final coefficients alone.

---

## 7. Simulation fidelity and sim-to-real

AirSim provides a high-fidelity visual and physical simulation for autonomous
vehicles, including camera, depth, segmentation, and LiDAR sensors [28]. It is
widely used for drone research and is a natural fit for perception-heavy work.
PyBullet provides a lightweight physics engine with a Python API and fast
headless stepping [29]. It is less visually realistic but much cheaper for
large-scale RL.

The sim-to-real gap is well documented. Domain randomisation can improve
transfer by varying textures, lighting, and dynamics during training [30].
CAD2RL showed that a single real image can be enough when simulation is
sufficiently randomised [31]. DroNet demonstrated learned flight from
camera data in real outdoor environments [32]. Surveys of sim-to-real RL
emphasise that transfer depends on the mismatch distribution, not just on
visual realism [33].

UAVision does **not** claim sim-to-real transfer. The two simulators are used
for different questions:

- AirSim: does the perception and control stack work with realistic sensors?
- PyBullet: can the search policy learn under a controlled, cheap curriculum?

A future hardware stage would need system identification, safety limits,
onboard compute constraints, and a detector validated on real imagery. Those
are explicitly out of scope for v1.0.

---

## 8. Multi-agent search and swarm coordination

Multi-agent RL and swarm robotics are the natural next step for large search
areas. PPO has been shown to be surprisingly effective in cooperative
multi-agent games when implemented with the right parameter sharing and
centralised training [34]. Swarm robotics reviews describe the engineering
challenges: local communication, scalability, robustness to agent loss, and
emergent coordination [35], [36].

The repository includes shared-map utilities and a branch named
`feature/sar-swarm`, but **it does not implement a multi-agent policy**. The
current map is shared by reference inside one process. A real swarm would need
a communication protocol, frontier assignment, collision avoidance between
agents, and a distributed consistency model. The roadmap lists these as future
work; the README does not present them as completed.

This distinction matters. A single-agent curriculum is already a substantial
engineering artifact. Claiming swarm behaviour without multi-agent training,
communication, and evaluation would be misleading.

---

## 9. Reproducibility and evaluation

Deep RL results are notoriously sensitive to seeds, hyperparameters, and
implementation details. Henderson et al. showed that variance across seeds can
be larger than the effect being studied [37]. Agarwal et al. recommend
distributional reporting and stratified bootstrap confidence intervals rather
than single-number comparisons [38]. Reproducibility initiatives emphasise
code, data, and protocol disclosure [39].

UAVision responds with:

- fixed seeds in smoke tests and evaluation examples;
- separate training and evaluation scripts;
- raw `evaluations.npz` for the baseline;
- a model card that states the baseline is weak;
- branch-separated history so changes can be audited;
- explicit observation and action contracts;
- a CI smoke test for the environment stack.

The project does not yet include a full statistical evaluation harness or
multi-seed benchmark table. That is a documented gap, not a hidden one.

---

## 10. Synthesis: gaps and project response

| Literature gap / practical problem | Project response | Status |
| --- | --- | --- |
| High-fidelity simulation is too slow for large-scale RL | PyBullet curriculum with compact observations | Implemented |
| Low-fidelity simulation lacks perceptual realism | AirSim RGB/depth/segmentation/LiDAR path | Implemented |
| Partial observability breaks feed-forward policies | RecurrentPPO + fog-of-war patch | Implemented |
| Reward shaping causes exploitation | Explicit reward terms, docstrings, ablations | Partially addressed |
| Curriculum gates are noisy | Fixed-window gates + separate evaluation guidance | Documented |
| Detection is often assumed, not studied | Ground-truth segmentation in AirSim; ray codes in PyBullet | Deliberate simplification |
| Sim-to-real is hard | No transfer claim; hardware stage on roadmap | Out of scope |
| Multi-agent coordination is hard | Shared map utilities; no multi-agent policy yet | Roadmap |
| Results are hard to reproduce | Seeds, logs, model card, branch history, CI | Partially addressed |

---

## 11. Positioning

UAVision does not introduce a new reinforcement-learning algorithm, a new
detector, or a new simulator. Its contribution is an **integration and
engineering** one:

- a multi-fidelity testbed that connects high-fidelity perception to a scalable
  search curriculum;
- explicit observation, action, and reward contracts that can be inspected and
  extended;
- a branch-separated history that preserves the evolution from manual flight to
  learned search;
- an honest baseline and a documented evaluation protocol.

That is the right scope for a research-engineering portfolio project. The
interesting claims are about the design and the failure modes, not about
state-of-the-art performance.

---

## References

[1] Stone, L. D. (1975). *Theory of Optimal Search*. Academic Press.

[2] Murphy, R. R., Tadokoro, S., Nardi, D., Jacoff, A., Fiorini, P., Choset, H., & Erkmen, A. M. (2008). Search and rescue robotics. In *Springer Handbook of Robotics* (pp. 1151-1173). Springer.

[3] Goodrich, M. A., Morse, B. S., Gerhardt, D., Cooper, J. L., Quigley, M., Adams, J. A., & Humphrey, C. (2008). Supporting wilderness search and rescue using a camera-equipped mini UAV. *Journal of Field Robotics*, 25(1-2), 89-110.

[4] Waharte, S., & Trigoni, N. (2010). Supporting search and rescue operations with UAVs. *International Conference on Emerging Security Technologies*, 142-147.

[5] Galceran, E., & Carreras, M. (2013). A survey on coverage path planning for robotics. *Robotics and Autonomous Systems*, 61(12), 1258-1276.

[6] Cabreira, T. M., Brisolara, L. B., & Ferreira Jr., P. R. (2019). Survey on coverage path planning for UAVs. *Drones*, 3(1), 4.

[7] Yamauchi, B. (1997). A frontier-based approach for autonomous exploration. *IEEE International Symposium on Computational Intelligence in Robotics and Automation (CIRA)*, 146-151.

[8] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, real-time object detection. *CVPR*. arXiv:1506.02640.

[9] Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4: Optimal speed and accuracy of object detection. arXiv:2004.10934.

[10] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *ICCV*.

[11] Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI*. arXiv:1505.04597.

[12] Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLOv8* [Software].

[13] Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518, 529-533.

[14] van Hasselt, H., Guez, A., & Silver, D. (2016). Deep reinforcement learning with double Q-learning. *AAAI*.

[15] Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.

[16] Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). Prioritized experience replay. *ICLR*. arXiv:1511.05952.

[17] Schulman, J., Levine, S., Abbeel, P., Jordan, M., & Moritz, P. (2015). Trust region policy optimization. *ICML*. arXiv:1502.05477.

[18] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

[19] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.

[20] Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22(268), 1-8.

[21] Kaelbling, L. P., Littman, M. L., & Cassandra, A. R. (1998). Planning and acting in partially observable stochastic domains. *Artificial Intelligence*, 101(1-2), 99-134.

[22] Hausknecht, M., & Stone, P. (2015). Deep recurrent Q-learning for partially observable MDPs. *AAAI Fall Symposium*. arXiv:1507.06527.

[23] Heess, N., Hunt, J. J., Lillicrap, T. P., & Silver, D. (2015). Memory-based control with recurrent neural networks. arXiv:1512.04455.

[24] Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *ICML*.

[25] Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). Curriculum learning for reinforcement learning domains: A framework and survey. *Journal of Machine Learning Research*, 21(181), 1-50.

[26] Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*.

[27] Amodei, D., Olah, C., Steinhardt, J., Christiano, P., Schulman, J., & Mané, D. (2016). Concrete problems in AI safety. arXiv:1606.06565.

[28] Shah, S., Dey, D., Lovett, C., & Kapoor, A. (2018). AirSim: High-fidelity visual and physical simulation for autonomous vehicles. *Field and Service Robotics (FSR)*, 621-635. arXiv:1705.05065.

[29] Coumans, E., & Bai, Y. (2016-2021). *PyBullet, a Python module for physics simulation for games, robotics and machine learning* [Software].

[30] Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W., & Abbeel, P. (2017). Domain randomization for transferring deep neural networks from simulation to the real world. *IROS*. arXiv:1703.06907.

[31] Sadeghi, F., & Levine, S. (2017). CAD2RL: Real single-image flight without a single real image. *RSS*. arXiv:1611.04201.

[32] Loquercio, A., Maqueda, A. I., del-Blanco, C. R., & Scaramuzza, D. (2018). DroNet: Learning to fly by driving. *IEEE Robotics and Automation Letters*, 3(2), 1088-1095.

[33] Zhao, W., Queralta, J. P., & Westerlund, T. (2020). Sim-to-real transfer in deep reinforcement learning for robotics: A survey. *IEEE Symposium Series on Computational Intelligence (SSCI)*.

[34] Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). The surprising effectiveness of PPO in cooperative multi-agent games. *NeurIPS*.

[35] Brambilla, M., Ferrante, E., Birattari, M., & Dorigo, M. (2013). Swarm robotics: A review from the swarm engineering perspective. *Swarm Intelligence*, 7, 1-41.

[36] Dorigo, M., Theraulaz, G., & Trianni, V. (2021). Swarm robotics: Past, present, and future. *Proceedings of the IEEE*, 109(7), 1152-1165.

[37] Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., & Meger, D. (2018). Deep reinforcement learning that matters. *AAAI*.

[38] Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *NeurIPS*.

[39] Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A., d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in machine learning research. *Journal of Machine Learning Research*, 22(164), 1-20.
