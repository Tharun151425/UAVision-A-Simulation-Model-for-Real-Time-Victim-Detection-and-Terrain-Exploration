"""
Phase 2B GRU Environment — Partial Occlusion
=============================================
Obs: 519 (fixed). New: survivor can disappear behind obstacles.
GRU hidden state (trained since Phase 1) now carries last-seen
survivor direction, obstacle pattern, steps-since-loss.
MAP_SIZE=45, 3-level curriculum by hidden_prob.
"""

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import math, sys, os
from collections import deque

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _PROJ_ROOT)

from sensors.raycast_gru                      import RaycastSensor
from utils.fog_map                             import FogOfWarMap
from phase_2.phase_2B.rewards.phase2B_rewards_gru import compute_reward_2B_gru

CURRICULUM_LEVELS = {
    1: {"min_obstacles": 8,  "max_obstacles": 12, "hidden_prob": 0.30,
        "max_steps": 1500, "min_height": 2.0, "max_height": 6.0,
        "shapes": ["cylinder","box","tree"],
        "drone_safe_dist": 8.0, "survivor_safe_dist": 3.0,
        "promote_success": 0.68, "promote_crash": 0.12, "promote_timeout": 0.20,
        "description": "2B-L1 — 30% hidden at spawn"},
    2: {"min_obstacles": 10, "max_obstacles": 15, "hidden_prob": 0.50,
        "max_steps": 2000, "min_height": 2.0, "max_height": 7.0,
        "shapes": ["cylinder","box","tree","rock","wall"],
        "drone_safe_dist": 8.0, "survivor_safe_dist": 3.0,
        "promote_success": 0.65, "promote_crash": 0.12, "promote_timeout": 0.23,
        "description": "2B-L2 — 50% hidden at spawn"},
    3: {"min_obstacles": 12, "max_obstacles": 18, "hidden_prob": 0.70,
        "max_steps": 2500, "min_height": 2.0, "max_height": 8.0,
        "shapes": ["cylinder","box","tree","rock","wall"],
        "drone_safe_dist": 8.0, "survivor_safe_dist": 3.0,
        "promote_success": None, "promote_crash": None, "promote_timeout": None,
        "description": "2B-L3 — 70% hidden (pass-criteria level)"},
}

PASS_CRITERIA = {"success_rate": 65.0, "crash_rate": 12.0, "timeout_rate": 23.0}


class Phase2BEnvGRU(gym.Env):

    MAP_SIZE            = 45
    RAY_RANGE           = 20.0
    FORWARD_STEP        = 1.0
    YAW_STEP            = 15.0
    ALTITUDE_STEP       = 0.5
    MIN_ALTITUDE        = 1.0
    MAX_ALTITUDE        = 10.0
    SURVIVOR_REACH_DIST = 2.0
    SURVIVOR_MIN_DIST   = 10
    SURVIVOR_MAX_DIST   = 30
    OBS_SIZE            = 519

    TURN_LEFT=0; TURN_RIGHT=1; MOVE_UP=2; MOVE_DOWN=3; MOVE_FORWARD=4; SCAN=5

    @property
    def MAX_STEPS(self):
        return CURRICULUM_LEVELS[self.curriculum_level]["max_steps"]

    def __init__(self, render=False, curriculum_level=1):
        super().__init__()
        self.render_mode      = render
        self.curriculum_level = max(1, min(3, curriculum_level))
        self._outcome_success = deque(maxlen=100)
        self._outcome_crash   = deque(maxlen=100)
        self._outcome_timeout = deque(maxlen=100)
        self._episode_count   = 0

        self.observation_space = gym.spaces.Box(
            low=-1.0, high=10.0, shape=(self.OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(6)

        if render:
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(cameraDistance=40,cameraPitch=-55,
                                         cameraYaw=45,cameraTargetPosition=[0,0,0])
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)

        self.sensor  = RaycastSensor(h_rays=32, v_rays=8, h_fov=120, v_fov=60,
                                     max_range=self.RAY_RANGE)
        self.fog_map = FogOfWarMap(world_size=self.MAP_SIZE, cell_size=2.0)

        # State
        self.drone_x = self.drone_y = self.drone_alt = 0.0
        self.drone_yaw = 0.0; self.survivor_x = self.survivor_y = 0.0
        self.ground_id = self.drone_id = self.survivor_id = -1
        self.obstacle_ids = []
        self.survivor_detected    = False
        self.survivor_was_visible = False
        self._consecutive_turns   = 0
        self._lost_for_steps      = 0
        self._reacquisitions      = 0
        self.steps_since_signal_loss = 0
        self.prev_dist = 0.0; self.done = False; self.step_count = 0
        self._last_ray_grid = np.zeros((8, 32), dtype=np.int32)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(); p.setGravity(0, 0, 0)
        self.step_count = 0; self.done = False
        self.survivor_detected    = False
        self.survivor_was_visible = False
        self._consecutive_turns   = 0
        self._lost_for_steps      = 0
        self._reacquisitions      = 0
        self.steps_since_signal_loss = 0
        self.fog_map.reset()
        self.ground_id = p.loadURDF("plane.urdf")
        if seed is not None: np.random.seed(seed)

        half = (self.MAP_SIZE / 2) - 1.5
        lvl  = CURRICULUM_LEVELS[self.curriculum_level]

        for _ in range(200):
            self.drone_x   = np.random.uniform(-half, half)
            self.drone_y   = np.random.uniform(-half, half)
            self.drone_alt = np.random.uniform(3.0, 5.0)
            self.drone_yaw = np.random.uniform(0, 360)
            dist  = np.random.uniform(self.SURVIVOR_MIN_DIST, self.SURVIVOR_MAX_DIST)
            angle = np.random.uniform(0, 2 * math.pi)
            sx = self.drone_x + dist * math.cos(angle)
            sy = self.drone_y + dist * math.sin(angle)
            if -half <= sx <= half and -half <= sy <= half:
                self.survivor_x = sx; self.survivor_y = sy; self.prev_dist = dist; break

        self.drone_id    = self._spawn_drone()
        self.survivor_id = self._spawn_survivor()
        p.addUserDebugLine([0.3,0.0,0.1],[1.5,0.0,0.1],lineColorRGB=[1,0,0],
                           lineWidth=4,parentObjectUniqueId=self.drone_id,parentLinkIndex=-1)

        n = np.random.randint(lvl["min_obstacles"], lvl["max_obstacles"] + 1)
        self.obstacle_ids = self._spawn_obstacles(n, lvl)

        self.fog_map.visit(self.drone_x, self.drone_y)
        self._last_ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id, physics_client=self.physics_client)
        return self._get_obs(self._last_ray_grid), {}

    def step(self, action):
        self.step_count += 1
        self._apply_action(action); p.stepSimulation()
        newly_revealed = int(self.fog_map.visit(self.drone_x, self.drone_y))

        ray_grid, _ = self.sensor.fire(
            self.drone_x, self.drone_y, self.drone_alt, self.drone_yaw,
            ground_id=self.ground_id, obstacle_ids=self.obstacle_ids,
            survivor_id=self.survivor_id, physics_client=self.physics_client)
        self._last_ray_grid = ray_grid

        current_dist   = math.hypot(self.drone_x-self.survivor_x, self.drone_y-self.survivor_y)
        delta_dist     = self.prev_dist - current_dist; self.prev_dist = current_dist

        obs    = self._get_obs(ray_grid)
        reward = compute_reward_2B_gru(self, action, delta_dist, newly_revealed)

        reason="Ongoing"; crashed=False; found=False
        if self.done:
            if self.survivor_reached():     reason="Survivor Found";    found=True
            elif self.collision_detected(): reason="Crashed (Obstacle)"; crashed=True
            elif self.drone_alt < self.MIN_ALTITUDE: reason="Crashed (Alt)"; crashed=True
        if not self.done and self.step_count >= self.MAX_STEPS:
            reason="Timeout"; self.done=True

        if self.done:
            self._episode_count += 1
            self._outcome_success.append(1 if found   else 0)
            self._outcome_crash.append(1   if crashed else 0)
            self._outcome_timeout.append(1 if "Timeout" in reason else 0)
            self._maybe_promote()

        info = {"step":self.step_count,"survivor_found":found,"crashed":crashed,
                "curriculum_level":self.curriculum_level,"termination_reason":reason,
                "cells_explored":self.fog_map.total_visited()}
        return obs, reward, self.done, False, info

    def _maybe_promote(self):
        if self.curriculum_level >= 3: return
        lvl = CURRICULUM_LEVELS[self.curriculum_level]
        w   = 100
        if len(self._outcome_success) < w: return
        sr=sum(self._outcome_success)/w; cr=sum(self._outcome_crash)/w; tr=sum(self._outcome_timeout)/w
        if sr>=lvl["promote_success"] and cr<=lvl["promote_crash"] and tr<=lvl["promote_timeout"]:
            old=self.curriculum_level; self.curriculum_level+=1
            self._outcome_success.clear(); self._outcome_crash.clear(); self._outcome_timeout.clear()
            print(f"\n[Phase 2B] PROMOTION: Level {old} → {self.curriculum_level}"
                  f"  success={sr*100:.1f}%  crash={cr*100:.1f}%  timeout={tr*100:.1f}%\n")

    def _get_obs(self, ray_grid=None):
        if ray_grid is None: ray_grid = np.zeros((8,32), dtype=np.float32)
        half = self.MAP_SIZE / 2
        rays    = ray_grid.flatten().astype(np.float32)
        ego     = np.array([self.drone_x/half, self.drone_y/half,
                             self.drone_alt/self.MAX_ALTITUDE,
                             math.cos(math.radians(self.drone_yaw)),
                             math.sin(math.radians(self.drone_yaw))], dtype=np.float32)
        fog     = self.fog_map.get_patch(self.drone_x, self.drone_y, 16)
        ts      = np.array([self.step_count / self.MAX_STEPS], dtype=np.float32)
        terrain = np.array([0.0], dtype=np.float32)
        return np.concatenate([rays, ego, fog, ts, terrain])

    def _apply_action(self, action):
        if action==self.TURN_LEFT:    self.drone_yaw=(self.drone_yaw-self.YAW_STEP)%360
        elif action==self.TURN_RIGHT: self.drone_yaw=(self.drone_yaw+self.YAW_STEP)%360
        elif action==self.MOVE_UP:    self.drone_alt=min(self.drone_alt+self.ALTITUDE_STEP,self.MAX_ALTITUDE)
        elif action==self.MOVE_DOWN:  self.drone_alt=max(self.drone_alt-self.ALTITUDE_STEP,self.MIN_ALTITUDE)
        elif action==self.MOVE_FORWARD:
            yr=math.radians(self.drone_yaw)
            self.drone_x+=self.FORWARD_STEP*math.cos(yr); self.drone_y+=self.FORWARD_STEP*math.sin(yr)
        half=self.MAP_SIZE/2
        self.drone_x=float(np.clip(self.drone_x,-half,half)); self.drone_y=float(np.clip(self.drone_y,-half,half))
        pos=[self.drone_x,self.drone_y,self.drone_alt]
        quat=p.getQuaternionFromEuler([0,0,math.radians(self.drone_yaw)])
        p.resetBasePositionAndOrientation(self.drone_id,pos,quat)
        p.resetBaseVelocity(self.drone_id,[0,0,0],[0,0,0])

    def _spawn_drone(self):
        col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[0.3,0.3,0.1])
        vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[0.3,0.3,0.1],rgbaColor=[0.2,0.4,0.9,1.0])
        quat=p.getQuaternionFromEuler([0,0,math.radians(self.drone_yaw)])
        return p.createMultiBody(baseMass=1.0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                                 basePosition=[self.drone_x,self.drone_y,self.drone_alt],baseOrientation=quat)

    def _spawn_survivor(self):
        col=p.createCollisionShape(p.GEOM_SPHERE,radius=1.0)
        vis=p.createVisualShape(p.GEOM_SPHERE,radius=0.5,rgbaColor=[1.0,0.1,0.1,1.0])
        return p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,
                                 basePosition=[self.survivor_x,self.survivor_y,0.5])

    def _spawn_obstacles(self, count, lvl):
        obs_ids=[]; placed=[]; half=(self.MAP_SIZE/2)-0.5
        for _ in range(count):
            for _ in range(300):
                x=np.random.uniform(-half,half); y=np.random.uniform(-half,half)
                if math.hypot(self.drone_x-x,self.drone_y-y)<lvl["drone_safe_dist"]: continue
                if math.hypot(self.survivor_x-x,self.survivor_y-y)<lvl["survivor_safe_dist"]: continue
                if any(math.hypot(px-x,py-y)<2.0 for px,py in placed): continue
                h=np.random.uniform(lvl["min_height"],lvl["max_height"])
                shape=np.random.choice(lvl["shapes"])
                ids=self._create_obstacle(shape,x,y,h)
                obs_ids.extend(ids); placed.append((x,y)); break
        return obs_ids

    def _create_obstacle(self, shape, x, y, h):
        if shape=="cylinder":
            col=p.createCollisionShape(p.GEOM_CYLINDER,radius=0.5,height=h)
            vis=p.createVisualShape(p.GEOM_CYLINDER,radius=0.5,length=h,rgbaColor=[0.55,0.35,0.15,1.0])
            return [p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=[x,y,h/2])]
        elif shape=="box":
            hw=np.random.uniform(0.5,2.5); hd=np.random.uniform(0.5,2.5)
            col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[hw,hd,h/2])
            vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[hw,hd,h/2],rgbaColor=[0.45,0.45,0.45,1.0])
            return [p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=[x,y,h/2])]
        elif shape=="tree":
            th=h*0.6; ct=p.createCollisionShape(p.GEOM_CYLINDER,radius=0.2,height=th)
            vt=p.createVisualShape(p.GEOM_CYLINDER,radius=0.2,length=th,rgbaColor=[0.4,0.25,0.1,1.0])
            tid=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=ct,baseVisualShapeIndex=vt,basePosition=[x,y,th/2])
            cr=np.random.uniform(0.8,1.5); cc=p.createCollisionShape(p.GEOM_SPHERE,radius=cr)
            vc=p.createVisualShape(p.GEOM_SPHERE,radius=cr,rgbaColor=[0.15,0.55,0.15,1.0])
            cid=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=cc,baseVisualShapeIndex=vc,basePosition=[x,y,th+cr*0.7])
            return [tid,cid]
        elif shape=="rock":
            rx=np.random.uniform(0.6,1.5); ry=np.random.uniform(0.6,1.5); rz=np.random.uniform(0.25,0.75)
            col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[rx,ry,rz])
            vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[rx,ry,rz],rgbaColor=[0.5,0.47,0.44,1.0])
            return [p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=[x,y,rz])]
        elif shape=="wall":
            wl=np.random.uniform(3.0,6.0); wh=np.random.uniform(2.0,5.0)
            col=p.createCollisionShape(p.GEOM_BOX,halfExtents=[wl/2,0.2,wh/2])
            vis=p.createVisualShape(p.GEOM_BOX,halfExtents=[wl/2,0.2,wh/2],rgbaColor=[0.7,0.7,0.65,1.0])
            return [p.createMultiBody(baseMass=0,baseCollisionShapeIndex=col,baseVisualShapeIndex=vis,basePosition=[x,y,wh/2])]
        return []

    def get_min_obstacle_dist(self):
        min_dist=self.RAY_RANGE
        for oid in self.obstacle_ids:
            pts=p.getClosestPoints(self.drone_id,oid,distance=6.0)
            for pt in pts:
                if pt[8]<min_dist: min_dist=pt[8]
        return min_dist

    def collision_detected(self):
        for oid in self.obstacle_ids:
            if p.getContactPoints(self.drone_id,oid): return True
        return False

    def survivor_reached(self):
        return math.hypot(self.drone_x-self.survivor_x,self.drone_y-self.survivor_y)<=self.SURVIVOR_REACH_DIST

    def survivor_in_fov(self, ray_grid=None):
        if ray_grid is None: ray_grid=self._last_ray_grid
        return int(3) in ray_grid

    def get_curriculum_level(self): return self.curriculum_level
    def set_curriculum_level(self, level):
        self.curriculum_level=max(1,min(3,level))
        self._outcome_success.clear(); self._outcome_crash.clear(); self._outcome_timeout.clear()

    def render(self): pass
    def close(self): p.disconnect(self.physics_client)
