"""
rl_report.py - Generate Analytics and Graphs for your Teacher
=============================================================
This script loads your .zip model, runs 10 evaluation episodes,
records every action, and generates beautiful charts.
"""
import setup_path
import airsim
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN
from rl_drone_env import AirSimDroneSearchEnv

# --- CONFIG ---
MODEL_PATH = "rl_best/best_model.zip"
TEST_EPISODES = 10
ACTION_NAMES = ["Forward", "Up", "Down", "Yaw Left", "Yaw Right"]

def generate_report():
    print("=" * 50)
    print(" 📊 DRONE PERFORMANCE EVALUATION")
    print("=" * 50)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Could not find model at {MODEL_PATH}")
        return

    print("[1/3] Loading Environment and Brain...")
    env = AirSimDroneSearchEnv()
    model = DQN.load(MODEL_PATH)
    
    # Trackers for our graphs
    rewards = []
    steps_survived = []
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    end_reasons = {}

    print(f"[2/3] Running {TEST_EPISODES} Test Flights. Please wait...")
    
    for ep in range(1, TEST_EPISODES + 1):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_steps = 0
        
        while not done and not truncated:
            # Predict the best action (deterministic=True turns off random exploration)
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            
            # Record action
            action_counts[action] += 1
            
            # Take step
            obs, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            
        # Record episode results
        rewards.append(ep_reward)
        steps_survived.append(ep_steps)
        reason = info.get("reason", "unknown")
        end_reasons[reason] = end_reasons.get(reason, 0) + 1
        
        print(f"  Flight {ep}/{TEST_EPISODES} -> Result: {reason.upper()} | Steps: {ep_steps} | Reward: {ep_reward:.1f}")

    env.close()

    print("\n[3/3] Generating Teacher's Dashboard Graphs...")

    # --- PLOTTING THE GRAPHS ---
    plt.style.use('ggplot') # Makes graphs look professional
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Autonomous Drone Search - AI Performance Report', fontsize=16, fontweight='bold')

    # Graph 1: Reward per Episode
    axs[0, 0].plot(range(1, TEST_EPISODES + 1), rewards, marker='o', color='b', linewidth=2)
    axs[0, 0].set_title('Total Reward per Flight')
    axs[0, 0].set_xlabel('Flight Number')
    axs[0, 0].set_ylabel('Score (Reward)')
    axs[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Graph 2: Steps Survived
    axs[0, 1].bar(range(1, TEST_EPISODES + 1), steps_survived, color='g', alpha=0.7)
    axs[0, 1].set_title('Survival Time (Steps per Flight)')
    axs[0, 1].set_xlabel('Flight Number')
    axs[0, 1].set_ylabel('Steps Taken')

    # Graph 3: Action Distribution (What moves it likes)
    labels = [ACTION_NAMES[i] for i in range(5)]
    sizes = [action_counts[i] for i in range(5)]
    # Filter out actions it never took so the pie chart doesn't break
    labels = [l for l, s in zip(labels, sizes) if s > 0]
    sizes = [s for s in sizes if s > 0]
    axs[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0'])
    axs[1, 0].set_title('AI Decision Breakdown (Actions Chosen)')

    # Graph 4: End Reasons (Win/Loss Ratio)
    reasons = list(end_reasons.keys())
    counts = list(end_reasons.values())
    colors = ['green' if r == 'target_found' else 'red' if r in ['collision', 'ground_death'] else 'orange' for r in reasons]
    axs[1, 1].bar(reasons, counts, color=colors)
    axs[1, 1].set_title('Mission Outcomes')
    axs[1, 1].set_ylabel('Number of Flights')

    plt.tight_layout()
    
    # Save the picture so you can email/print it
    save_path = "drone_report.png"
    plt.savefig(save_path)
    print(f"\n✅ Done! Graphs saved as '{save_path}'")
    
    # Show the window
    plt.show()

if __name__ == "__main__":
    generate_report()