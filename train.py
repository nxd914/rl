import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import PortfolioEnv

def main():
    print("Setting up the Portfolio Environment...")
    env = PortfolioEnv(num_assets=10, initial_nav=1000000.0)
    
    # Wrap it
    vec_env = DummyVecEnv([lambda: env])

    print("Initializing PPO Agent...")
    model = PPO("MlpPolicy", vec_env, verbose=1)

    print("Training...")
    model.learn(total_timesteps=2000)

    print("Training complete. Evaluating...")
    obs = vec_env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        print(f"Eval Step {i+1} | NAV: {info[0]['nav']:.2f} | Reward: {rewards[0]:.4f}")

if __name__ == "__main__":
    main()
