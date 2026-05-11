import time
import numpy as np
from stable_baselines3 import PPO

# Universe of 10 assets
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

class LocalPaperBroker:
    def __init__(self, initial_capital=1000000.0):
        self.equity = initial_capital
        self.positions = {sym: 0.0 for sym in SYMBOLS}
        self.current_weights = np.zeros(len(SYMBOLS), dtype=np.float32)

    def get_current_state(self):
        return self.equity, self.current_weights

    def execute_rebalance(self, target_weights):
        print(f"\n--- Rebalancing Portfolio ---")
        self.current_weights = target_weights
        for i, sym in enumerate(SYMBOLS):
            alloc = target_weights[i] * self.equity
            print(f"[{sym}] Target Allocation: ${alloc:,.2f} ({target_weights[i]*100:.1f}%)")
        print("-----------------------------\n")

def main():
    print("Initializing Zero-Setup Paper Broker...")
    broker = LocalPaperBroker(initial_capital=1000000.0)
    
    print("Loading PPO Model...")
    try:
        model = PPO.load("ppo_portfolio")
    except Exception as e:
        print(f"Warning: Model could not be loaded: {e}. Defaulting to equal weights.")
        model = None
        
    while True:
        try:
            equity, current_weights = broker.get_current_state()
            print(f"Current Paper Equity: ${equity:,.2f}")
            
            # Mock trailing observation data
            mock_trailing_returns = np.random.normal(0, 0.01, len(SYMBOLS)).astype(np.float32)
            mock_rolling_var = np.random.uniform(0.0001, 0.001, len(SYMBOLS)).astype(np.float32)
            
            obs = np.concatenate([
                mock_trailing_returns,
                mock_rolling_var,
                current_weights
            ]).astype(np.float32)
            
            if model:
                action, _ = model.predict(obs, deterministic=True)
                exp_actions = np.exp(action - np.max(action))
                target_weights = exp_actions / exp_actions.sum()
            else:
                target_weights = np.ones(len(SYMBOLS)) / len(SYMBOLS)
            
            broker.execute_rebalance(target_weights)

            # Mock market movement for the next tick
            market_return = np.sum(target_weights * np.random.normal(0.0005, 0.01, len(SYMBOLS)))
            broker.equity *= (1.0 + market_return)

            print("Rebalance complete. Sleeping 60 seconds (simulated tick)...")
            time.sleep(60)
            
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
