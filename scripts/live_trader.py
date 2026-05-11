import os
import time
import numpy as np
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from stable_baselines3 import PPO

# Environment variables
API_KEY = os.environ.get('ALPACA_API_KEY', 'your_api_key_here')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'your_secret_key_here')

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Universe of 10 assets
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V']

def get_current_state():
    account = trading_client.get_account()
    equity = float(account.equity)
    
    positions = {p.symbol: float(p.market_value) for p in trading_client.get_all_positions()}
    
    current_weights = []
    for sym in SYMBOLS:
        val = positions.get(sym, 0.0)
        current_weights.append(val / equity)
        
    return equity, np.array(current_weights, dtype=np.float32)

def main():
    print("Loading PPO Model...")
    try:
        model = PPO.load("ppo_portfolio")
    except Exception as e:
        print(f"Warning: Model could not be loaded: {e}. Ensure it is trained and saved.")
        model = None
        
    while True:
        try:
            equity, current_weights = get_current_state()
            print(f"Current Equity: {equity}")
            
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
            
            for i, sym in enumerate(SYMBOLS):
                target_weight = target_weights[i]
                current_weight = current_weights[i]
                
                delta_weight = target_weight - current_weight
                delta_value = delta_weight * equity
                
                if abs(delta_weight) > 0.01: # 1% threshold
                    side = OrderSide.BUY if delta_value > 0 else OrderSide.SELL
                    qty = abs(delta_value)
                    
                    print(f"Submitting {side} order for {sym}, value: {qty}")
                    req = MarketOrderRequest(
                        symbol=sym,
                        notional=round(qty, 2),
                        side=side,
                        time_in_force=TimeInForce.DAY
                    )
                    trading_client.submit_order(order_data=req)

            print("Rebalance complete. Sleeping 5 minutes...")
            time.sleep(300)
            
        except Exception as e:
            print(f"Error in loop: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
