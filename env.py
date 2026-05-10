import numpy as np
import gymnasium as gym
from gymnasium import spaces
import portfolio_engine

class PortfolioEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, num_assets=10, initial_nav=1000000.0, taker_fee_bps=10.0, impact_coeff=0.01):
        super(PortfolioEnv, self).__init__()
        
        self.num_assets = num_assets
        
        # Engine setup
        self.engine = portfolio_engine.PortfolioEngine(
            num_assets, 
            initial_nav, 
            taker_fee_bps, 
            impact_coeff
        )
        
        # Action space: target weights for each asset. They should sum to 1.0 but
        # the model will output unconstrained continuous values which we'll normalize with softmax
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_assets,), dtype=np.float32)
        
        # Observation space: 
        # 1. Trailing returns for num_assets (let's say 1 step for simplicity)
        # 2. Rolling variance/covariance (simplified to 1 step var for now)
        # 3. Current portfolio weights
        self.obs_dim = self.num_assets * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        self.current_step = 0
        self.max_steps = 252 # 1 trading year
        self.initial_nav = initial_nav

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Re-initialize engine
        self.engine = portfolio_engine.PortfolioEngine(
            self.num_assets, 
            self.initial_nav, 
            10.0, 
            0.01
        )
        self.current_step = 0
        
        # Generate initial dummy observation
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # In a real environment, this would fetch real market data.
        # Here we mock it.
        mock_trailing_returns = np.random.normal(0, 0.01, self.num_assets)
        mock_rolling_var = np.random.uniform(0.0001, 0.001, self.num_assets)
        current_weights = np.array(self.engine.get_weights())
        
        obs = np.concatenate([
            mock_trailing_returns,
            mock_rolling_var,
            current_weights
        ]).astype(np.float32)
        return obs

    def step(self, action):
        # Softmax to ensure weights sum to 1.0 and are positive (long-only)
        exp_actions = np.exp(action - np.max(action))
        target_weights = exp_actions / exp_actions.sum()
        
        # Mock asset returns for the step
        asset_returns = np.random.normal(0.0005, 0.01, self.num_assets)
        
        prev_nav = self.engine.get_nav()
        
        # Call C++ Engine
        new_nav = self.engine.step(target_weights.tolist(), asset_returns.tolist())
        
        # Reward is the fractional change in NAV
        reward = (new_nav - prev_nav) / prev_nav
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        obs = self._get_obs()
        info = {'nav': new_nav}
        
        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"Step: {self.current_step}, NAV: {self.engine.get_nav()}")
