#include "engine.hpp"

PortfolioEngine::PortfolioEngine(int num_assets, double initial_nav, double taker_fee_bps, double impact_coeff)
    : num_assets(num_assets), current_nav(initial_nav), taker_fee_bps(taker_fee_bps), impact_coeff(impact_coeff) {
    
    if (num_assets <= 0) throw std::invalid_argument("Number of assets must be > 0");
    
    // Initialize equal weight portfolio
    current_weights.assign(num_assets, 1.0 / num_assets);
}

double PortfolioEngine::step(const std::vector<double>& target_weights, const std::vector<double>& asset_returns) {
    if (target_weights.size() != num_assets || asset_returns.size() != num_assets) {
        throw std::invalid_argument("Input dimensions do not match number of assets.");
    }

    double turnover = 0.0;
    double impact_turnover = 0.0;

    for (int i = 0; i < num_assets; ++i) {
        double delta = std::abs(target_weights[i] - current_weights[i]);
        turnover += delta;
        impact_turnover += std::pow(delta, 1.5);
    }

    // Calculate Friction (in dollar terms)
    double transaction_cost = turnover * (taker_fee_bps / 10000.0) * current_nav;
    double market_impact_cost = impact_turnover * impact_coeff * current_nav;
    double total_friction = transaction_cost + market_impact_cost;

    // Apply friction to NAV
    current_nav -= total_friction;

    // Calculate Portfolio Return for the step
    double portfolio_return = 0.0;
    for (int i = 0; i < num_assets; ++i) {
        portfolio_return += target_weights[i] * asset_returns[i];
    }

    // Update NAV based on market movement
    current_nav *= (1.0 + portfolio_return);
    
    // Update current weights (Assuming weights drift with returns, but for simplicity 
    // in many RL setups, we assume the target weights become the new current weights 
    // at the end of the rebalance period).
    current_weights = target_weights;

    return current_nav;
}
