#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

class PortfolioEngine {
public:
    PortfolioEngine(int num_assets, double initial_nav, double taker_fee_bps, double impact_coeff);

    // Steps the environment forward. Returns the new NAV.
    double step(const std::vector<double>& target_weights, const std::vector<double>& asset_returns);

    double get_nav() const { return current_nav; }
    const std::vector<double>& get_weights() const { return current_weights; }

private:
    int num_assets;
    double current_nav;
    double taker_fee_bps;
    double impact_coeff;
    std::vector<double> current_weights;
};
