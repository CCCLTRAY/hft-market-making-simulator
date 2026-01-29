#include "price_simulator.h"
#include <cmath>
#include <iostream>

PriceSimulator::PriceSimulator(double init_price, double mu_val, 
                               double sigma_val, double dt_val, int steps_val)
    : initial_price(init_price), mu(mu_val), sigma(sigma_val), 
      dt(dt_val), steps(steps_val), normal_dist(0.0, 1.0) {
    
    generator.seed(std::random_device{}());
    
    std::cout << "GBM模拟器初始化: " << std::endl;
    std::cout << "  初始价格: " << initial_price << std::endl;
    std::cout << "  漂移率(μ): " << mu << " (年化" << mu*100 << "%)" << std::endl;
    std::cout << "  波动率(σ): " << sigma << " (年化" << sigma*100 << "%)" << std::endl;
    std::cout << "  时间步长(dt): " << dt << " (约" << 1.0/dt << "步/年)" << std::endl;
    std::cout << "  总步数: " << steps << " (约" << steps*dt*252 << "个交易日)" << std::endl;
}

/**
 * S_{t+Δt} = S_t * exp((μ - 0.5σ²)Δt + σ√Δt * Z)
 */
std::vector<double> PriceSimulator::generate_price_path() {
    std::vector<double> prices;
    prices.reserve(steps + 1);
    
    double current_price = initial_price;
    prices.push_back(current_price);
    
    double drift_term = (mu - 0.5 * sigma * sigma) * dt;
    double volatility_term = sigma * sqrt(dt);
    
    std::cout << "生成价格路径中..." << std::endl;
    std::cout << "  漂移项系数: " << drift_term << std::endl;
    std::cout << "  波动项系数: " << volatility_term << std::endl;
    
    for (int i = 0; i < steps; i++) {
        double Z = normal_dist(generator);

        current_price = current_price * exp(drift_term + volatility_term * Z);
        prices.push_back(current_price);
    }
    
    return prices;
}

double PriceSimulator::get_next_price(double current_price) {
    double drift_term = (mu - 0.5 * sigma * sigma) * dt;
    double volatility_term = sigma * sqrt(dt);
    double Z = normal_dist(generator);
    
    return current_price * exp(drift_term + volatility_term * Z);
}