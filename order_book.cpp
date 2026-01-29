#include "order_book.h"
#include <cmath>
#include <iostream>

OrderBookSimulator::OrderBookSimulator(double base, double noise, double persistence)
    : base_volume(base), volume_noise(noise), 
      imbalance_persistence(persistence), last_imbalance(0.5) {
    
    std::random_device rd;
    generator.seed(rd());
}

double OrderBookSimulator::generate_imbalance(double price_change) {
    
    std::normal_distribution<double> noise_dist(0.0, 0.1);

    double signal = 0.0;
    if (price_change > 0) {
        signal = 0.3;  
    } else if (price_change < 0) {
        signal = -0.3; 
    }
    
    // 加入随机噪声
    signal += noise_dist(generator);
    
    // 应用持续性
    double imbalance = imbalance_persistence * last_imbalance + 
                      (1 - imbalance_persistence) * (0.5 + signal);
    
    // 限制在[0,1]范围内
    if (imbalance > 1.0) imbalance = 1.0;
    if (imbalance < 0.0) imbalance = 0.0;
    
    last_imbalance = imbalance;
    return imbalance;
}

void OrderBookSimulator::get_best_levels(double& bid_volume, double& ask_volume, double price_change) {
    std::normal_distribution<double> volume_dist(base_volume, volume_noise);
    
    double imbalance = generate_imbalance(price_change);

    double total_volume = volume_dist(generator) + volume_dist(generator);
    bid_volume = total_volume * imbalance;
    ask_volume = total_volume * (1.0 - imbalance);
    
    if (bid_volume < 10.0) bid_volume = 10.0;
    if (ask_volume < 10.0) ask_volume = 10.0;
}

double OrderBookSimulator::calculate_obi(double price_change) {
    double bid_volume, ask_volume;
    get_best_levels(bid_volume, ask_volume, price_change);
    
    double total_volume = bid_volume + ask_volume;
    if (total_volume < 1e-10) return 0.5; 
    
    return bid_volume / total_volume;
}