#ifndef ORDER_BOOK_H
#define ORDER_BOOK_H

#include <vector>
#include <random>

class OrderBookSimulator {
private:
    std::default_random_engine generator;
    double base_volume;           
    double volume_noise;          
    double imbalance_persistence; 
    double last_imbalance;        
    
public:
    OrderBookSimulator(double base = 100.0, 
                       double noise = 30.0,
                       double persistence = 0.7);
    
    double generate_imbalance(double price_change);
    
    void get_best_levels(double& bid_volume, double& ask_volume, double price_change);
    
    double calculate_obi(double price_change);
};

#endif