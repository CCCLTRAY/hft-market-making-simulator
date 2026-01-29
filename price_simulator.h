#ifndef PRICE_SIMULATOR_H
#define PRICE_SIMULATOR_H

#include <vector>
#include <random>

class PriceSimulator {
private:
    double initial_price;   // 初始价格
    double mu;              // 漂移率（预期收益）
    double sigma;           // 波动率
    double dt;              // 时间步长
    int steps;              // 总步数
    
    std::default_random_engine generator;
    std::normal_distribution<double> normal_dist;
    
public:

    PriceSimulator(double init_price = 100.0, 
                   double mu_val = 0.0, 
                   double sigma_val = 0.2, 
                   double dt_val = 1.0/252.0,  
                   int steps_val = 1000);
    
    // 生成价格路径
    std::vector<double> generate_price_path();
    
    // 获取单个下一时刻价格
    double get_next_price(double current_price);
};

#endif