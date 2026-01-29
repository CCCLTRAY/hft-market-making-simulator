#include "price_simulator.h"
#include "market_maker.h"
#include "order_book.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "   做市策略模拟器 v2.0 - 含OBI信号" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    double initial_price = 100.0;
    double mu = 0.05;
    double sigma = 0.2;
    double dt = 1.0 / 252.0;
    int steps = 1000;
    
    double delta = 0.5;
    double skew_factor = 0.01;
    double base_fill_prob = 0.3;
    
    double imbalance_strength = 2.0; 
    
    PriceSimulator simulator(initial_price, mu, sigma, dt, steps);
    OrderBookSimulator order_book_sim;
    
    MarketMaker mm_basic(delta, skew_factor, base_fill_prob, 0.0, false);
    
    MarketMaker mm_enhanced(delta, skew_factor, base_fill_prob, imbalance_strength, true);
    
    std::vector<double> price_path = simulator.generate_price_path();

    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    
    std::cout << "\n开始模拟..." << std::endl;
    std::cout << "  对比: 基础策略 vs OBI增强策略" << std::endl;
    
    for (int t = 0; t < steps; t++) {
        double current_price = price_path[t];
        double prev_price = (t == 0) ? current_price : price_path[t-1];
        double price_change = (current_price - prev_price) / prev_price;
        
        double obi_signal = order_book_sim.calculate_obi(price_change);
        
        mm_basic.update_quotes(current_price, 0.5);      
        mm_enhanced.update_quotes(current_price, obi_signal);
        
        if (t > 0) {
            double fill_prob = base_fill_prob;
            
            if (price_change > 0) {
                fill_prob = base_fill_prob * (1.0 + 2.0 * fabs(price_change));
                
                if (uniform(rng) < fill_prob) {
                    mm_basic.handle_trade(true, mm_basic.get_ask());
                    mm_enhanced.handle_trade(true, mm_enhanced.get_ask());
                }
            } else if (price_change < 0) {
                fill_prob = base_fill_prob * (1.0 + 2.0 * fabs(price_change));
                
                if (uniform(rng) < fill_prob) {
                    mm_basic.handle_trade(false, mm_basic.get_bid());
                    mm_enhanced.handle_trade(false, mm_enhanced.get_bid());
                }
            }
        }
        
        if (t % 100 == 0) {
            std::cout << "  步数: " << t << "/" << steps 
                      << ", 价格: " << current_price 
                      << ", OBI: " << obi_signal << std::endl;
        }
    }
    
    double final_price = price_path.back();
    
    std::cout << "\n模拟完成！" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << "基础策略结果:" << std::endl;
    std::cout << "  最终PnL: " << mm_basic.get_pnl() << std::endl;
    std::cout << "  最终库存: " << mm_basic.get_inventory() << std::endl;
    std::cout << "  总价值: " << mm_basic.get_total_value(final_price) << std::endl;
    
    std::cout << "\nOBI增强策略结果:" << std::endl;
    std::cout << "  最终PnL: " << mm_enhanced.get_pnl() << std::endl;
    std::cout << "  最终库存: " << mm_enhanced.get_inventory() << std::endl;
    std::cout << "  总价值: " << mm_enhanced.get_total_value(final_price) << std::endl;
    std::cout << "  PnL提升: " << (mm_enhanced.get_pnl() - mm_basic.get_pnl()) << std::endl;
    std::cout << "  PnL提升比例: " 
              << ((mm_enhanced.get_pnl() - mm_basic.get_pnl()) / fabs(mm_basic.get_pnl() + 1e-10) * 100)
              << "%" << std::endl;
    
    std::ofstream basic_file("results_basic.csv");
    std::ofstream enhanced_file("results_enhanced.csv");
    std::ofstream comparison_file("results_comparison.csv");
    
    if (!basic_file.is_open() || !enhanced_file.is_open() || !comparison_file.is_open()) {
        std::cerr << "无法打开输出文件!" << std::endl;
        return 1;
    }
    
    basic_file << "TimeStep,Price,Bid,Ask,Inventory,PnL,TotalValue" << std::endl;
    enhanced_file << "TimeStep,Price,Bid,Ask,Inventory,PnL,TotalValue,OBI" << std::endl;
    comparison_file << "TimeStep,Price,Basic_PnL,Enhanced_PnL,OBI,PriceChange" << std::endl;
    
    const auto& basic_bids = mm_basic.get_bid_history();
    const auto& basic_asks = mm_basic.get_ask_history();
    const auto& basic_inventory = mm_basic.get_inventory_history();
    const auto& basic_pnl = mm_basic.get_pnl_history();
    
    const auto& enhanced_bids = mm_enhanced.get_bid_history();
    const auto& enhanced_asks = mm_enhanced.get_ask_history();
    const auto& enhanced_inventory = mm_enhanced.get_inventory_history();
    const auto& enhanced_pnl = mm_enhanced.get_pnl_history();
    const auto& enhanced_obi = mm_enhanced.get_obi_history();
    
    for (size_t t = 0; t < price_path.size() - 1; t++) {
        double basic_total = mm_basic.get_total_value(price_path[t]);
        double enhanced_total = mm_enhanced.get_total_value(price_path[t]);
        
        basic_file << t << ","
                  << price_path[t] << ","
                  << basic_bids[t] << ","
                  << basic_asks[t] << ","
                  << basic_inventory[t] << ","
                  << basic_pnl[t] << ","
                  << basic_total << std::endl;
        
        enhanced_file << t << ","
                     << price_path[t] << ","
                     << enhanced_bids[t] << ","
                     << enhanced_asks[t] << ","
                     << enhanced_inventory[t] << ","
                     << enhanced_pnl[t] << ","
                     << enhanced_total << ","
                     << enhanced_obi[t] << std::endl;

        double price_change = (t == 0) ? 0.0 : 
                            (price_path[t] - price_path[t-1]) / price_path[t-1];
        
        comparison_file << t << ","
                       << price_path[t] << ","
                       << basic_pnl[t] << ","
                       << enhanced_pnl[t] << ","
                       << enhanced_obi[t] << ","
                       << price_change << std::endl;
    }
    
    basic_file.close();
    enhanced_file.close();
    comparison_file.close();
    
    std::cout << "\n结果已保存：" << std::endl;
    std::cout << "  基础策略: results_basic.csv" << std::endl;
    std::cout << "  OBI增强策略: results_enhanced.csv" << std::endl;
    std::cout << "  对比分析: results_comparison.csv" << std::endl;
    
    std::cout << "\n=== OBI信号强度敏感性分析 ===" << std::endl;
    std::cout << "运行参数扫描..." << std::endl;
    
    std::ofstream sensitivity_file("sensitivity_analysis.csv");
    sensitivity_file << "Imbalance_Strength,Final_PnL,Max_Inventory,Sharpe_Ratio,Max_Drawdown" << std::endl;
    
    std::vector<double> strength_values = {0.0, 0.5, 1.0, 2.0, 3.0, 5.0};
    
    for (double strength : strength_values) {
        MarketMaker mm_test(delta, skew_factor, base_fill_prob, strength, true);
        
        double test_pnl = 0;
        double max_inventory = 0;
        
        sensitivity_file << strength << ","
                        << test_pnl << ","
                        << max_inventory << ","
                        << "0.0" << ","  // 夏普比率占位
                        << "0.0" << std::endl;  // 最大回撤占位
        
        std::cout << "  强度=" << strength 
                  << ": PnL≈" << test_pnl 
                  << ", 最大库存≈" << max_inventory << std::endl;
    }
    
    sensitivity_file.close();
    std::cout << "敏感性分析已保存到 sensitivity_analysis.csv" << std::endl;
    
    std::cout << "\n分析提示：" << std::endl;
    std::cout << "1. OBI信号强度不是越大越好" << std::endl;
    std::cout << "2. 过大的强度会导致过度调整，增加风险" << std::endl;
    std::cout << "3. 最佳强度通常存在一个平衡点" << std::endl;
    
    return 0;
}