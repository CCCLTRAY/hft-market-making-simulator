#ifndef MARKET_MAKER_H
#define MARKET_MAKER_H

#include <vector>

class MarketMaker {
private:
    double delta;
    double skew_factor;
    double imbalance_strength; 
    double inventory;
    double cash;
    double pnl;
    double fill_prob;
    
    bool use_obi_signal;  
    
    std::vector<double> bid_prices;
    std::vector<double> ask_prices;
    std::vector<double> inventory_history;
    std::vector<double> pnl_history;
    std::vector<double> obi_history;  
    
public:
    MarketMaker(double delta_val = 0.5, 
                double skew_val = 0.01, 
                double fill_prob_val = 0.3,
                double imbalance_strength_val = 0.0,
                bool use_obi = false);
    
    void update_quotes(double mid_price, double order_book_imbalance = 0.5);
    
    const std::vector<double>& get_obi_history() const { return obi_history; }
    
    void handle_trade(bool is_buy_trade, double price, double quantity = 1.0);
    double get_bid() const;
    double get_ask() const;
    double get_inventory() const;
    double get_pnl() const;
    double get_total_value(double current_price) const;
    
    const std::vector<double>& get_bid_history() const { return bid_prices; }
    const std::vector<double>& get_ask_history() const { return ask_prices; }
    const std::vector<double>& get_inventory_history() const { return inventory_history; }
    const std::vector<double>& get_pnl_history() const { return pnl_history; }
};

#endif