#include "market_maker.h"
#include <iostream>
#include <cmath>

MarketMaker::MarketMaker(double delta_val, double skew_val, 
                         double fill_prob_val, double imbalance_strength_val,
                         bool use_obi)
    : delta(delta_val), skew_factor(skew_val), inventory(0.0), 
      cash(0.0), pnl(0.0), fill_prob(fill_prob_val),
      imbalance_strength(imbalance_strength_val), use_obi_signal(use_obi) {
    
    std::cout << "做市商初始化: " << std::endl;
    std::cout << "  报价价差(delta): " << delta << std::endl;
    std::cout << "  库存调整因子: " << skew_factor << std::endl;
    std::cout << "  OBI信号强度: " << imbalance_strength << std::endl;
    std::cout << "  使用OBI信号: " << (use_obi ? "是" : "否") << std::endl;
}

void MarketMaker::update_quotes(double mid_price, double order_book_imbalance) {
    if (use_obi_signal) {
        obi_history.push_back(order_book_imbalance);
    } else {
        obi_history.push_back(0.5);  
    }

    double inventory_skew = skew_factor * inventory;
    
    double obi_skew = 0.0;
    if (use_obi_signal) {
        obi_skew = imbalance_strength * (order_book_imbalance - 0.5) * mid_price * 0.01;

    }
    
    double quote_center = mid_price + obi_skew;
    
    double bid = quote_center - delta - inventory_skew;
    double ask = quote_center + delta - inventory_skew;
    
    if (bid >= ask) {
        double mid = (bid + ask) / 2.0;
        bid = mid - delta;
        ask = mid + delta;
    }
    
    bid_prices.push_back(bid);
    ask_prices.push_back(ask);
    
    inventory_history.push_back(inventory);
    pnl_history.push_back(pnl);
}

void MarketMaker::handle_trade(bool is_buy_trade, double price, double quantity) {
    if (is_buy_trade) {
        cash += price * quantity;
        inventory -= quantity;
    } else {
        cash -= price * quantity;
        inventory += quantity;
    }
    pnl = cash + inventory * price;
}

double MarketMaker::get_bid() const { 
    return bid_prices.empty() ? 0.0 : bid_prices.back(); 
}

double MarketMaker::get_ask() const { 
    return ask_prices.empty() ? 0.0 : ask_prices.back(); 
}

double MarketMaker::get_inventory() const { return inventory; }
double MarketMaker::get_pnl() const { return pnl; }

double MarketMaker::get_total_value(double current_price) const {
    return cash + inventory * current_price;
}