import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional financial charts
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def calculate_sharpe(pnl_series, risk_free_rate=0.0, freq=252):
    """Calculate Sharpe Ratio"""
    returns = pnl_series.diff().dropna()
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate/freq
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(freq)
    return sharpe

def calculate_sortino(pnl_series, risk_free_rate=0.0, freq=252):
    """Calculate Sortino Ratio (only penalizes downside volatility)"""
    returns = pnl_series.diff().dropna()
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate/freq
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(freq)
    return sortino

def calculate_max_drawdown(pnl_series):
    """Calculate Maximum Drawdown"""
    cumulative = pnl_series.values
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / (peak + 1e-10)
    return np.max(drawdown)

def calculate_calmar(pnl_series, freq=252):
    """Calculate Calmar Ratio (return / max drawdown)"""
    final_return = pnl_series.iloc[-1] - pnl_series.iloc[0]
    mdd = calculate_max_drawdown(pnl_series)
    if mdd == 0:
        return 0.0
    return final_return / mdd / (len(pnl_series)/freq)

def calculate_win_rate(pnl_series):
    """Calculate winning trade percentage"""
    returns = pnl_series.diff().dropna()
    if len(returns) == 0:
        return 0.0
    win_rate = (returns > 0).sum() / len(returns)
    return win_rate

def analyze_obi_strategy():
    """Analyze OBI-enhanced market making strategy"""
    
    print("=" * 60)
    print("OBI-ENHANCED MARKET MAKING STRATEGY ANALYSIS")
    print("=" * 60)
    
    try:
        # Read comparison data
        df_comparison = pd.read_csv('results_comparison.csv')
        df_enhanced = pd.read_csv('results_enhanced.csv')
        df_basic = pd.read_csv('results_basic.csv')
        
        print(f"Data loaded successfully:")
        print(f"  - Comparison data: {len(df_comparison)} time steps")
        print(f"  - Enhanced strategy: {len(df_enhanced)} time steps")
        print(f"  - Basic strategy: {len(df_basic)} time steps")
        
    except FileNotFoundError as e:
        print(f"Error: Required CSV files not found. {e}")
        print("Please run the C++ simulation first to generate the data files.")
        return
    
    # 1. Basic Performance Comparison
    print("\n" + "=" * 40)
    print("1. STRATEGY PERFORMANCE COMPARISON")
    print("=" * 40)
    
    basic_final_pnl = df_comparison['Basic_PnL'].iloc[-1]
    enhanced_final_pnl = df_comparison['Enhanced_PnL'].iloc[-1]
    
    print(f"Basic Strategy Final PnL: ${basic_final_pnl:.2f}")
    print(f"OBI Strategy Final PnL: ${enhanced_final_pnl:.2f}")
    print(f"PnL Improvement: ${enhanced_final_pnl - basic_final_pnl:.2f}")
    print(f"Improvement Percentage: {(enhanced_final_pnl/basic_final_pnl - 1)*100:.1f}%")
    
    # Calculate daily returns for risk metrics
    basic_daily_returns = df_comparison['Basic_PnL'].diff().dropna()
    enhanced_daily_returns = df_comparison['Enhanced_PnL'].diff().dropna()
    
    # 2. Risk-Adjusted Return Metrics
    print("\n" + "=" * 40)
    print("2. RISK-ADJUSTED PERFORMANCE METRICS")
    print("=" * 40)
    
    # Sharpe Ratio
    basic_sharpe = calculate_sharpe(df_comparison['Basic_PnL'])
    enhanced_sharpe = calculate_sharpe(df_comparison['Enhanced_PnL'])
    print(f"\nSharpe Ratio Comparison:")
    print(f"  Basic Strategy: {basic_sharpe:.3f}")
    print(f"  OBI Strategy: {enhanced_sharpe:.3f}")
    print(f"  Improvement: {enhanced_sharpe - basic_sharpe:.3f} ({((enhanced_sharpe/basic_sharpe)-1)*100:.1f}%)")
    
    # Sortino Ratio
    basic_sortino = calculate_sortino(df_comparison['Basic_PnL'])
    enhanced_sortino = calculate_sortino(df_comparison['Enhanced_PnL'])
    print(f"\nSortino Ratio (Downside Risk Adjusted):")
    print(f"  Basic Strategy: {basic_sortino:.3f}")
    print(f"  OBI Strategy: {enhanced_sortino:.3f}")
    
    # Maximum Drawdown
    basic_mdd = calculate_max_drawdown(df_comparison['Basic_PnL'])
    enhanced_mdd = calculate_max_drawdown(df_comparison['Enhanced_PnL'])
    print(f"\nMaximum Drawdown (Risk Metric):")
    print(f"  Basic Strategy: {basic_mdd:.2%}")
    print(f"  OBI Strategy: {enhanced_mdd:.2%}")
    print(f"  Risk Reduction: {basic_mdd - enhanced_mdd:.2%}")
    
    # Calmar Ratio
    basic_calmar = calculate_calmar(df_comparison['Basic_PnL'])
    enhanced_calmar = calculate_calmar(df_comparison['Enhanced_PnL'])
    print(f"\nCalmar Ratio (Return/Drawdown):")
    print(f"  Basic Strategy: {basic_calmar:.3f}")
    print(f"  OBI Strategy: {enhanced_calmar:.3f}")
    
    # Win Rate
    basic_win_rate = calculate_win_rate(df_comparison['Basic_PnL'])
    enhanced_win_rate = calculate_win_rate(df_comparison['Enhanced_PnL'])
    print(f"\nWin Rate (Percentage of Profitable Time Steps):")
    print(f"  Basic Strategy: {basic_win_rate:.1%}")
    print(f"  OBI Strategy: {enhanced_win_rate:.1%}")
    
    # 3. OBI Signal Analysis
    print("\n" + "=" * 40)
    print("3. ORDER BOOK IMBALANCE (OBI) SIGNAL ANALYSIS")
    print("=" * 40)
    
    # OBI correlation with price changes
    correlation = df_comparison['OBI'].corr(df_comparison['PriceChange'])
    print(f"OBI-Price Change Correlation: {correlation:.3f}")
    if correlation > 0.3:
        print("  → Strong positive correlation: OBI predicts price movements well")
    elif correlation > 0.1:
        print("  → Moderate positive correlation: OBI has some predictive power")
    else:
        print("  → Weak correlation: OBI may not be a strong predictor in this simulation")
    
    # OBI statistics
    print(f"\nOBI Signal Statistics:")
    print(f"  Mean: {df_comparison['OBI'].mean():.3f}")
    print(f"  Std Dev: {df_comparison['OBI'].std():.3f}")
    print(f"  Skewness: {df_comparison['OBI'].skew():.3f}")
    print(f"  Kurtosis: {df_comparison['OBI'].kurtosis():.3f}")
    
    # OBI distribution
    obi_above_50 = (df_comparison['OBI'] > 0.5).mean()
    print(f"  Percentage > 0.5 (Buying Pressure): {obi_above_50:.1%}")
    print(f"  Percentage < 0.5 (Selling Pressure): {1 - obi_above_50:.1%}")
    
    # 4. Volatility Analysis
    print("\n" + "=" * 40)
    print("4. VOLATILITY AND MARKET CONDITIONS")
    print("=" * 40)
    
    price_volatility = df_comparison['Price'].pct_change().std() * np.sqrt(252)
    print(f"Annualized Price Volatility: {price_volatility:.1%}")
    
    # Market regime analysis
    up_days = (df_comparison['PriceChange'] > 0).sum()
    down_days = (df_comparison['PriceChange'] < 0).sum()
    neutral_days = len(df_comparison) - up_days - down_days - 1  # subtract first day
    
    print(f"\nMarket Regime Analysis:")
    print(f"  Up Days: {up_days} ({up_days/len(df_comparison)*100:.1f}%)")
    print(f"  Down Days: {down_days} ({down_days/len(df_comparison)*100:.1f}%)")
    print(f"  Neutral Days: {neutral_days} ({neutral_days/len(df_comparison)*100:.1f}%)")
    
    # 5. Trading Statistics
    print("\n" + "=" * 40)
    print("5. TRADING STATISTICS")
    print("=" * 40)
    
    # Estimate number of trades (simplified)
    basic_inventory_changes = df_basic['Inventory'].diff().abs().sum()
    enhanced_inventory_changes = df_enhanced['Inventory'].diff().abs().sum()
    
    print(f"Estimated Number of Trades (via Inventory Changes):")
    print(f"  Basic Strategy: {basic_inventory_changes:.0f}")
    print(f"  OBI Strategy: {enhanced_inventory_changes:.0f}")
    print(f"  Difference: {enhanced_inventory_changes - basic_inventory_changes:.0f}")
    
    # Average profit per trade
    basic_avg_trade_pnl = basic_final_pnl / max(basic_inventory_changes, 1)
    enhanced_avg_trade_pnl = enhanced_final_pnl / max(enhanced_inventory_changes, 1)
    
    print(f"\nAverage PnL per Trade (Estimated):")
    print(f"  Basic Strategy: ${basic_avg_trade_pnl:.2f}")
    print(f"  OBI Strategy: ${enhanced_avg_trade_pnl:.2f}")
    
    # 6. Visualization
    print("\n" + "=" * 40)
    print("6. GENERATING VISUALIZATIONS")
    print("=" * 40)
    
    try:
        plot_comprehensive_results(df_comparison, df_enhanced, df_basic)
        print("✓ All visualizations generated successfully")
    except Exception as e:
        print(f"✗ Error generating visualizations: {e}")
    
    # 7. Sensitivity Analysis (if available)
    try:
        df_sensitivity = pd.read_csv('sensitivity_analysis.csv')
        if len(df_sensitivity) > 1:
            print("\n" + "=" * 40)
            print("7. SENSITIVITY ANALYSIS RESULTS")
            print("=" * 40)
            plot_sensitivity_analysis(df_sensitivity)
            print("✓ Sensitivity analysis plotted")
        else:
            print("\nNote: Sensitivity analysis data is limited or empty.")
            print("      Run full parameter scan in C++ simulation for detailed analysis.")
    except FileNotFoundError:
        print("\nNote: sensitivity_analysis.csv not found.")
        print("      Sensitivity analysis requires running parameter scan in C++ program.")
    
    # 8. Summary and Recommendations
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    print("\nKEY FINDINGS:")
    
    if enhanced_final_pnl > basic_final_pnl:
        print(f"✓ OBI strategy outperforms basic strategy by {(enhanced_final_pnl/basic_final_pnl - 1)*100:.1f}%")
    else:
        print(f"✗ OBI strategy underperforms basic strategy by {(1 - enhanced_final_pnl/basic_final_pnl)*100:.1f}%")
    
    if enhanced_sharpe > basic_sharpe:
        print(f"✓ OBI strategy has better risk-adjusted returns (Sharpe: {enhanced_sharpe:.3f} vs {basic_sharpe:.3f})")
    else:
        print(f"✗ Basic strategy has better risk-adjusted returns")
    
    if enhanced_mdd < basic_mdd:
        print(f"✓ OBI strategy has lower maximum drawdown ({enhanced_mdd:.2%} vs {basic_mdd:.2%})")
    else:
        print(f"✗ Basic strategy has lower maximum drawdown")
    
    print(f"\nOBI SIGNAL EFFECTIVENESS:")
    print(f"  Correlation with price changes: {correlation:.3f}")
    if correlation > 0.2:
        print("  → OBI shows meaningful predictive power in this simulation")
    else:
        print("  → OBI predictive power is limited in this simulation")
    
    print("\nRECOMMENDATIONS:")
    if enhanced_final_pnl > basic_final_pnl and enhanced_sharpe > basic_sharpe:
        print("1. OBI-enhanced strategy is recommended for implementation")
        print("2. Consider fine-tuning OBI signal strength parameter")
        print("3. Test strategy with different market volatility regimes")
    else:
        print("1. Basic strategy may be more suitable given current parameters")
        print("2. Consider adjusting OBI signal strength or other parameters")
        print("3. Validate strategy with more extensive backtesting")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    
    print("\nGenerated Visualizations:")
    print("  - obi_strategy_comprehensive.png (main comparison chart)")
    print("  - obi_predictive_analysis.png (OBI effectiveness)")
    print("  - obi_performance_metrics.png (key metrics)")
    print("  - obi_sensitivity_analysis.png (if data available)")

def plot_comprehensive_results(df_comparison, df_enhanced, df_basic):
    """Plot comprehensive analysis results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Price and PnL Comparison (top-left)
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(df_comparison['Price'], label='Market Price', color='blue', linewidth=2)
    ax1.set_title('Market Price Movement', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # 2. PnL Comparison (top-middle)
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(df_comparison['Basic_PnL'], label='Basic Strategy', linewidth=2, alpha=0.8)
    ax2.plot(df_comparison['Enhanced_PnL'], label='OBI Strategy', linewidth=2)
    ax2.set_title('Strategy PnL Comparison', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('PnL ($)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. OBI Signal (top-right)
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(df_comparison['OBI'], label='OBI Signal', color='purple', linewidth=2)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Neutral (0.5)')
    ax3.fill_between(df_comparison.index, 0.5, df_comparison['OBI'], 
                     where=df_comparison['OBI']>0.5, alpha=0.2, color='green', label='Buy Pressure')
    ax3.fill_between(df_comparison.index, 0.5, df_comparison['OBI'], 
                     where=df_comparison['OBI']<0.5, alpha=0.2, color='red', label='Sell Pressure')
    ax3.set_title('Order Book Imbalance (OBI) Signal', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('OBI Value')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # 4. Inventory Comparison (middle-left)
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(df_basic['Inventory'], label='Basic Strategy', linewidth=2, alpha=0.8)
    ax4.plot(df_enhanced['Inventory'], label='OBI Strategy', linewidth=2)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_title('Inventory Management', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Inventory')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # 5. Bid-Ask Spread (middle-middle)
    ax5 = plt.subplot(3, 3, 5)
    time_steps = min(200, len(df_comparison))  # Show first 200 steps for clarity
    ax5.plot(df_enhanced['Bid'][:time_steps], label='Bid Price', color='green', linewidth=1.5, alpha=0.8)
    ax5.plot(df_enhanced['Ask'][:time_steps], label='Ask Price', color='red', linewidth=1.5, alpha=0.8)
    ax5.plot(df_comparison['Price'][:time_steps], label='Market Price', color='blue', linewidth=2)
    ax5.fill_between(range(time_steps), 
                     df_enhanced['Bid'][:time_steps], 
                     df_enhanced['Ask'][:time_steps], 
                     alpha=0.2, color='gray', label='Spread')
    ax5.set_title('OBI Strategy Quotes (First 200 Steps)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Price ($)')
    ax5.grid(True, alpha=0.3)
    ax5.legend(loc='upper right')
    
    # 6. OBI vs Price Change Scatter (middle-right)
    ax6 = plt.subplot(3, 3, 6)
    scatter = ax6.scatter(df_comparison['OBI'], df_comparison['PriceChange'], 
                         c=df_comparison['TimeStep'], cmap='viridis', alpha=0.6, s=15)
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax6.axvline(x=0.5, color='red', linestyle='--', alpha=0.3)
    
    # Add regression line
    z = np.polyfit(df_comparison['OBI'], df_comparison['PriceChange'], 1)
    p = np.poly1d(z)
    ax6.plot(df_comparison['OBI'], p(df_comparison['OBI']), "r--", alpha=0.8, 
             label=f'Regression (r={df_comparison["OBI"].corr(df_comparison["PriceChange"]):.3f})')
    
    ax6.set_title('OBI vs Price Change Correlation', fontsize=12, fontweight='bold')
    ax6.set_xlabel('OBI Value')
    ax6.set_ylabel('Price Change')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    plt.colorbar(scatter, ax=ax6, label='Time Step')
    
    # 7. Return Distributions (bottom-left)
    ax7 = plt.subplot(3, 3, 7)
    basic_returns = df_comparison['Basic_PnL'].diff().dropna()
    enhanced_returns = df_comparison['Enhanced_PnL'].diff().dropna()
    
    ax7.hist(basic_returns, bins=30, alpha=0.5, label='Basic Strategy', density=True, edgecolor='black')
    ax7.hist(enhanced_returns, bins=30, alpha=0.5, label='OBI Strategy', density=True, edgecolor='black')
    ax7.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax7.set_title('Return Distributions', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Daily Return ($)')
    ax7.set_ylabel('Density')
    ax7.grid(True, alpha=0.3)
    ax7.legend()
    
    # 8. Cumulative Returns (bottom-middle)
    ax8 = plt.subplot(3, 3, 8)
    basic_cumulative = (1 + basic_returns).cumprod() - 1
    enhanced_cumulative = (1 + enhanced_returns).cumprod() - 1
    
    ax8.plot(basic_cumulative, label='Basic Strategy', linewidth=2, alpha=0.8)
    ax8.plot(enhanced_cumulative, label='OBI Strategy', linewidth=2)
    ax8.set_title('Cumulative Returns (Normalized)', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Time Step')
    ax8.set_ylabel('Cumulative Return')
    ax8.grid(True, alpha=0.3)
    ax8.legend()
    
    # 9. Performance Metrics Summary (bottom-right)
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate metrics
    metrics = ['Final PnL', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    basic_metrics = [
        df_comparison['Basic_PnL'].iloc[-1],
        calculate_sharpe(df_comparison['Basic_PnL']),
        calculate_max_drawdown(df_comparison['Basic_PnL']),
        calculate_win_rate(df_comparison['Basic_PnL'])
    ]
    enhanced_metrics = [
        df_comparison['Enhanced_PnL'].iloc[-1],
        calculate_sharpe(df_comparison['Enhanced_PnL']),
        calculate_max_drawdown(df_comparison['Enhanced_PnL']),
        calculate_win_rate(df_comparison['Enhanced_PnL'])
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax9.bar(x - width/2, basic_metrics, width, label='Basic Strategy', alpha=0.8)
    ax9.bar(x + width/2, enhanced_metrics, width, label='OBI Strategy')
    
    ax9.set_title('Performance Metrics Comparison', fontsize=12, fontweight='bold')
    ax9.set_xticks(x)
    ax9.set_xticklabels(metrics, rotation=45, ha='right')
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Format y-axis based on metric
    ax9.set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig('obi_strategy_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional specialized plots
    plot_obi_predictive_analysis(df_comparison)
    plot_performance_metrics(df_comparison)

def plot_obi_predictive_analysis(df_comparison):
    """Plot OBI signal predictive analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. OBI distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(df_comparison['OBI'], bins=30, alpha=0.7, color='teal', edgecolor='black', density=True)
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Neutral (0.5)')
    ax1.axvline(x=df_comparison['OBI'].mean(), color='blue', linestyle='-', linewidth=2, alpha=0.7, 
                label=f'Mean ({df_comparison["OBI"].mean():.3f})')
    ax1.set_title('OBI Signal Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('OBI Value')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. OBI predictive power by bucket
    ax2 = axes[0, 1]
    df_comparison['OBI_Bucket'] = pd.cut(df_comparison['OBI'], bins=10)
    bucket_stats = df_comparison.groupby('OBI_Bucket')['PriceChange'].agg(['mean', 'std', 'count'])
    
    # Calculate confidence intervals
    ci = 1.96 * bucket_stats['std'] / np.sqrt(bucket_stats['count'])
    
    x_pos = np.arange(len(bucket_stats))
    ax2.errorbar(x_pos, bucket_stats['mean'], yerr=ci, 
                 fmt='o-', capsize=5, linewidth=2, markersize=8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('OBI Predictive Power Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('OBI Bucket (Low → High)')
    ax2.set_ylabel('Average Subsequent Price Change')
    ax2.grid(True, alpha=0.3)
    
    # Add bucket labels
    bucket_labels = [f'{interval.left:.2f}-{interval.right:.2f}' for interval in bucket_stats.index]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(bucket_labels, rotation=45, ha='right')
    
    # 3. OBI autocorrelation
    ax3 = axes[1, 0]
    max_lag = min(50, len(df_comparison) // 2)
    obi_autocorr = [df_comparison['OBI'].autocorr(lag) for lag in range(1, max_lag + 1)]
    
    ax3.bar(range(1, max_lag + 1), obi_autocorr, alpha=0.7, color='orange', edgecolor='black')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.set_title('OBI Signal Autocorrelation', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Lag (Time Steps)')
    ax3.set_ylabel('Autocorrelation')
    ax3.grid(True, alpha=0.3)
    
    # 4. OBI vs strategy performance
    ax4 = axes[1, 1]
    
    # Create OBI-based performance analysis
    df_comparison['OBI_Level'] = pd.qcut(df_comparison['OBI'], q=4, labels=['Very Low', 'Low', 'High', 'Very High'])
    
    # Calculate strategy performance by OBI level
    performance_by_obi = df_comparison.groupby('OBI_Level').agg({
        'Enhanced_PnL': 'last',
        'Basic_PnL': 'last',
        'PriceChange': 'mean'
    })
    
    x = np.arange(len(performance_by_obi))
    width = 0.35
    
    ax4.bar(x - width/2, performance_by_obi['Basic_PnL'], width, label='Basic Strategy', alpha=0.8)
    ax4.bar(x + width/2, performance_by_obi['Enhanced_PnL'], width, label='OBI Strategy')
    
    ax4.set_title('Strategy Performance by OBI Level', fontsize=12, fontweight='bold')
    ax4.set_xlabel('OBI Level')
    ax4.set_ylabel('Final PnL')
    ax4.set_xticks(x)
    ax4.set_xticklabels(performance_by_obi.index)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('obi_predictive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(df_comparison):
    """Plot detailed performance metrics comparison"""
    
    # Calculate comprehensive metrics
    metrics_data = {
        'Metric': [],
        'Basic Strategy': [],
        'OBI Strategy': [],
        'Improvement': []
    }
    
    # Final PnL
    basic_pnl = df_comparison['Basic_PnL'].iloc[-1]
    enhanced_pnl = df_comparison['Enhanced_PnL'].iloc[-1]
    metrics_data['Metric'].append('Final PnL ($)')
    metrics_data['Basic Strategy'].append(basic_pnl)
    metrics_data['OBI Strategy'].append(enhanced_pnl)
    metrics_data['Improvement'].append(enhanced_pnl - basic_pnl)
    
    # Sharpe Ratio
    basic_sharpe = calculate_sharpe(df_comparison['Basic_PnL'])
    enhanced_sharpe = calculate_sharpe(df_comparison['Enhanced_PnL'])
    metrics_data['Metric'].append('Sharpe Ratio')
    metrics_data['Basic Strategy'].append(basic_sharpe)
    metrics_data['OBI Strategy'].append(enhanced_sharpe)
    metrics_data['Improvement'].append(enhanced_sharpe - basic_sharpe)
    
    # Sortino Ratio
    basic_sortino = calculate_sortino(df_comparison['Basic_PnL'])
    enhanced_sortino = calculate_sortino(df_comparison['Enhanced_PnL'])
    metrics_data['Metric'].append('Sortino Ratio')
    metrics_data['Basic Strategy'].append(basic_sortino)
    metrics_data['OBI Strategy'].append(enhanced_sortino)
    metrics_data['Improvement'].append(enhanced_sortino - basic_sortino)
    
    # Max Drawdown
    basic_mdd = calculate_max_drawdown(df_comparison['Basic_PnL'])
    enhanced_mdd = calculate_max_drawdown(df_comparison['Enhanced_PnL'])
    metrics_data['Metric'].append('Max Drawdown (%)')
    metrics_data['Basic Strategy'].append(basic_mdd * 100)
    metrics_data['OBI Strategy'].append(enhanced_mdd * 100)
    metrics_data['Improvement'].append((enhanced_mdd - basic_mdd) * 100)
    
    # Win Rate
    basic_win = calculate_win_rate(df_comparison['Basic_PnL'])
    enhanced_win = calculate_win_rate(df_comparison['Enhanced_PnL'])
    metrics_data['Metric'].append('Win Rate (%)')
    metrics_data['Basic Strategy'].append(basic_win * 100)
    metrics_data['OBI Strategy'].append(enhanced_win * 100)
    metrics_data['Improvement'].append((enhanced_win - basic_win) * 100)
    
    # Calmar Ratio
    basic_calmar = calculate_calmar(df_comparison['Basic_PnL'])
    enhanced_calmar = calculate_calmar(df_comparison['Enhanced_PnL'])
    metrics_data['Metric'].append('Calmar Ratio')
    metrics_data['Basic Strategy'].append(basic_calmar)
    metrics_data['OBI Strategy'].append(enhanced_calmar)
    metrics_data['Improvement'].append(enhanced_calmar - basic_calmar)
    
    # Create DataFrame
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Bar chart of key metrics
    ax1 = axes[0]
    metrics_to_plot = ['Sharpe Ratio', 'Sortino Ratio', 'Win Rate (%)', 'Calmar Ratio']
    df_subset = df_metrics[df_metrics['Metric'].isin(metrics_to_plot)]
    
    x = np.arange(len(df_subset))
    width = 0.35
    
    ax1.bar(x - width/2, df_subset['Basic Strategy'], width, label='Basic Strategy', alpha=0.8)
    ax1.bar(x + width/2, df_subset['OBI Strategy'], width, label='OBI Strategy')
    
    ax1.set_title('Risk-Adjusted Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_subset['Metric'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Improvement visualization
    ax2 = axes[1]
    
    # Prepare improvement data (skip metrics where lower is better)
    improvement_data = df_metrics.copy()
    # For drawdown, negative improvement is better
    improvement_data.loc[improvement_data['Metric'] == 'Max Drawdown (%)', 'Improvement'] *= -1
    
    # Plot improvements
    colors = ['green' if x > 0 else 'red' for x in improvement_data['Improvement']]
    ax2.barh(improvement_data['Metric'], improvement_data['Improvement'], color=colors, alpha=0.7)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('OBI Strategy Improvement vs Basic', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Improvement (OBI - Basic)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(improvement_data['Improvement']):
        ax2.text(v, i, f'{v:+.2f}', va='center', 
                ha='left' if v >= 0 else 'right', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('obi_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print metrics table
    print("\n" + "=" * 80)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 80)
    print(df_metrics.to_string(index=False))
    
    return df_metrics

def plot_sensitivity_analysis(df_sensitivity):
    """Plot sensitivity analysis results"""
    
    if len(df_sensitivity) <= 1:
        print("Insufficient sensitivity analysis data to plot.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. PnL vs Signal Strength
    ax1 = axes[0, 0]
    ax1.plot(df_sensitivity['Imbalance_Strength'], df_sensitivity['Final_PnL'], 
             'o-', linewidth=2, markersize=8)
    # Highlight maximum
    max_idx = df_sensitivity['Final_PnL'].idxmax()
    ax1.plot(df_sensitivity.loc[max_idx, 'Imbalance_Strength'], 
             df_sensitivity.loc[max_idx, 'Final_PnL'], 
             'o', markersize=12, color='red', label=f'Max PnL at {df_sensitivity.loc[max_idx, "Imbalance_Strength"]}')
    
    ax1.set_title('OBI Signal Strength vs Final PnL', fontsize=12, fontweight='bold')
    ax1.set_xlabel('OBI Signal Strength Parameter')
    ax1.set_ylabel('Final PnL ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Max Inventory vs Signal Strength
    ax2 = axes[0, 1]
    ax2.plot(df_sensitivity['Imbalance_Strength'], df_sensitivity['Max_Inventory'], 
             's-', linewidth=2, markersize=8, color='orange')
    ax2.set_title('Signal Strength vs Maximum Inventory', fontsize=12, fontweight='bold')
    ax2.set_xlabel('OBI Signal Strength Parameter')
    ax2.set_ylabel('Maximum Inventory')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sharpe Ratio vs Signal Strength
    ax3 = axes[1, 0]
    if 'Sharpe_Ratio' in df_sensitivity.columns:
        ax3.plot(df_sensitivity['Imbalance_Strength'], df_sensitivity['Sharpe_Ratio'], 
                 '^-', linewidth=2, markersize=8, color='green')
        # Highlight maximum
        max_sharpe_idx = df_sensitivity['Sharpe_Ratio'].idxmax()
        ax3.plot(df_sensitivity.loc[max_sharpe_idx, 'Imbalance_Strength'], 
                 df_sensitivity.loc[max_sharpe_idx, 'Sharpe_Ratio'], 
                 '^', markersize=12, color='red', 
                 label=f'Max Sharpe at {df_sensitivity.loc[max_sharpe_idx, "Imbalance_Strength"]}')
        ax3.legend()
        ax3.set_title('Signal Strength vs Sharpe Ratio', fontsize=12, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Sharpe Ratio data not available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Sharpe Ratio Data Missing', fontsize=12, fontweight='bold')
    ax3.set_xlabel('OBI Signal Strength Parameter')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.grid(True, alpha=0.3)
    
    # 4. Max Drawdown vs Signal Strength
    ax4 = axes[1, 1]
    if 'Max_Drawdown' in df_sensitivity.columns:
        ax4.plot(df_sensitivity['Imbalance_Strength'], df_sensitivity['Max_Drawdown'], 
                 'v-', linewidth=2, markersize=8, color='red')
        # Highlight minimum (lower drawdown is better)
        min_dd_idx = df_sensitivity['Max_Drawdown'].idxmin()
        ax4.plot(df_sensitivity.loc[min_dd_idx, 'Imbalance_Strength'], 
                 df_sensitivity.loc[min_dd_idx, 'Max_Drawdown'], 
                 'v', markersize=12, color='green', 
                 label=f'Min Drawdown at {df_sensitivity.loc[min_dd_idx, "Imbalance_Strength"]}')
        ax4.legend()
        ax4.set_title('Signal Strength vs Maximum Drawdown', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Max Drawdown data not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Max Drawdown Data Missing', fontsize=12, fontweight='bold')
    ax4.set_xlabel('OBI Signal Strength Parameter')
    ax4.set_ylabel('Maximum Drawdown')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('obi_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print optimal parameter analysis
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS - OPTIMAL PARAMETER FINDINGS")
    print("=" * 60)
    
    if 'Final_PnL' in df_sensitivity.columns:
        optimal_pnl_idx = df_sensitivity['Final_PnL'].idxmax()
        print(f"\nMaximum PnL achieved at:")
        print(f"  Signal Strength: {df_sensitivity.loc[optimal_pnl_idx, 'Imbalance_Strength']}")
        print(f"  Final PnL: ${df_sensitivity.loc[optimal_pnl_idx, 'Final_PnL']:.2f}")
    
    if 'Sharpe_Ratio' in df_sensitivity.columns:
        optimal_sharpe_idx = df_sensitivity['Sharpe_Ratio'].idxmax()
        print(f"\nMaximum Sharpe Ratio achieved at:")
        print(f"  Signal Strength: {df_sensitivity.loc[optimal_sharpe_idx, 'Imbalance_Strength']}")
        print(f"  Sharpe Ratio: {df_sensitivity.loc[optimal_sharpe_idx, 'Sharpe_Ratio']:.3f}")
    
    print("\nKey Insight:")
    print("The optimal signal strength balances predictive power with risk management.")
    print("Too low: Not utilizing OBI information effectively")
    print("Too high: Overreacting to noise, increasing transaction costs and risk")

def explain_obi_concept():
    """Explain OBI concept in detail"""
    
    print("\n" + "=" * 80)
    print("ORDER BOOK IMBALANCE (OBI) - CONCEPT EXPLANATION")
    print("=" * 80)
    
    print("\nWHAT IS ORDER BOOK IMBALANCE (OBI)?")
    print("-" * 40)
    print("OBI = Volume_Bid / (Volume_Bid + Volume_Ask)")
    print("\nWhere:")
    print("  • Volume_Bid: Total order volume on the bid side (buy orders)")
    print("  • Volume_Ask: Total order volume on the ask side (sell orders)")
    print("\nThe OBI ranges from 0 to 1:")
    print("  • OBI = 0.5: Perfect balance between buyers and sellers")
    print("  • OBI > 0.5: More buying pressure (bid volume > ask volume)")
    print("  • OBI < 0.5: More selling pressure (ask volume > bid volume)")
    
    print("\nFINANCIAL INTERPRETATION")
    print("-" * 40)
    print("OBI is a leading indicator of short-term price direction because:")
    print("1. Order book reflects real-time supply and demand")
    print("2. Large bid volume indicates strong buying interest")
    print("3. When bids significantly outweigh asks, price tends to rise")
    print("4. The imbalance often precedes actual trades and price movements")
    
    print("\nACADEMIC & INDUSTRY RELEVANCE")
    print("-" * 40)
    print("• Cited in high-frequency trading literature as predictive signal")
    print("• Used by institutional market makers to adjust quotes")
    print("• Empirical studies show correlation with 1-10 second price moves")
    print("• Particularly effective in liquid markets with continuous trading")
    
    print("\nSTRATEGY IMPLEMENTATION IN THIS PROJECT")
    print("-" * 40)
    print("Our OBI-enhanced market making strategy:")
    print("1. Calculates OBI from simulated order book data")
    print("2. Adjusts quote center based on OBI signal:")
    print("   • OBI > 0.5 (bullish): Raise quote center, be more willing to sell")
    print("   • OBI < 0.5 (bearish): Lower quote center, be more willing to buy")
    print("3. Combines OBI signal with inventory management")
    print("4. Tests different OBI signal strengths for optimization")
    
    print("\nEXPECTED BENEFITS")
    print("-" * 40)
    print("✓ Better prediction of short-term price movements")
    print("✓ Asymmetric quoting: Wider spreads on 'wrong' side, tighter on 'right' side")
    print("✓ Improved risk management through directional bias")
    print("✓ Higher information ratio compared to symmetric market making")
    
    print("\nLIMITATIONS AND CONSIDERATIONS")
    print("-" * 40)
    print("• OBI signals can be noisy and short-lived")
    print("• Optimal signal strength depends on market conditions")
    print("• May increase adverse selection if signal is weak")
    print("• Requires careful calibration with inventory management")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("OBI-ENHANCED MARKET MAKING STRATEGY ANALYZER")
    print("Version 2.0 | Professional Quantitative Analysis")
    print("=" * 80)
    
    # Explain OBI concept (optional - can be skipped for faster analysis)
    show_concept = input("\nShow OBI concept explanation? (y/n): ").lower()
    if show_concept == 'y':
        explain_obi_concept()
    
    # Run comprehensive analysis
    analyze_obi_strategy()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutput Files Generated:")
    print("  • obi_strategy_comprehensive.png - Main comparison chart")
    print("  • obi_predictive_analysis.png    - OBI effectiveness analysis")
    print("  • obi_performance_metrics.png    - Key metrics comparison")
    print("  • obi_sensitivity_analysis.png   - Parameter sensitivity (if data available)")
    print("\nNext Steps:")
    print("  1. Review generated charts and metrics")
    print("  2. Consider adjusting OBI signal strength parameter")
    print("  3. Test with different market volatility regimes")
    print("  4. Extend analysis with more sophisticated OBI calculations")