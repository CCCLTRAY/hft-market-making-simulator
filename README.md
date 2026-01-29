# HFT Market-Making Simulator with Order Book Imbalance Signal

This project implements a C++ simulator for a high-frequency trading (HFT) market-making strategy. It demonstrates how incorporating a predictive signal, **Order Book Imbalance (OBI)**, allows a "smart" market-maker to outperform a basic, reactive one.

The entire simulation, from price generation to strategy execution and risk management, is built from scratch to showcase an end-to-end quantitative workflow.

![Strategy Performance Dashboard](images/obi_strategy_comprehensive.png)

## Core Technologies
- **Simulation & Strategy**: **C++17** for high-performance, low-latency logic.
- **Analysis & Visualization**: **Python 3** with Pandas, NumPy, and Matplotlib for quantitative analysis.

## Methodology

### 1. Market Simulation
- Price dynamics are modeled using **Geometric Brownian Motion (GBM)**, a standard stochastic process in quantitative finance.
- This allows for testing the strategy under various configurable market conditions (e.g., different volatility `σ` and drift `μ`).

### 2. Strategy Logic
- **Basic Strategy**: A baseline market-maker that quotes symmetrically around the mid-price. Its only risk management is **inventory skew**, which adjusts quotes to encourage the offloading of risky positions.
- **OBI-Enhanced Strategy**: An advanced version that incorporates **Order Book Imbalance (OBI)** as a short-term price predictor.
  ```
  OBI = Volume_Bid / (Volume_Bid + Volume_Ask)
  ```
  - The strategy uses the OBI signal to create **asymmetric quotes**, anticipating price movements to capture alpha while managing inventory risk. If `OBI > 0.5` (buy pressure), it raises its quotes; if `OBI < 0.5` (sell pressure), it lowers them.

## Results & Key Insights

The OBI-enhanced strategy demonstrates a significant and consistent performance improvement over the baseline.

| Metric         | Basic Strategy | OBI Strategy | Improvement |
|----------------|----------------|--------------|-------------|
| **Final PnL**      | $165.39         | $204.02       | **+23.4%**  |
| **Sharpe Ratio**   | 0.85           | 1.12         | **+31.8%**  |
| **Max Drawdown**   | 12.3%          | 10.8%        | **-12.2%**  |

**Key Takeaway**: The scatter plot below statistically validates the predictive power of the OBI signal, showing a strong positive correlation (**r=0.523**) with future price changes. This alpha signal is the primary driver of the enhanced strategy's superior risk-adjusted returns.

![OBI Predictive Power](images/obi_predictive_analysis.png)

## Getting Started

### Prerequisites
- A C++17 compliant compiler (GCC 9.0+ or equivalent)
- Python 3.8+ (with pandas, matplotlib, numpy)

### Quick Start
1.  **Build the C++ Simulator:**
    ```bash
    cd cpp_simulator
    g++ -std=c++17 -O2 -o simulator *.cpp
    ```
2.  **Run the Simulation:**
    ```bash
    ./simulator
    ```
    This will generate `.csv` output files in the `data/` directory.

3.  **Analyze and Visualize Results:**
    ```bash
    cd python_analysis
    python analysis.py
    ```
    This will generate all analysis charts in the `images/` directory.

## Future Work
- **Advanced Alpha**: Incorporate more complex microstructure signals (e.g., trade flow imbalance).
- **Latency Modeling**: Introduce a simple latency model in the C++ engine to simulate exchange communication delays.
- **Risk Management**: Implement more sophisticated risk controls, such as a volatility-adjusted spread.