# Quantitative Analysis of Stock Market

## Overview

This project focuses on performing a quantitative analysis of the stock market using Python. The analysis includes various statistical and financial methodologies to examine and compare the performance of different stocks over a specific period.

## Tools & Technologies

- **Programming Language**: Python
- **Environment**: Jupyter Notebook
- **Libraries**:
  - **Pandas**: For data manipulation and analysis.
  - **Matplotlib**: For data visualization.
  - **Seaborn**: For statistical data visualization.

## Analysis Performed

### 1. Descriptive Statistics

This analysis provides summary statistics for each stock, including:

- **Count**: Number of trading days in the dataset.
- **Mean**: Average closing price.
- **Standard Deviation**: Measures the variation in closing prices.
- **Minimum and Maximum**: The lowest and highest closing prices.
- **Percentiles**: 25th, 50th (Median), and 75th percentiles of the closing prices.

#### Example:

- **AAPL (Apple Inc.)**: Mean = 158.24, Standard Deviation = 7.36
- **GOOG (Alphabet Inc.)**: Mean = 100.63, Standard Deviation = 6.28
- **MSFT (Microsoft Corporation)**: Mean = 275.04, Standard Deviation = 17.68
- **NFLX (Netflix Inc.)**: Mean = 327.61, Standard Deviation = 18.55

### 2. Time Series Analysis

Analyzed trends and patterns over time by focusing on the closing prices of AAPL, GOOG, MSFT, and NFLX. Key insights include:

- **Trend**: General upward trends observed in AAPL and MSFT.
- **Volatility**: NFLX exhibits more pronounced fluctuations compared to others.
- **Comparative Performance**: MSFT and NFLX generally trade at higher price levels.

### 3. Volatility Analysis

Calculated and compared the volatility (standard deviation) of closing prices for each stock. Rankings based on volatility:

- **NFLX**: Highest volatility (~18.55)
- **MSFT**: Second highest (~17.68)
- **AAPL**: Moderate volatility (~7.36)
- **GOOG**: Least volatile (~6.28)

### 4. Comparative Analysis

Compared the performance of stocks based on percentage changes in closing prices over the period:

- **MSFT**: Highest positive change (~16.10%)
- **AAPL**: Positive change (~12.23%)
- **GOOG**: Slight negative change (~-1.69%)
- **NFLX**: Significant negative change (~-11.07%)

### 5. Risk-Return Trade-off Analysis

Assessed the balance between the risk (standard deviation) and average daily return for each stock:

- **AAPL**: Lowest risk with positive average daily return.
- **GOOG**: Higher volatility with slightly negative daily return.
- **MSFT**: Moderate risk with the highest average daily return.
- **NFLX**: Highest risk with negative average daily return.

## Summary

This project demonstrates how to perform a quantitative analysis of the stock market using Python. By applying various statistical techniques, this analysis provides valuable insights into the performance, volatility, and risk-return profile of different stocks. Quantitative analysis is a powerful tool for making informed investment decisions, especially in portfolio management.

## Conclusion

Quantitative analysis in the stock market is essential for investors seeking to make data-driven decisions. This project provides a comprehensive approach to analyzing stock performance using Python, offering a foundation for more advanced financial analyses.

## Contact

Feel free to reach out if you have any questions or need further assistance.
