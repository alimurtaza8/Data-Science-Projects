# California House Price Prediction Project

## Objective
The objective of this project was to build a machine learning model to predict house prices across various districts in California. The model uses several features such as location, house size, and proximity to the ocean to make accurate predictions.

## 1. Data Source and Key Features
The data was sourced from the California housing dataset. Key features include:

- **Location:** Longitude and Latitude of each district.
- **House Characteristics:** Total rooms, total bedrooms, population, households, and the median age of houses.
- **Economic Information:** Median income of the residents.
- **Proximity to the Ocean:** Whether a house is located near the ocean.

This data helped us analyze trends and correlations to better understand how they affect house prices.

## 2. Data Cleaning and Preparation
We prepared the data for modeling by:

- Handling missing values (e.g., filling missing bedroom or population numbers).
- Removing outliers to avoid skewed predictions.
- Normalizing features like total rooms and bedrooms for better comparability across districts.

## 3. Exploratory Data Analysis (EDA)
Key insights from our data analysis include:

- **Higher income correlates with higher house prices:** Districts with higher median income tend to have more expensive houses.
- **Location is crucial:** Proximity to the ocean is a major factor in determining house prices, with coastal properties generally being more expensive.
- **Total rooms and house age** also have a correlation with house prices, but not as strong as income or location.

### Note: The **`median_income`** is a great predictor of **`median house value`**.

## 4. Modeling and Performance

We trained the machine learning model using the **Random Forest Regressor**:

- **Model Choice:** Random Forest was chosen for its ability to handle non-linear relationships and variable interactions.
- **Training and Testing:** The dataset was split into training (80%) and testing (20%) sets for evaluation.

### Model Evaluation:
- **Root Mean Squared Error (RMSE):** The model's RMSE was **X**, indicating how far the predicted prices were from the actual prices.
- **R-Squared Score:** The model achieved an R-squared value of **Y**, explaining **Y%** of the variability in house prices.

## 5. Predictions and Insights

Key takeaways from our model's predictions:

- **High-income districts** had significantly higher predicted house prices.
- **Ocean proximity** led to consistently higher predicted house values.
- **Affordable housing** options were mostly found in districts farther from the ocean, with lower income levels.

## 6. Visualizations
We included the following visualizations to support our predictions:

- **Correlation Heatmap:** Highlights relationships between features like median income, number of rooms, and house prices.
- **Price Distribution Map:** A geographic map showing how house prices vary across different districts in California.
- **Predicted vs. Actual Prices Chart:** Compares our model's predictions with actual prices, showing a close alignment.

## 7. Conclusion
Our Random Forest model was effective in predicting house prices using key features from the dataset. This model can guide real estate investments, pricing strategies, and affordability analyses.

## Next Steps
- **Feature Addition:** Incorporate additional features such as crime rates, school ratings, and proximity to amenities (parks, public transport, etc.) to enhance predictions.
- **Advanced Models:** Explore more advanced models like **Gradient Boosting** or **XGBoost** for improved accuracy.
