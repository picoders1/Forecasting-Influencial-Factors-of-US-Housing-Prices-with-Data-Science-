# US HOUSE PRICE KEY FACTORS - Data Science Model DEVELOPMENT 

In this project, I embark on a data-driven journey to understand the key factors influencing home prices in the United States over the past two decades. By leveraging publicly available data and advanced data science techniques, MY goal is to build a comprehensive model that sheds light on the intricate relationship between various key factors and home prices.

- [🔗 Data Cleaning](https://colab.research.google.com/drive/1YoTJzHTePaPhwS-DM2VneieLBCxyJOAR?usp=sharing): Details on data cleaning and preprocessing.
- [🔗 EDA_and_Feature_Engineering](https://colab.research.google.com/drive/1ExOaY0WC44ibqWGv7AvZlxaca0onzswg?usp=sharing): Information about data relations and distributions.
- [🔗 Model_Training](https://colab.research.google.com/drive/1VPxb8e7U6ktcoJhh0qqjrIpFhDkhvroF?usp=sharing): Model Training and Evaluation of features.


## Libraries and Tools used:

- Programming Languages: Python
- Data Analysis Libraries: NumPy, pandas, matplotlib, seaborn
- Machine Learning Libraries: scikit-learn
- Data Visualization: Matplotlib, Seaborn
- Version Control: Git, GitHub
- Google Colab for data exploration and analysis

## Data Collected

- Target (S&P/Case-Shiller U.S. National Home Price Index.)
- Population (Population includes resident population plus armed forces overseas.)
- Personal Income (Income that persons receive in return for their provision of labor, land, and capital used in
current production and the net current transfer payments that they receive from business and from government.)
- Gross Domestic Product (Featured measure of U.S. output, is the market value of the goods and services produced by labor and property located in the United States.)
- Unemployment Rate (The unemployment rate represents the number of unemployed as a percentage of the labor force. (16 years age or above))
- Mortgage Rate (A mortgage rate is the interest rate charged for a home loan. (Percentage))
- Employment- (Population Ratio (emratio))
- Building Construction issued permit in the US (Total Units)
- Labor Force Participation Rate (The participation rate is the percentage of the population that is either working or actively looking for work.)
- Monthly Supply of New Houses in the United States (The monthly supply is the ratio of new houses for sale to new houses sold.)
- Housing starts (New Housing Project) (This is a measure of the number of units of new housing projects started in a given period.)
- Median Sales Price. (Median Sales Price of Houses Sold for the United States.(US Dollers))
- Producer Price Index -Cement Manufacturing
- Producer Price Index by Industry: Concrete Brick
- All Employees, Residential Building Construction (Thousands of Peoples)
(Construction employees in the construction sector include: Working supervisors, qualified craft workers, mechanics,
apprentices, helpers, laborers, and so forth, engaged in new work, alterations, demolition, repair, maintenance etc.)
- All Employees, Construction (Thousands of persons)
(Construction employees in the construction sector include: Working supervisors, qualified craft workers, mechanics,
apprentices, helpers, laborers, and so forth, engaged in new work, alterations, demolition, repair, maintenance.)
- Industrial Production: Cement
(The industrial production (IP) index measures the real output of all relevant establishments located in the United States)
- Homeownership Rate (Percentage)
(The homeownership rate is the proportion of households that is owner-occupied.)
- Personal Saving Rate (Percent)
(Personal saving as a percentage of disposable personal income (DPI), frequently referred to as "the personal
saving rate," is calculated as the ratio of personal saving to DPI. Personal income that is used either to provide
funds to capital markets or to invest in real assets such as residences.)
- New Privately-Owned Housing Construction Completed: (Total units in thousands)
- New Privately-Owned Housing Units Under Construction: Total Units in thousands

## Feature Selection

In our analysis, we identified several key features and their correlations with the target variable, represented by the S&P Case-Shiller Home Price Index.

| Feature                | Correlation with Home Price Index |
|------------------------|-----------------------------|
| MSPUS                  | 0.981175                    |
| PPI_Cement             | 0.966502                    |
| GDP                    | 0.962712                    |
| income                 | 0.956485                    |
| PPI_Concrete           | 0.942683                    |
| population             | 0.888915                    |
| total_emp_cons         | 0.818630                    |
| new_private_hw_under   | 0.661211                    |
| all_Const_Emp          | 0.575482                    |
| home_ow_rate           | 0.221800                    |
| monthly_supply         | 0.199625                    |
| permit                 | 0.146953                    |
| house_st               | 0.006308                    |
| new_private_house      | -0.049181                   |
| unemployed_rate        | -0.279579                   |
| IPI_Cement             | -0.280124                   |
| p_saving_rate          | -0.290136                   |
| emratio                | -0.534926                   |
| mortgage_rate          | -0.690069                   |
| labor_percent          | -0.789705                   |

The positive correlation values indicate a direct relationship with home prices, while negative values suggest an inverse relationship. Features with higher absolute correlation values have a larger impact on home prices.

## Model Selection and Cross-Validation

In this project, I made use of **Lasso regression** model due to indications of significant collinearity in the dataset. The Lasso regression model is known for its ability to handle collinearity by applying L1 regularization, which encourages sparsity in feature coefficients.

To optimize the Lasso model's performance and select the best regularization hyperparameter (alpha), we utilized **cross-validation**.

- Optimal alpha: **0.0126**.

The cross-validation process resulted in the following **R-squared** scores for different folds:
- Fold 1: 0.9954
- Fold 2: 0.9963
- Fold 3: 0.9952
- Fold 4: 0.9947
- Fold 5: 0.9950
  
# predicted vs observed values
![predicted vs  observed values](https://github.com/picoders1/Forecasting-Influencial-Factors-of-US-Housing-Prices-with-Data-Science-/assets/87698874/42d9526c-3a4e-47b2-abee-4526376a5273)


- **mean R-squared** : **0.9954**
- **standard deviation R squared** : **0.0005**

## Best Features With non-zero Coefficients

Here are the features and their respective coefficients obtained from our Lasso regression model:

| Feature                | Coefficient  |
|------------------------|--------------|
| labor_percent          | 5.872566     |
| total_emp_cons         | -5.644055    |
| GDP                    | 25.443967    |
| p_saving_rate          | -0.749873    |
| PPI_Cement             | -6.022079    |
| Permit                 | -2.580253    |
| IPI_Cement             | 1.685986     |
| mortgage_rate          | -2.389226    |
| home_ow_rate           | -0.897918    |
| MSPUS                  | 29.067722    |
| PPI_Concrete           | 23.508051    |
| all_Const_Emp          | 16.703562    |
| emratio                | -1.648161    |
| monthly_supply         | 1.792269     |
| unemployed_rate        | 6.995340     |
| house_st               | 1.020058     |
| population             | -12.797367   |
| new_private_hw_under   | 7.203690     |

These coefficients represent the impact of each feature on the prediction of home prices. Positive coefficients indicate a direct relationship with home prices, while negative coefficients suggest an inverse relationship.

## Features with zero Coefficient

In our analysis, some features in our Lasso regression model had coefficients of 0.0, indicating that they do not significantly impact the prediction of home prices. Here are the features with coefficients of 0.0:

| Feature               | Coefficient  |
|-----------------------|--------------|
| new_private_house     | 0.0          |
| income                | 0.0          |

These features do not have a significant impact on the prediction of home prices in our model.

## THANK YOU!!!


