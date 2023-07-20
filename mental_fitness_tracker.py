# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Reading CSV Data into Pandas DataFrames 
df1 = pd.read_csv('mental-and-substance-use-as-share-of-disease.csv')
df2 = pd.read_csv('prevalence-by-mental-and-substance-use-disorder.csv')


# Filling Missing Values
numeric_columns = df1.select_dtypes(include=[np.number]).columns
df1[numeric_columns] = df1[numeric_columns].fillna(df1[numeric_columns].mean())

numeric_columns = df2.select_dtypes(include=[np.number]).columns
df2[numeric_columns] = df2[numeric_columns].fillna(df2[numeric_columns].mean())


# Convert specific columns to the appropriate data types
df1['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'] = df1['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)'].astype(float)
df2['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
# Repeat this line for other columns that need conversion
df2['Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
df2['Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
df2['Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
df2['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
df2['Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)
df2['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'] = df2['Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)'].astype(float)


merged_df = pd.merge(df1, df2, on=['Entity', 'Code', 'Year'])


X = merged_df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',
               'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)']]

y = merged_df['DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# Fit regression models
ridge_model = Ridge(alpha=0.5)
ridge_model.fit(X_train, y_train)
ridge_y_pred = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_y_pred)
ridge_r2 = r2_score(y_test, ridge_y_pred)

lasso_model = Lasso(alpha=0.5)
lasso_model.fit(X_train, y_train)
lasso_y_pred = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_y_pred)
lasso_r2 = r2_score(y_test, lasso_y_pred)

elastic_net_model = ElasticNet(alpha=0.5, l1_ratio=0.5)
elastic_net_model.fit(X_train, y_train)
elastic_net_y_pred = elastic_net_model.predict(X_test)
elastic_net_mse = mean_squared_error(y_test, elastic_net_y_pred)
elastic_net_r2 = r2_score(y_test, elastic_net_y_pred)

poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X_train)
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)
X_test_poly = poly_features.transform(X_test)
poly_y_pred = poly_model.predict(X_test_poly)
poly_mse = mean_squared_error(y_test, poly_y_pred)
poly_r2 = r2_score(y_test, poly_y_pred)

tree_model = DecisionTreeRegressor()
tree_model.fit(X_train, y_train)
tree_y_pred = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_y_pred)
tree_r2 = r2_score(y_test, tree_y_pred)

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
forest_y_pred = forest_model.predict(X_test)
forest_mse = mean_squared_error(y_test, forest_y_pred)
forest_r2 = r2_score(y_test, forest_y_pred)

svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_y_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_y_pred)
svr_r2 = r2_score(y_test, svr_y_pred)

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_y_pred = xgb_model.predict(X_test)
xgb_mse = mean_squared_error(y_test, xgb_y_pred)
xgb_r2 = r2_score(y_test, xgb_y_pred)

knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_y_pred = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_y_pred)
knn_r2 = r2_score(y_test, knn_y_pred)

bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
bayesian_y_pred = bayesian_model.predict(X_test)
bayesian_mse = mean_squared_error(y_test, bayesian_y_pred)
bayesian_r2 = r2_score(y_test, bayesian_y_pred)

nn_model = MLPRegressor(max_iter=1000)
nn_model.fit(X_train, y_train)
nn_y_pred = nn_model.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_y_pred)
nn_r2 = r2_score(y_test, nn_y_pred)

gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_y_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_y_pred)
gb_r2 = r2_score(y_test, gb_y_pred)

# Create a dictionary to store the model performance
model_performance = {
    'Ridge Regression': {'MSE': ridge_mse, 'R-squared': ridge_r2},
    'Lasso Regression': {'MSE': lasso_mse, 'R-squared': lasso_r2},
    'Elastic Net Regression': {'MSE': elastic_net_mse, 'R-squared': elastic_net_r2},
    'Polynomial Regression': {'MSE': poly_mse, 'R-squared': poly_r2},
    'Decision Tree Regression': {'MSE': tree_mse, 'R-squared': tree_r2},
    'Random Forest Regression': {'MSE': forest_mse, 'R-squared': forest_r2},
    'Support Vector Regression': {'MSE': svr_mse, 'R-squared': svr_r2},
    'XGBoost Regression': {'MSE': xgb_mse, 'R-squared': xgb_r2},
    'K-Nearest Neighbors Regression': {'MSE': knn_mse, 'R-squared': knn_r2},
    'Bayesian Regression': {'MSE': bayesian_mse, 'R-squared': bayesian_r2},
    'Neural Network Regression': {'MSE': nn_mse, 'R-squared': nn_r2},
    'Gradient Boosting Regression': {'MSE': gb_mse, 'R-squared': gb_r2}
}

# Print model performance
sorted_models = sorted(model_performance.items(), key=lambda x: (x[1]['MSE'], -x[1]['R-squared']))

print("Regression Models in Order of Precision:")
for i, (model, scores) in enumerate(sorted_models, start=1):
    print(f"{i}. {model}")
    print("   Mean Squared Error (MSE):", scores['MSE'])
    print("   R-squared Score:", scores['R-squared'])

# Define colors for the bar plots
mse_colors = ['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#3a86ff', '#8338ec']
r2_colors = ['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#3a86ff', '#8338ec']

plt.figure(figsize=(10, 6))
models = list(model_performance.keys())
mse_scores = [model_performance[model]['MSE'] for model in models]
plt.barh(models, mse_scores, color=['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b'])
plt.xlabel('Mean Squared Error (MSE)')
plt.ylabel('Regression Models')
plt.title('Performance Comparison - Mean Squared Error (MSE)')
plt.show()

plt.figure(figsize=(10, 6))
r2_scores = [model_performance[model]['R-squared'] for model in models]
plt.barh(models, r2_scores, color=['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b'])
plt.xlabel('R-squared Score')
plt.ylabel('Regression Models')
plt.title('Performance Comparison - R-squared Score')
plt.show()

# --------------------------------------------------------------------------


# Create a dictionary to store the model performance
model_performance = {
    'Ridge Regression': {'Predicted': ridge_y_pred, 'Actual': y_test},
    'Lasso Regression': {'Predicted': lasso_y_pred, 'Actual': y_test},
    'Elastic Net Regression': {'Predicted': elastic_net_y_pred, 'Actual': y_test},
    'Polynomial Regression': {'Predicted': poly_y_pred, 'Actual': y_test},
    'Decision Tree Regression': {'Predicted': tree_y_pred, 'Actual': y_test},
    'Random Forest Regression': {'Predicted': forest_y_pred, 'Actual': y_test},
    'Support Vector Regression': {'Predicted': svr_y_pred, 'Actual': y_test},
    'XGBoost Regression': {'Predicted': xgb_y_pred, 'Actual': y_test},
    'K-Nearest Neighbors Regression': {'Predicted': knn_y_pred, 'Actual': y_test},
    'Bayesian Regression': {'Predicted': bayesian_y_pred, 'Actual': y_test},
    'Neural Network Regression': {'Predicted': nn_y_pred, 'Actual': y_test},
    'Gradient Boosting Regression': {'Predicted': gb_y_pred, 'Actual': y_test}
}


import seaborn as sns

# Set up figure and axes
num_models = len(model_performance)
num_rows = (num_models // 3) + (1 if num_models % 3 != 0 else 0)
fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))

# Define color palette with vibrant colors
colors = ['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#ff006e', '#01a2a6', '#ff7c43', '#ff3838', '#b5179e', '#ce4257', '#ffc857']

# Iterate over the models and plot the predicted vs actual values using seaborn
for i, (model, performance) in enumerate(model_performance.items()):
    row = i // 3
    col = i % 3
    ax = axes[row, col] if num_rows > 1 else axes[col]

    # Get the predicted and actual values
    y_pred = performance['Predicted']
    y_actual = performance['Actual']

    # Create a scatter plot using seaborn
    sns.scatterplot(x=y_actual, y=y_pred, ax=ax, color=colors[i], alpha=0.5)

    # Add a diagonal line for reference
    ax.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], color='r')

    # Set the title and labels
    ax.set_title(model)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')

    # Add gridlines
    ax.grid(True)

    # Adjust spacing between elements to avoid overlapping text
    ax.margins(0.2)

# Adjust spacing between subplots and increase spacing between elements
fig.tight_layout(pad=2.0)

# Set subplot configurations
plt.subplots_adjust(left=0.055, bottom=0.036, right=0.75, top=0.975, wspace=0.22, hspace=0.4)

# Create a legend with vibrant colors
legend_colors = [plt.Line2D([0], [0], marker='o', color='w', label=model, markerfacecolor=colors[i], markersize=10) for i, model in enumerate(model_performance.keys())]
plt.legend(handles=legend_colors, loc='upper right')

# Show the plot
plt.show()


# ------------------------------------------------------------------------
# Define the models and their precision scores
models = [
    'Random Forest Regression',
    'XGBoost Regression',
    'K-Nearest Neighbors Regression',
    'Decision Tree Regression',
    'Gradient Boosting Regression',
    'Neural Network Regression',
    'Polynomial Regression',
    'Support Vector Regression',
    'Bayesian Regression',
    'Ridge Regression',
    'Elastic Net Regression',
    'Lasso Regression'
]

precision_scores = [
    0.990276194994097,
    0.9858625242363743,
    0.9831451090155859,
    0.9765698104608589,
    0.9177696972828497,
    0.8720966836894983,
    0.8343106562401075,
    0.7100732276074263,
    0.703886669212529,
    0.7026070438622325,
    0.3918876650365126,
    0.3845537371325256
]

# Set up figure and axes
fig, ax = plt.subplots(figsize=(8, 8))

# Define color palette for the pie chart
colors = ['#3a86ff', '#8338ec', '#ff006e', '#fb5607', '#ffbe0b', '#ff006e', '#01a2a6', '#ff7c43', '#ff3838', '#b5179e', '#ce4257', '#ffc857']

# Create the pie chart
ax.pie(precision_scores, labels=models, colors=colors, autopct='%1.1f%%', startangle=90)

# Set the title
ax.set_title('Model Precision')

# Equal aspect ratio ensures that pie is drawn as a circle
ax.axis('equal')

# Add greyish gridlines
ax.grid(color='lightgrey')

# Show the plot
plt.show()


# --------------------------------------------------------------------------
# Compute the correlation matrix
corr_matrix = merged_df[['Prevalence - Schizophrenia - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Bipolar disorder - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Eating disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Anxiety disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Drug use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Depressive disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'Prevalence - Alcohol use disorders - Sex: Both - Age: Age-standardized (Percent)',
                         'DALYs (Disability-Adjusted Life Years) - Mental disorders - Sex: Both - Age: All Ages (Percent)']].corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap - Diseases and Mental Fitness')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(rotation=0, ha='right', fontsize=8)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------

# Perform hyperparameter tuning for Random Forest Regression
from sklearn.model_selection import RandomizedSearchCV

# Define the parameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
    'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
}

# Create the Random Forest model instance
rf_model = RandomForestRegressor()

# Perform RandomizedSearchCV
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_grid_rf, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
random_search_rf.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params_rf = random_search_rf.best_params_
best_rf_model = random_search_rf.best_estimator_

# Evaluate the best model on the test set
best_rf_y_pred = best_rf_model.predict(X_test)
best_rf_mse = mean_squared_error(y_test, best_rf_y_pred)
best_rf_r2 = r2_score(y_test, best_rf_y_pred)

print("Best Random Forest Regression Model:")
print("Best Hyperparameters:", best_params_rf)
print("MSE:", best_rf_mse)
print("R-squared Score:", best_rf_r2)