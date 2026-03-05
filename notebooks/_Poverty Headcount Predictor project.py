#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load dataset
df = pd.read_csv("Poverty Dataset.csv")

print("Dataset Shape:", df.shape)
df.head()


# In[2]:


print("\nColumns:\n", df.columns)
print("\nMissing Values:\n", df.isnull().sum())

print("\nStatistical Summary:\n")
df.describe()


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select numeric columns only
numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(10,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Poverty Indicators")
plt.show()


# In[4]:


plt.figure(figsize=(8,5))
sns.histplot(df["Headcount_Ratio (H)"], bins=15, kde=True)

plt.title("Distribution of Poverty Headcount Ratio")
plt.xlabel("Headcount Ratio (H)")
plt.ylabel("Count")
plt.show()


# In[5]:


fig, axes = plt.subplots(1, 3, figsize=(18,5))

sns.scatterplot(
    x="Average Household Size",
    y="Headcount_Ratio (H)",
    data=df,
    ax=axes[0]
)

sns.scatterplot(
    x="Electricity Access",
    y="Headcount_Ratio (H)",
    data=df,
    ax=axes[1]
)

sns.scatterplot(
    x="Average Annual Income",
    y="Headcount_Ratio (H)",
    data=df,
    ax=axes[2]
)

axes[0].set_title("Household Size vs Poverty")
axes[1].set_title("Electricity Access vs Poverty")
axes[2].set_title("Income vs Poverty")

plt.tight_layout()
plt.show()


# In[6]:


plt.figure(figsize=(8,5))
sns.boxplot(x=df["Average Annual Income"])
plt.title("Income Before Outlier Handling")
plt.show()


# In[7]:


from scipy.stats.mstats import winsorize

df["Income_Winsorized"] = winsorize(
    df["Average Annual Income"],
    limits=[0.05, 0.05]
)

plt.figure(figsize=(8,5))
sns.boxplot(x=df["Income_Winsorized"])
plt.title("Income After Winsorization")
plt.show()


# In[8]:


X = df[[
    "Average Household Size",
    "Electricity Access",
    "Income_Winsorized"
]]

y = df["Headcount_Ratio (H)"]


# In[9]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)


# In[10]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

model = RandomForestRegressor(
    n_estimators=1000,
    max_depth=10,          # stability improvement
    min_samples_split=4,
    min_samples_leaf=3,    # stability improvement
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Random Forest Results:")
print("R2 Score:", r2)
print("MSE:", mse)
print("RMSE:", rmse)


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Gradient Boosting
gb = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)

print("\nModel Comparison:")
print("Linear Regression R2:", r2_score(y_test, lr_pred))
print("Random Forest R2:", r2)
print("Gradient Boosting R2:", r2_score(y_test, gb_pred))


# In[12]:


from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("Cross Validation Scores:", cv_scores)
print("Average CV R2:", np.mean(cv_scores))


# In[13]:


importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(8,5))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.ylabel("Importance Score")
plt.show()


# In[14]:


plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()],
         [y.min(), y.max()],
         'r--')

plt.xlabel("Actual Poverty Ratio")
plt.ylabel("Predicted Poverty Ratio")
plt.title("Actual vs Predicted")
plt.show()


# In[15]:


residuals = y_test - y_pred

plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()


# In[ ]:




