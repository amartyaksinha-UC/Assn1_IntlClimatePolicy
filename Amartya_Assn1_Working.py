# CNETID: amartyaksinha
# Working document for Assignment 1 for International Climate Policy

import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

path = r'C:\Users\amart\OneDrive - The University of Chicago\IntlClimatePolicy_PPHA39930\Assignments\Assn1_IntlClimatePolicy'

# Q2 (Perth data)
# Q2 a) 
perth = 'indiv1_perth_airport.csv'
perth_df = pd.read_csv(os.path.join(path, perth), engine='python')

perth_df['DATE'] = pd.to_datetime(perth_df['DATE'])
perth_df.set_index('DATE', inplace=True)

# Filter data from 1981 to 2010
filtered_df = perth_df.loc['1981':'2010']
# Calculate the average precipitation for each month
monthly_climatology = filtered_df.groupby(filtered_df.index.month)['PRCP'].mean()
# Print the average precipitation for each month
for month, avg_precipitation in monthly_climatology.items():
    print(f"Month: {month}, Average Precipitation: {avg_precipitation}")
# Find the rainiest month
rainiest_month = monthly_climatology.idxmax()
print(f"The rainiest month on average across 1981 to 2019 is month number {rainiest_month}")

# Create a figure and a set of subplots
fig, ax = plt.subplots()
# Plot the average precipitation for each month
ax.bar(monthly_climatology.index, monthly_climatology.values)
# Setting x and y-axis labels
ax.set_xlabel('Month')
ax.set_ylabel('Average Precipitation (mm)')

ax.set_title('Monthly Precipitation Climatology of Perth (1981-2010)')

# List of month names for x-tick labels
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
ax.set_xticks(range(1, 13))
ax.set_xticklabels(months, rotation=45)

# Showing values of y-axis by annotating each data point with its 
# corresponding value
for i, v in enumerate(monthly_climatology.values):
    ax.text(i+1, v + 0.5, str(round(v, 2)), ha='center')

plt.show()

# Q2 b)
# Filter data from 1944 onwards
filtered_df = perth_df.loc['1944':]

def plot_rainfall_trend(df, month, y_label, title):
    # Ensure month is a list-like object
    if not isinstance(month, (list, tuple)):
        month = [month]

    # Filter data for the specified month
    if len(month) == 1:
        monthly_rainfall = df[df.index.month == month[0]]['PRCP'].resample('Y').mean().reset_index()
    else:
        monthly_rainfall = df[df.index.month.isin(month)]['PRCP'].resample('Y').mean().reset_index()
    
    # Create a figure and set the size
    plt.figure(figsize=(10, 6))
    
    # Scatter plot for average monthly rainfall
    plt.scatter(x=monthly_rainfall['DATE'].dt.year, y=monthly_rainfall['PRCP'], color='skyblue', label='Average Monthly Rainfall')
    
    # Fit a linear trend line
    X = monthly_rainfall['DATE'].dt.year.values.reshape(-1, 1)
    y = monthly_rainfall['PRCP'].values
    model = LinearRegression().fit(X, y)
    trend_line = model.predict(X)
   
    # Calculate R-squared value
    r_squared = r2_score(y, trend_line)

    # Plot the linear trend line
    plt.plot(monthly_rainfall['DATE'].dt.year, trend_line, color='red', label=f'Linear Trend Line (R²={r_squared:.2f})')

    # Setting x and y-axis labels
    plt.xlabel('Year')
    plt.ylabel(y_label)
    
    # Set the title and show legend
    plt.title(title)
    plt.legend()
    
    # Show the plot
    plt.show()

plot_rainfall_trend(filtered_df, 7, 'July Rainfall (mm)', 'Average July Rainfall in Perth with Linear Trend Line (1944-2019)')

# Performing statistical test using two-sample t-test
earlier_period = perth_df.loc['1951-01-01':'1980-12-31']['PRCP']
later_period = perth_df.loc['1981-01-01':'2010-12-31']['PRCP']

t_statistic, p_value = ttest_ind(earlier_period, later_period)

print(f'Two-Sample T-Test Results:')
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')

# Check significance at a 95% confidence level
alpha = 0.05
if p_value < alpha:
    print(f'Difference between the two periods is statistically significant (reject H0).')
else:
    print(f'Difference between the two periods is not statistically significant (fail to reject H0).')

# Q2 c)
plot_rainfall_trend(filtered_df, [5, 6, 7, 8], 'Winter Rainfall (mm)', 'Average Winter Rainfall in Perth with Linear Trend Line (1944-2019)')

# Filter data for winter months (May-August)
winter_rainfall = perth_df[perth_df.index.month.isin(range(5, 9))]['PRCP'].resample('Y').mean().reset_index()

# Performing statistical test using two-sample t-test for average winter rainfall trend
early_period = winter_rainfall[(winter_rainfall['DATE'].dt.year >= 1951) & (winter_rainfall['DATE'].dt.year <= 1980)]['PRCP']
later_period = winter_rainfall[(winter_rainfall['DATE'].dt.year >= 1981) & (winter_rainfall['DATE'].dt.year <= 2010)]['PRCP']

t_statistic, p_value = ttest_ind(early_period, later_period)

print(f'Two-Sample T-Test Results for Average Winter Rainfall Trend:')
print(f'T-statistic: {t_statistic}')
print(f'P-value: {p_value}')

# Check significance at a 95% confidence level
alpha = 0.05
if p_value < alpha:
    print(f'Difference in average winter rainfall trend is statistically significant (reject H0).')
else:
    print(f'Difference in average winter rainfall trend is not statistically significant (fail to reject H0).')
