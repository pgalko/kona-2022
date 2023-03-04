import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv('kristian_blummenfelt_copyright_entalpi_as.csv')

# drop NaN values
df = df.dropna()

# compute and print dataframe correlations
df_corr = df.corr()
print(df_corr)

# Split the data 80/20 and store in a new test/train dataframes
data_train, data_test = train_test_split(df, test_size=0.2, random_state=8)

# Define the Random Forrest Regressor ML Model to predict core temperature
rf_model = ensemble.RandomForestRegressor(n_estimators=140,min_samples_split=3,min_samples_leaf=1,max_features='sqrt',max_depth=140,bootstrap=False, random_state=8)

### Train the model on the 80% of the dataset and evaluate on the 20% of the dataset ####

# Train the model on data_train dataframe
model = rf_model.fit(np.column_stack((data_train['speed'],data_train['elevation'],data_train['cadence'],data_train['stride_length'])), data_train['core_temperature'])
# Make a prediction and store in the yhat_variable
yhat = model.predict(np.column_stack((data_test['speed'],data_test['elevation'],data_test['cadence'],data_test['stride_length'])))

# compute the correlation, rmse and mape
rmse = np.sqrt(mean_squared_error(data_test['core_temperature'], yhat))
mape = mean_absolute_percentage_error(data_test['core_temperature'], yhat)
corr = np.corrcoef(data_test['core_temperature'], yhat)

# calculate and plot feature importance for the model
importance = model.feature_importances_
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()

# print the results
print('---------------------------------')
print('Core Temp Correlation: {}'.format(corr[0,1]))
print('Core Temp RMSE: {}'.format(rmse))
print('Core MAPE: {} %'.format( mape*100))

# plot the actual core temperature vs the predicted core temperature
fig, ax = plt.subplots()
plt.scatter(data_test['core_temperature'], yhat,alpha=0.5)
ax.plot([min(data_test['core_temperature']), max(data_test['core_temperature'])], [min(data_test['core_temperature']), max(df['core_temperature'])], 'k--', lw=1)
plt.xlabel('Actual Core Temp')
plt.ylabel('Predicted Core Temp')
# plot density on the right y axes
ax2 = plt.twinx()
density = gaussian_kde(data_test['core_temperature'])
xs = np.linspace(min(data_test['core_temperature']), max(data_test['core_temperature']), 200)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs)*120,linewidth=1, color='grey', alpha=1, linestyle='--')
ax2.set_ylabel('Density')
plt.title('Actual Core Temp vs Predicted Core Temp')
plt.show()



