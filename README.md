# grab_sea_safety
Submission for Grab AI challenge

<br/>
To make prediction, the steps are as followed:

- Ensure these packages are available: numpy, pandas, keras, tensorflow, sklearn, dill, joblib, xgboost, hyperopt, random, datetime, scipy, . All of the packages can be installed by: pip install <package_name>
- Download the repository
- Open the notebook "Grab, Safety Challenge - Prediction.ipynb" via Jupyter Notebook, paste the file location into test_features_link variable (the format should have front slash "/", for example: C:/Data), and run the cell.
- Afterward, the "safety_predictions.csv" will be created. The process might take a few minutes.

<br/>
The prediction model involve the following models and steps:

- Sequential neural network
- K-nearest neighbors classifier
- Elastic net logistic regression
- Random forest
- Extreme gradient boosting
- The prediction of the previous models are combined via logistic regression.
- Hyperopt is used to tune hyper-paramters

<br/>
Assumptions used in feature engineering:

- Maximum trip length is 130 minutes. Those timestamps with more than 6,000,000 seconds are removed.
- Maximum speed is 40 m/s. Those records with more than 40 m/s are removed.
- Accuracy of 65 is of WiFi GPS, and accuracy of 1414 is of cell tower GPS.
- For label data, a trip which is labelled as both safe and dangerous is relablled as dangerous.

<br/>
Key features involved:

- Change in bearing: assume that bearing change is always below 180 degree. For example, the change of bearing from 2 degree to 359 degree is 3 degree, instead of 357 degree.
- Total acceleration and total gyro, based on Pythagorean formula.
- Rolling mean of various measures.
- Movement speed during turning to the left/right.
- Change of acceleration and speed over time. 
- Measurements of distribution shapes: mean, median, mode, skewness, kurtosis, semi standard deviation, mean extreme values.
- Percentage of faulty measurements, such as percentage of extreme accuracy, percentage of negative speed records, etc.

