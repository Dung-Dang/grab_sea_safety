# grab_sea_safety
Submission for Grab AI challenge

The prediction model aims to stack the results of 5 different models, namely:
- Sequential neural network
- K-nearest neighbors classifier
- Elastic net logistic regression
- Random forest
- Extreme gradient boosting

Assumptions used in feature engineering:
- Maximum trip length is 130 minutes. Those timestamps with more than 6,000,000 seconds are removed.
- Maximum speed is 40 m/s
- Accuracy of 65 is of WiFi GPS, and accuracy of 1414 is of cell tower GPS.
- For label data, a trip which is labelled as both safe and dangerous is relablled as dangerous.
