# [Cross Validation]

# We can get a better sense of a model's performance using what's known as a holdout set:

from sklearn.cross_validation import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)


# We can also use cross-validation to do a sequence of fits where each subset of data is used both as a training and validation set
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

# which would yield -> (0.95999999999999996, 0.90666666666666662)

# We can then combine by taking the mean to get a better measure of the global model performance. This form of cross-validation 
# is a two-fold cross-validation.

# We can also split the data into 5 groups, and use each of them in turn to evaluate the model fit on the other 4/5 of the data.

from sklearn.cross_validation import cross_val_score
cross_val_score(model, X, y, cv=5)

# We might also wish to go to the extreme case in which our number of folds is equal to the number of data points; so we train on 
# all points but one in each trial. This is known as leave-one-out cross-validation:

from skearn.cross_validation import LeaveOneOut
scores = cross_val_score(model, X, y, cv=LeaveOneOut(len(X)))



# [Scoring]

# R^2 - coefficient of determination - measures how well a model performs relative to a simple mean of the target values.
# R^2 = 1 indicates a perfect match while R^2 = 0 indicates the model does no better than simply taking the mean of the data, and
# negative values mean even worse models.



# [Feature Engineering]




