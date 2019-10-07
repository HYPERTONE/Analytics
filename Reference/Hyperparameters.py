
# We can get a better sense of a model's performance using what's known as a holdout set:

from sklearn.cross_validation import train_test_split
# split the data with 50% in each set
X1, X2, y1, y2 = train_test_split(X, y, random_state=0, train_size=0.5)


# We can also use cross-validation to do a sequence of fits where each subset of data is used both as a training and validation set
y2_model = model.fit(X1, y1).predict(X2)
y1_model = model.fit(X2, y2).predict(X1)
accuracy_score(y1, y1_model), accuracy_score(y2, y2_model)

# which would yield -> (0.95999999999999996, 0.90666666666666662)

# We can then combine by taking th emean to get a better measure of the global model performance. This form of cross-validation 
# is a two-fold cross-validation.
