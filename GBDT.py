

from sklearn.datasets import make_regression 

X, y = make_regression(n_samples = 1000, n_features = 10, n_informative = 6, noise = 2.0)

X.shape, y.shape

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

from sklearn.ensemble import GradientBoostingRegressor

M = [10, 50, 100, 200, 500, 1000, 2000]    #No. of Models


train_scores = []
test_scores = []


for m in M:
  model = GradientBoostingRegressor(n_estimators = m)
  model.fit(x_train, y_train)
  tr_sc = model.score(x_train, y_train)
  te_sc = model.score(x_test, y_test)

  train_scores.append(tr_sc)
  test_scores.append(te_sc)

train_scores

test_scores
