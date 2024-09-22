import numpy as np
import pandas as pd

golf = pd.read_csv('/content/golf.csv')
golf

def prior_prob(golf, label):
  total_examples = golf.shape[0]
  class_examples = (golf['Play'] == label).sum()
  return class_examples / float(total_examples)

PRIOR = {
    'yes': prior_prob(golf, 'yes'),
    'no': prior_prob(golf, 'no')
}

print(PRIOR)

def cond_prob(golf, feature, feature_value, label):
  filtered_data = golf[golf['Play'] == label]
  numerator = np.sum(filtered_data[feature]==feature_value)
  denominator = filtered_data.shape[0]

  return numerator / float(denominator)

golf[golf['Play'] == 'yes']

cond_prob(golf, 'Windy', False, 'no')


features = list(golf.columns)[:-1]
COND_PROB = {}

for label in golf['Play'].unique():
  COND_PROB[label] = {}
  for feature in features:
    COND_PROB[label][feature] = {}

    feature_values = golf[feature].unique()

    for fea_value in feature_values:

      # no, outlook, sunny,
      prob =  round(cond_prob(golf, feature, fea_value, label), 2)
      COND_PROB[label][feature][fea_value] = prob
      print(label, feature, fea_value, prob)

    print()


COND_PROB

x_test = ["sunny", "hot", "normal", False]

for label in golf['Play'].unique():

  prior = PRIOR[label]
  liklihood = 1.0

  for i in range(len(features)):
    feature = features[i]
    fea_value = x_test[i]

    liklihood *= COND_PROB[label][feature][fea_value]


  post = liklihood*prior

  print(label, post)

Prob_Yes = (0.0139/(0.0139+0.0068))*100
Prob_No = (0.0068/(0.0139+0.0068))*100

print(Prob_Yes, Prob_No)

//Using Sklearn

golf = pd.read_csv('/content/golf.csv')
golf

from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()
golf['Outlook'] = le1.fit_transform(golf['Outlook'])

le2 = LabelEncoder()
golf['Temperature'] = le2.fit_transform(golf['Temperature'])

le3 = LabelEncoder()
golf['Humidity'] = le3.fit_transform(golf['Humidity'])

le4 = LabelEncoder()
golf['Windy'] = le4.fit_transform(golf['Windy'])

le5 = LabelEncoder()
golf['Play'] = le5.fit_transform(golf['Play'])

golf

#Output :
# 	Outlook	Temperature	Humidity	Windy	Play
# 0	  2	1	0	0	0
# 1	  2	1	0	1	0
# 2	  0	1	0	0	1
# 3	  1	2	0	0	1
# 4	  1	0	1	0	1
# 5	  1	0	1	1	0
# 6	  0	0	1	1	1
# 7	  2	2	0	0	0
# 8	  2	0	1	0	1
# 9	  1	2	1	0	1
# 10	2	2	1	1	1
# 11	0	2	0	1	1
# 12	0	1	1	0	1
# 13	1	2	0	1	0

x = golf.iloc[:, :-1]
y = golf.iloc[:, -1]

from sklearn.naive_bayes import CategoricalNB

model = CategoricalNB()
model.fit(x, y)

x_test = ["sunny", "hot", "normal", False]

le1.transform(['sunny'])
le2.transform(['hot'])
le3.transform(['normal'])
le4.transform([False])

x_test = np.array([[2, 1, 1, 0]])

model.predict_proba(x_test)
