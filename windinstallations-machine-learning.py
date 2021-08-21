import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
from graphviz import Source


df = pd.read_csv('emirhan-project-mw.csv')
#print(df.head())

connects = df.drop(['Country'],axis = 'columns')
target = df.drop(['W.I. Numbers - MW','W.I. - Offshore - MW','W.I. - Onshore - MW'],axis= 'columns')
#print(target)
#print(connects)

country_variables = LabelEncoder()

target['Country_Data'] = country_variables.fit_transform(target['Country'])
#print(connects)

target_n = target.drop(['Country'],axis='columns')
#print(target_n)


model_machine = tree.DecisionTreeClassifier()

model_machine.fit(connects,target_n)

#print(model_machine.score(connects,target))

#decisiontree

dotfile = open("dtreee.dot", 'w')
graph= Source(tree.export_graphviz(model_machine,
              #out_file=dotfile,
              #filled=True,
              #rounded=True))

model_run = model_machine.predict([[1500,250,1250]]) #Germany Prediction :))

country = pd.read_csv('country_classification.csv',index_col=None, na_values=['NA'])
country_detect = country.columns.values[model_run]
print("Predicted country: {}".format(country_detect))

print(target)
#dotfile.close()
#model_run = model_machine.predict([[1500,250,1250]])
#print(target)
#print("Predicted country:",model_run, "{}".format(x in b))
