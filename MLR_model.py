import pandas as p 

dataset = p.read_csv("insurance_pre.csv")

dataset = p.get_dummies(dataset,drop_first=True)
#print(dataset)

independant = dataset[['age','bmi','children','sex_male','smoker_yes']]
#print(independant)
dependant = dataset[['charges']]
print(dependant)

from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test = train_test_split(independant,dependant,test_size=0.30,random_state=0)

from sklearn.linear_model import LinearRegression  
reg_or = LinearRegression() 
reg_or.fit(x_train.values,y_train.values)  

#weight and bias is calculated
weight = reg_or.coef_
bias = reg_or.intercept_
print('Weight: ',weight)
print('Bias: ',bias)
# Passing inputs to process prediction
predict = reg_or.predict(x_test.values)  #y_test is actual data to compare the result

from sklearn.metrics import r2_score
# R2 is validating the prediction score between 0 and 1.
r_score = r2_score(y_test,predict) 
print('R Score: ',r_score) # good model score should be nearest to 1.

#R Score:  0.7894790349867009