import pandas as p 

dataset = p.read_csv("insurance_pre.csv")

dataset = p.get_dummies(dataset,drop_first=True)
#print(dataset)

independant = dataset[['age','bmi','children','sex_male','smoker_yes']]
#print(independant)
dependant = dataset[['charges']]
#print(dependant)

from sklearn.model_selection import train_test_split  
x_train,x_test,y_train,y_test = train_test_split(independant,dependant,test_size=0.30,random_state=0)
#print("before std: ", x_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
#print("after std: ", x_train)

from sklearn.svm import SVR   
reg_or = SVR(kernel="linear") 
reg_or.fit(x_train,y_train)  

bias = reg_or.intercept_ 
nosv = reg_or.n_support_ 
print('Bias: ',bias)
print('nosv: ',nosv)

predict = reg_or.predict(x_test)  

from sklearn.metrics import r2_score
r_score = r2_score(y_test,predict) 
print('R Score: ',r_score) # good model score should be nearest to 1.

# R Score:  -0.010102665316081394