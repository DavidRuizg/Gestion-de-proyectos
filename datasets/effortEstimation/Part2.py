import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

#read "Telecom1.csv"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

telecom1 = pd.read_csv("Telecom1.csv", sep=",", header=0, dtype=str, decimal=".")
telecom1['size'] = telecom1['size'].astype(float)
telecom1['effort'] = telecom1['effort'].astype(float)

samplesize = int(np.floor(0.66*telecom1.shape[0]))
np.random.seed(12)
train_idx = np.random.choice(telecom1.index, size=samplesize, replace=False)
telecom1_train = telecom1.loc[train_idx]
telecom1_test = telecom1.drop(train_idx)

# transformation of variables to log-log
xtrain = np.log(telecom1_train['size'])
ytrain = np.log(telecom1_train['effort'])

lmtelecom1 = np.polyfit(xtrain, ytrain, 1)
plt.scatter(xtrain, ytrain)
plt.plot(xtrain, np.polyval(lmtelecom1, xtrain), 'b-', linewidth=2)
plt.show()

#compute the mae, mse, mgae, mre of the estimations
xtest = np.log(telecom1_test['size'])
ytest = np.log(telecom1_test['effort'])
estimEffTelecom1 = np.exp(np.polyval(lmtelecom1, xtest))
actualEffTelecom1 = np.exp(ytest)
mae1 = np.mean(np.abs(estimEffTelecom1 - actualEffTelecom1))
mse1 = np.mean((estimEffTelecom1 - actualEffTelecom1)**2)
mre1 = np.mean(np.abs((estimEffTelecom1 - actualEffTelecom1) / actualEffTelecom1))


telecom1 = pd.read_csv("Telecom1.csv", sep=",", header=0, dtype=str, decimal=".")
telecom1['effort'] = telecom1['effort'].astype(float)
actualEffTelecom1 = telecom1['effort'].values
estimEffTelecom1 = telecom1['EstTotal'].values
numruns = 9999
randguessruns = np.zeros(numruns)
for i in range(numruns):
    for j in range(len(estimEffTelecom1)):
        estimEffTelecom1[j] = np.random.choice(actualEffTelecom1[actualEffTelecom1 != actualEffTelecom1[j]], size=1)
    randguessruns[i] = np.mean(np.abs(estimEffTelecom1 - actualEffTelecom1))
marp0telecom1 = np.mean(randguessruns)
plt.hist(randguessruns)
plt.title("MARP0 distribution of the Telecom1 dataset")
plt.show()

#compute the mae, mse, mgae, mre of the estimations
mae2 = np.mean(np.abs(estimEffTelecom1 - actualEffTelecom1))
mse2 = np.mean((estimEffTelecom1 - actualEffTelecom1)**2)
mre2 = np.mean(np.abs((estimEffTelecom1 - actualEffTelecom1) / actualEffTelecom1))


print("MSE1: ", mse1, "MSE2: ", mse2)
print("MAE1: ", mae1, "MAE2: ", mae2)
print("MRE1: ", mre1, "MRE2: ", mre2)
