import os
import pandas as pd
res_list =os.listdir("../results/regular/pegasos_results")
df = pd.DataFrame()
for res in res_list:
    if "pegasos" in res:
        new_df = pd.read_csv("../results/regular/pegasos_results/" + res)
        df = pd.concat([df, new_df], axis = 0).reset_index(drop=True)

df.to_csv("../results/regular/pegasos_results/all.csv", index=False)

df = pd.read_csv("../results/regular/pegasos_results/all.csv")

#df.sort_values("f1_val", ascending=False) # replace test with val

import numpy as np
from sklearn.model_selection import cross_val_score
from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import BasicAer
from qiskit.utils import algorithm_globals
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../data/train_fe_small.csv")
test = pd.read_csv("../data/test_fe.csv")

cols = ['region_South', 'region_West', 'account_length','number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_intl_charge', 'customer_service_calls', 'churn']

train = train[cols]
x_train_use, y_train_use = train.drop("churn", axis = 1), train["churn"]

x_train_use = x_train_use.to_numpy()
y_train_use = y_train_use.to_numpy()

algorithm_globals.random_seed = 12345

feature_map = PauliFeatureMap(feature_dimension=len(cols)-1, reps=1)

qkernel = FidelityQuantumKernel(feature_map=feature_map)

model = PegasosQSVC(quantum_kernel=qkernel, C = 1000, num_steps= 200)

import time

start = time.time()
model.fit(x_train_use, y_train_use)
end = time.time()
elapsed = end - start

pred_use = model.predict(x_train_use)

from sklearn.metrics import classification_report,f1_score, precision_score, recall_score

print(classification_report(y_train_use, pred_use))

f1_train = f1_score(y_train_use, pred_use)
prec_train = precision_score(y_train_use, pred_use)
recall_train = recall_score(y_train_use, pred_use)

test = test[cols]

x_test, y_test = test.drop("churn", axis =1), test["churn"]

x_test = x_test.to_numpy()
y_test = y_test.to_numpy()

pred_test = model.predict(x_test)

print(classification_report(y_test, pred_test))

f1_test = f1_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
recall_test = recall_score(y_test, pred_test)

df = pd.DataFrame()
df["f1_test"] = [f1_test]
df["f1_train"] = f1_train
df["prec_train"] = prec_train
df["prec_test"] = prec_test
df["recall_train"] = recall_train
df["recall_test"] = recall_test
df["elapsed"] = elapsed
df["model"] = "Pegasos QSVC"

df.to_csv("../results/regular/pegasos_qsvc.csv", index=False)