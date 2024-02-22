import pandas as pd
import os
import time
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, GridSearchCV
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../data/train_fe_small.csv")

cols = ['region_South', 'region_West', 'account_length','number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_intl_charge', 'customer_service_calls', 'churn']

train = train[cols]

x_train_use, y_train_use = train.drop("churn", axis = 1), train["churn"]

x_train, x_val, y_train, y_val = train_test_split(x_train_use, y_train_use, train_size=0.8, random_state = 42)

x_train=np.array(x_train)
y_train=np.array(y_train)

num_qubits=len(cols)-1
num_qubits

from qiskit.circuit.library import PauliFeatureMap

feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1)


from qiskit_machine_learning.algorithms import QSVC

from qiskit import BasicAer
from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.kernels import FidelityQuantumKernel



algorithm_globals.random_seed = 12345

qkernel = FidelityQuantumKernel(feature_map=feature_map)


model = QSVC(quantum_kernel=qkernel)


start = time.time()
model.fit(x_train, y_train)
elapsed = time.time() - start
train_score=model.score(x_train, y_train)
qsvc_score = model.score(x_val, y_val)

print(f"QSVC classification test score: {qsvc_score}")

y_pred = model.predict(x_train)
val_pred=model.predict(x_val)


f1_train=f1_score(y_train, y_pred)

f1_test=f1_score(y_val, val_pred)

prec_train = precision_score(y_train, y_pred)
prec_test = precision_score(y_val, val_pred)

recall_train =recall_score(y_train, y_pred)
recall_test =recall_score(y_val, val_pred)


print(classification_report(y_train, y_pred))

print(classification_report(y_val, val_pred))

def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

    
conf = confusion_matrix(y_train, y_pred)
plot_confusion_matrix(conf)

conf = confusion_matrix(y_val, val_pred)
plot_confusion_matrix(conf)


#Extract Result

model.fit(x_train_use, y_train_use)
pred_use = model.predict(x_train_use)

print(classification_report(y_train_use, pred_use))
f1_train = f1_score(y_train_use, pred_use)
prec_train = precision_score(y_train_use, pred_use)
recall_train = recall_score(y_train_use, pred_use)

test = pd.read_csv("../data/test_fe.csv")

test = test[cols]
x_test, y_test = test.drop("churn", axis =1), test["churn"]
pred_test = model.predict(x_test)
print(classification_report(y_test, pred_test))

f1_test = f1_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
recall_test = recall_score(y_test, pred_test)

conf = confusion_matrix(y_test, pred_test)

sns.heatmap(conf, annot=True, cmap="Blues")
#plt.show();

qsvc_mod_num = "../model/mod_qsvc_pauli.model"
qsvc.save(qsvc_mod_num)

df = pd.DataFrame()
df["f1_test"] = [f1_test]
df["f1_train"] = f1_train
df["prec_train"] = prec_train
df["prec_test"] = prec_test
df["recall_train"] = recall_train
df["recall_test"] = recall_test
df["model"] = "QSVC"

df.to_csv("../result/regular/qsvc.csv", index = False)  