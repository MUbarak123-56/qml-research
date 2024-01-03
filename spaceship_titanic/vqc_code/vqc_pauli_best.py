import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from qiskit import *
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.circuit.library import EfficientSU2, TwoLocal, NLocal, RealAmplitudes, ZFeatureMap, ZZFeatureMap, PauliFeatureMap
from qiskit.circuit.library import CCXGate, CRZGate, RXGate
from qiskit.algorithms.optimizers import COBYLA, SLSQP, SPSA
from matplotlib import pyplot as plt
from IPython.display import clear_output
from qiskit.primitives import Sampler
from qiskit.circuit import ParameterVector
from sklearn.metrics import recall_score, precision_score, f1_score
import time
from qiskit_machine_learning.algorithms.classifiers import VQC
import os

train = pd.read_csv("../data/train_fe_small.csv")
test = pd.read_csv("../data/test_fe.csv")

cols = ['region_South', 'region_West', 'account_length','number_vmail_messages', 'total_day_minutes', 'total_day_calls',
        'total_intl_charge', 'customer_service_calls', 'churn']

train = train[cols]

x_train_use, y_train_use = train.drop("churn", axis = 1), train["churn"]

test = test[cols]

x_test, y_test = test.drop("churn", axis =1), test["churn"]

pauli_feature_map = PauliFeatureMap(feature_dimension=len(x_train_use.columns), reps=1)
sampler = Sampler()
ansatz_su = EfficientSU2(num_qubits=pauli_feature_map.width(), reps = 2, su2_gates=["ry", "rz"], entanglement= "full",
                         insert_barriers=True)
num_iter=300
cobyla = COBYLA(maxiter = num_iter)
model = VQC(
        sampler=sampler,
        feature_map=pauli_feature_map,
        ansatz=ansatz_su,
        optimizer=cobyla)

#model.load("../vqc_model/train_process/su2_cobyla.model")

x_train_arr = x_train_use.to_numpy()
y_train_arr = y_train_use.to_numpy()
x_test_arr = x_test.to_numpy()
y_test_arr = y_test.to_numpy()

start = time.time()
model.fit(x_train_arr, y_train_arr)
elapsed = time.time() - start

pred_use = model.predict(x_train_arr)
pred_test = model.predict(x_test_arr)

f1_train = f1_score(y_train_use, pred_use)
prec_train = precision_score(y_train_use, pred_use)
recall_train = recall_score(y_train_use, pred_use)

f1_test = f1_score(y_test, pred_test)
prec_test = precision_score(y_test, pred_test)
recall_test = recall_score(y_test, pred_test)

conf = confusion_matrix(y_test, pred_test)
plt.figure(figsize=(10,10))
sns.heatmap(conf, annot=True, cmap="Blues")
#plt.show();
plt.savefig("../vqc_conf/vqc_best.png")

df = pd.DataFrame()
df["f1_test"] = [f1_test]
df["f1_train"] = f1_train
df["prec_train"] = prec_train
df["prec_test"] = prec_test
df["recall_train"] = recall_train
df["recall_test"] = recall_test
df["model"] = "VQC_best"
df["elapsed"] = elapsed

df.to_csv("../vqc_results/final/vqc.csv", index=False)