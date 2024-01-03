import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train = pd.read_csv("../data/train_fe_small.csv")
#test = pd.read_csv("../data/test_fe.csv")

cols = ['total_day_minutes', 'total_day_calls','total_intl_charge', 'customer_service_calls', 'account_length','number_vmail_messages', 
        'region_South', 'region_West','churn']

import numpy as np

from qiskit.circuit.library import PauliFeatureMap
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit import BasicAer
from qiskit.utils import algorithm_globals
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

train = train[cols]

x_train_use, y_train_use = train.drop("churn", axis = 1), train["churn"]
x_train, x_val, y_train, y_val = train_test_split(x_train_use, y_train_use, train_size=0.8, random_state = 42)
x_train=np.array(x_train)
y_train=np.array(y_train)

num_qubits=len(cols)-1
taus = [1, 10, 20, 50, 100, 200,]
Cs = [10, 100, 200, 500, 750, 1000]

train_features= x_train
train_labels=y_train

val_features=x_val
val_labels=y_val

train_features= np.array(train_features)
train_labels=np.array(train_labels)

val_features= np.array(val_features)
val_labels=np.array(val_labels)

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(conf_matrix,tau, C):
    #num = len(os.listdir("../vqc_conf/train_"))
    conf_num = "../conf/pegasos_conf/" + str(tau) + "_" + str(C) + ".png"
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(conf_num)
    #plt.show()
    
from qiskit_machine_learning.algorithms import PegasosQSVC
from itertools import product
import time

algorithm_globals.random_seed = 12345

feature_map = PauliFeatureMap(feature_dimension=num_qubits, reps=1)

qkernel = FidelityQuantumKernel(feature_map=feature_map)

results = {}
predicts={}
conf_matrixs={}


# Loop over all combinations of hyperparameters
for tau, C in product(taus, Cs):
    pegasos_qsvc = PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
    
    start_time = time.time()
    pegasos_qsvc.fit(train_features, train_labels)
    elapsed = time.time() - start_time
    print(f"Tau: {tau}, C: {C}, Training Completed, Start Validating")
    
    train_score = pegasos_qsvc.score(train_features, train_labels)
    pegasos_score = pegasos_qsvc.score(val_features, val_labels)
    
    results[(tau, C)] = {'Training Time': elapsed, 'val Score': pegasos_score}
    print(results[(tau, C)])
    
    #Predictions
    train_pred= pegasos_qsvc.predict(train_features)
    y_pred = pegasos_qsvc.predict(val_features)
    predicts[(tau, C)] = {'Predict': y_pred}
    
    #f1
    f1_train=f1_score(train_labels, train_pred)
    f1_val=f1_score(val_labels, y_pred)
    
    #precision
    prec_train = precision_score(train_labels, train_pred)
    prec_val = precision_score(val_labels, y_pred)
    
    #recall
    recall_train = recall_score(train_labels, train_pred)
    recall_val= recall_score(val_labels, y_pred)
    
    #Conf Matrix
    conf_matrix = confusion_matrix(train_labels, train_pred)
    conf_matrixs[(tau, C)] = {'Conf_Matrix': conf_matrix}
    #plot_confusion_matrix(conf_matrix, tau, C)
    
    conf_matrix = confusion_matrix(val_labels, y_pred)
    conf_matrixs[(tau, C)] = {'Conf_Matrix': conf_matrix}
    plot_confusion_matrix(conf_matrix, tau, C)
    
    #Classification Report
    print("Classification Report Train:")
    print(classification_report(train_labels, train_pred))
    
    print("Classification Report val:")
    print(classification_report(val_labels, y_pred))
    
    #num = len(os.listdir("../model"))
    pegasos_mod_num = "../model/mod_pegasos__pauli_" + str(tau)+'_'+str(C) + ".model"
    pegasos_qsvc.save(pegasos_mod_num)
    
    #Save_result
    df = pd.DataFrame({"one":[1]})
    df["feature_map_type"] = "Pauli"
    df["train_score"] = train_score
    df["val_score"] = pegasos_score
    #df["recall_score"] = recall_score(val_labels, y_pred, average='weighted')
    #df["f1_score"] = f1_score(val_labels, y_pred, average='weighted')
    df["f1_val"] = f1_val
    df["f1_train"] = f1_train
    df["prec_train"] = prec_train
    df["prec_val"] = prec_val
    df["recall_train"] = recall_train
    df["recall_val"] = recall_val
    #df["precision_score"] = precision_score(val_labels, y_pred, average='weighted')
    df["Quantum Kernel"] = "Yes"
    #df["PCA"] = "No"
    df["Training time"] = elapsed
    df["Model"] = "PEGASOS"
    df["tau"]=tau
    df["C"]=C
    #df["Max Iter"] = ''
    df = df.drop("one", axis = 1)
    #num = len(os.listdir("../result"))
    
    results[(tau, C)] =df
    
    df.to_csv("../results/regular/pegasos_results/res_pegasos_pauli_" + str(tau)+"_"+str(C) + ".csv", index = False)