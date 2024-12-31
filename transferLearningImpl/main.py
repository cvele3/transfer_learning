# tensorinjo = "tensorflow-2.11.0-cp38-cp38-win_amd64.whl"
# rdkitinjo = "rdkit_pypi-2022.9.5-cp38-cp38-win_amd64.whl"
# stelargrfinjo = "stellargraph-1.2.1-py3-none-any.whl"
#
# import pip
#
# def install_whl(path):
#    pip.main(['install', path])
import math

# install_whl(tensorinjo)
# install_whl(rdkitinjo)
# install_whl(stelargrfinjo)


from stellargraph.mapper import PaddedGraphGenerator
from stellargraph.layer import DeepGraphCNN
from stellargraph import StellarGraph

from stellargraph import datasets

from sklearn import model_selection
from stellargraph import StellarGraph
from stellargraph.mapper import PaddedGraphGenerator

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.losses import binary_crossentropy
import tensorflow as tf
import pandas as pd
import rdkit.Chem
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw
import numpy as np
from stellargraph import StellarGraph

# Import required libraries
from rdkit import Chem
from rdkit.Chem import Draw
from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, link_classification
from tensorflow.keras import Model, optimizers, losses, metrics
import numpy as np
import pandas as pd

# filepath_raw = 'ToxinSequenceSMILES.xlsx'
# data_file = pd.read_excel(filepath_raw, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])

# listOfTuples = []

# data_file.reset_index()
# for index, row in data_file.iterrows():
#     smiles = row['SMILES']
#     label = row["TOXICITY"]
#     molecule = (row["SMILES"], row["TOXICITY"])
#     listOfTuples.append(molecule)

# Set the new filepath to the 'out.csv' file in the 'datasetout' folder
filepath_raw = 'TransferLearning\out.xlsx'

# Read the CSV file and select only the required columns
data_file = pd.read_excel(filepath_raw, header=0, usecols=["SMILES", "HEK"])

# Initialize an empty list to store tuples
listOfTuples = []

# Iterate through each row to extract the SMILES and HEK columns
for index, row in data_file.iterrows():
    molecule = (row["SMILES"], row["HEK"])
    listOfTuples.append(molecule)


ZeroActivity = 0
OneActivity = 0
stellarGraphAllList = []

from collections import Counter

# Get a list of all unique elements in the molecules and extract features.
# At the same time, create a list of seen properties from each atom.
all_elements = []
lst_degree = []
lst_formal_charge = []
lst_radical_electrons = []
lst_hybridization = []
lst_aromatic = []
for molecule in listOfTuples:
    mol = Chem.MolFromSmiles(molecule[0])
    atoms = mol.GetAtoms()
    for atom in atoms:
        lst_degree.append(atom.GetDegree())
        lst_formal_charge.append(atom.GetFormalCharge())
        lst_radical_electrons.append(atom.GetNumRadicalElectrons())
        lst_hybridization.append(atom.GetHybridization().real)
        lst_aromatic.append(atom.GetIsAromatic())

        element = atom.GetSymbol()
        if element not in all_elements:
            all_elements.append(element)

# Determine min and max values for each property/feature.
# This values will be later used for min-max scaling.
min_degree, max_degree = min(lst_degree), max(lst_degree)
min_formal_charge, max_formal_charge = min(lst_formal_charge), max(lst_formal_charge)
min_radical_electrons, max_radical_electrons = min(lst_radical_electrons), max(lst_radical_electrons)
min_hybridization, max_hybridization = min(lst_hybridization), max(lst_hybridization)
min_aromatic, max_aromatic = min(lst_aromatic), max(lst_aromatic)

# Create a dictionary that maps each element to a unique index
element_to_index = {element: index for index, element in enumerate(all_elements)}

for molecule in listOfTuples:

    smileString = molecule[0]
    smileLabel = molecule[1]

    # Convert the SMILES string into a molecular graph using RDKit
    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    # Convert the RDKit atom objects to a Numpy array of node features
    node_features = []
    for atom in atoms:
        element = atom.GetSymbol()
        degree = (atom.GetDegree() - min_degree) / (max_degree - min_degree)
        # formal_charge = (atom.GetFormalCharge() - min_formal_charge) / (max_formal_charge - min_formal_charge)
        # num_radical_electrons = (atom.GetNumRadicalElectrons() - min_radical_electrons) / (max_radical_electrons - min_radical_electrons)
        hybridization = (atom.GetHybridization().real - min_hybridization) / (max_hybridization - min_hybridization)
        aromatic = (atom.GetIsAromatic() - min_aromatic) / (max_aromatic - min_aromatic)
        element_onehot = [0] * len(all_elements)
        element_onehot[element_to_index[element]] = 1
        node_features.append(element_onehot)
        #node_features.append(element_onehot + [degree, hybridization, aromatic])
    node_features = np.array(node_features)

    # Convert the edges to a pandas DataFrame
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    # Create a StellarGraph object from the molecular graph
    G = StellarGraph(nodes=node_features, edges=edges_df)

    # if smileLabel == 1 and OneActivity < 1000:
    if smileLabel == 1:
        OneActivity += 1
        skup = (G, smileLabel)
        stellarGraphAllList.append(skup)

    # if smileLabel == 0 and ZeroActivity < 1000:
    if smileLabel == 0:
        ZeroActivity += 1
        skup = (G, smileLabel)
        stellarGraphAllList.append(skup)

print(ZeroActivity)
print(OneActivity)

print(len(stellarGraphAllList))

import pandas as pd

# assume that the 'stellarGraphAllList' variable is defined somewhere in the code

graphs = []
labels = []

for triple in stellarGraphAllList:
    grafinjo = triple[0]
    active = triple[1]
    graphs.append(grafinjo)
    labels.append(active)

import pandas as pd

# assume that the 'stellarGraphAllList' variable is defined somewhere in the code

graph_labels = pd.Series(labels)

print(graph_labels.value_counts().to_frame())

# graph_labels = pd.get_dummies(graph_labels, drop_first=True)
generator = PaddedGraphGenerator(graphs=graphs)

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence

epochs = 10000

# Define the number of rows for the output tensor and the layer sizes
k = 25
layer_sizes = [25, 25, 25, 1]
filter1 = 16
filter2 = 32
filter3 = 128

# Create the DeepGraphCNN model
dgcnn_model = DeepGraphCNN(
    layer_sizes=layer_sizes,
    activations=["tanh", "tanh", "tanh", "tanh"],
    k=k,
    bias=False,
    generator=generator,
)
x_inp, x_out = dgcnn_model.in_out_tensors()

x_out = Conv1D(filters=filter1, kernel_size=sum(layer_sizes), strides=sum(layer_sizes))(x_out)
x_out = MaxPool1D(pool_size=2)(x_out)

x_out = Conv1D(filters=filter2, kernel_size=5, strides=1)(x_out)

x_out = Flatten()(x_out)

x_out = Dense(units=filter3, activation="relu")(x_out)
x_out = Dropout(rate=0.2)(x_out)

predictions = Dense(units=1, activation="sigmoid")(x_out)

# Create the model and compile it
model = Model(inputs=x_inp, outputs=predictions)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, f1_score
import tensorflow.keras.backend as K

gm_values = []
precision_values = []
recall_values = []
f1_values = []
roc_auc_values = []
fpr_values = []
tpr_values = []


def roc_auc_metric(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array

    roc_auc = roc_auc_score(y_true, y_pred)
    print("ROC AUC: ", roc_auc)
    roc_auc_values.append(K.get_value(roc_auc))


def rest_of_metrics(y_true, y_pred):
    y_pred = K.cast(K.round(y_pred), K.floatx())
    y_true = K.cast(y_true, K.floatx())

    y_pred_np = K.eval(y_pred)  # Convert y_pred tensor to NumPy array

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_np).ravel()
    # fpr = K.variable(fp / (fp + tn + K.epsilon()))
    # tpr = K.variable(tp / (tp + fn + K.epsilon()))
    # tnr = K.variable(tn / (tn + fp + K.epsilon()))
    # gm = K.sqrt(tpr * tnr)
    # precision = precision_score(y_true, y_pred_np)
    # recall = recall_score(y_true, y_pred_np)
    # f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    # roc_auc = roc_auc_score(y_true, y_pred_np)

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gm = math.sqrt(tpr * tnr)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("GM: ", gm)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    gm_values.append(K.get_value(gm))
    precision_values.append(K.get_value(precision))
    recall_values.append(K.get_value(recall))
    f1_values.append(K.get_value(f1))
    fpr_values.append(K.get_value(fpr))
    tpr_values.append(K.get_value(tpr))


restOfMetrics_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: metrics(test_gen.targets, model.predict(test_gen)))

mcc_values = []


# Create a LambdaCallback to calculate MCC at the end of each epoch
def mcc_metric(y_true, y_pred):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_pred = y_pred.round()
    mcc = matthews_corrcoef(y_true, y_pred)
    print("MCC: ", mcc)
    mcc_values.append(mcc)


mcc_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: mcc_metric(test_gen.targets, model.predict(test_gen)))
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

model.compile(
    optimizer=Adam(lr=0.0001),
    loss=binary_crossentropy,
    metrics=["acc"]
)

# Add evaluation metrics to the model
# model.metrics_names.append('mcc')
# model.metrics_names.append('tpr')
# model.metrics_names.append('tnr')
# model.metrics_names.append('gm')
# model.metrics_names.append('precision')
# model.metrics_names.append('recall')
# model.metrics_names.append('f1')
# model.metrics_names.append('roc_auc')


cv = StratifiedKFold(n_splits=10, shuffle=True)

import matplotlib.pyplot as plt
import numpy as np

histories = []

# Use the cross-validator to get the train and test indices
for train_index, test_index in cv.split(graphs, graph_labels):
    # Extract the train and test sets using the indices
    graphs = np.array(graphs)
    X_train, X_test = graphs[train_index.astype(int)], graphs[test_index.astype(int)]
    y_train, y_test = graph_labels.iloc[train_index.astype(int)], graph_labels.iloc[test_index.astype(int)]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    gen = PaddedGraphGenerator(graphs=graphs)

    # Create generators for the train and test sets
    train_gen = gen.flow(
        X_train,
        targets=y_train,
        batch_size=32,
        symmetric_normalization=False,
    )

    val_gen = gen.flow(
        X_val,
        targets=y_val,
        batch_size=50,
        symmetric_normalization=False,
    )

    test_gen = gen.flow(
        X_test,
        targets=y_test,
        batch_size=50,
        symmetric_normalization=False,
    )

    # Train the model and evaluate on the test set
    history = model.fit(
        train_gen, epochs=epochs, verbose=1, validation_data=val_gen, shuffle=True, callbacks=[callback]
    )

    histories.append(history)

    y_pred = model.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))
    roc_auc_metric(y_test, y_pred)
    y_pred = [0 if prob < 0.5 else 1 for prob in y_pred]

    y_test = y_test.to_numpy()
    y_test = np.reshape(y_test, (-1,))

    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

# save model
model.save('toxicityModel25small.h5')

import os
import matplotlib.pyplot as plt
import stellargraph as sg

# save_dir = r"C:\Users\jcvetko\Desktop\stuff\school\6. semestar\Zavrsni rad\plotHistory 444-598 split with k30 [30,30,30,1]"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\plotHistory 444-598 split with k35 [32,32,32,1]_history"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\testDir"

# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k35 [32,32,32,1] with 32- 64- 128"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k35 [32,32,32,1] with 16 -32 -128"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k30 [30,30,30,1] with 16 -32 -128"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k25 [25,25,25,1] with 16 -32 -128"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k25 [25,25,25,1] with 8 -16 -64"
# save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k20 [15,15,15,1] with 8 -16 -64"
#save_dir = r"C:\Users\legion\PycharmProjects\PeptideML\zavrsni\updatedNodeFeaturesWithFeatures\history\plotHistory 444-598 split with k15 [10,10,10,1] with 8 -16 -64"
save_dir = r"D:\Ostalo\faks\2. semestar\Projekt_Evo\projekt3.8\TransferLearning\plotHistory"

for i, history in enumerate(histories):
    fig = sg.utils.plot_history(history, individual_figsize=(7, 4), return_figure=True)
    fig.savefig(os.path.join(save_dir, f"history_{i}.png"))
    plt.close(fig)

test_metrics = model.evaluate(test_gen)
print("\nTest Set Metrics:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

# print("AVG MCC: ")
# print(np.mean(mcc_values))
# print("MAX MCC: ")
# print(np.max(mcc_values))
# print("MIN MCC: ")
# print(np.min(mcc_values))

# Create a dictionary with the data
data = {
    "Metric": ["MCC", "GM", "Precision", "Recall", "F1", "ROC AUC", "FPR", "TPR"],
    "Average": [np.mean(mcc_values), np.mean(gm_values), np.mean(precision_values), np.mean(recall_values),
                np.mean(f1_values), np.mean(roc_auc_values), np.mean(fpr_values), np.mean(tpr_values)],
    "Maximum": [np.max(mcc_values), np.max(gm_values), np.max(precision_values), np.max(recall_values),
                np.max(f1_values), np.max(roc_auc_values), np.max(fpr_values), np.max(tpr_values)],
    "Minimum": [np.min(mcc_values), np.min(gm_values), np.min(precision_values), np.min(recall_values),
                np.min(f1_values), np.min(roc_auc_values), np.min(fpr_values), np.min(tpr_values)]
}

# Create a pandas DataFrame from the dictionary
df = pd.DataFrame(data)

# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k35 [32,32,32,1] with 32- 64- 128/"
# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k35 [32,32,32,1] with 16 -32 -128/"
# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k30 [30,30,30,1] with 16 -32 -128/"
# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k25 [25,25,25,1] with 16 -32 -128/"
# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k25 [25,25,25,1] with 8 -16 -64/"
# excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k20 [15,15,15,1] with 8 -16 -64/"
#excel_save = "zavrsni/updatedNodeFeaturesWithFeatures/metrics/plotHistory 444-598 split with k15 [10,10,10,1] with 8 -16 -64/"
excel_save = ""

filename = f"metrics k{k} {layer_sizes} with {filter1}-{filter2}-{filter3}.xlsx"

# Define the save path
excel_save = f"treciSastanak/{filename}"

# Save the DataFrame to an Excel file
df.to_excel(excel_save, index=False)