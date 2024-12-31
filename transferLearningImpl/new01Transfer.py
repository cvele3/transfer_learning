import math
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import rdkit.Chem
import networkx as nx
import matplotlib.pyplot as plt
import stellargraph as sg

from rdkit import Chem
from rdkit.Chem import Draw

from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import matthews_corrcoef, precision_recall_curve

from stellargraph import StellarGraph, datasets
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import PaddedGraphGenerator, GraphSAGELinkGenerator
from stellargraph.layer import DeepGraphCNN, GraphSAGE, link_classification

from tensorflow.keras import Model, optimizers, losses, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.utils import Sequence
import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPool1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split

# Ako je potrebno za custom slojeve iz StellarGrapha:
from stellargraph.layer import DeepGraphCNN
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph

from rdkit import Chem

# Uvoz ovih slojeva ako ih treba prilikom load_model
from stellargraph.layer import (
    DeepGraphCNN,
    GCNSupervisedGraphClassification,
    SortPooling,
    GraphConvolution,
)
from stellargraph.layer.graph_classification import SortPooling


# ======================================================================
# 1) UČITAVANJE PEPTIDA IZ XLSX-A
#    Pretpostavljamo da su kolone: SEQUENCE, TOXICITY, SMILES
# ======================================================================
filepath_peptides = "ToxinSequenceSMILES.xlsx"  # prilagodite po želji

data_peptides = pd.read_excel(filepath_peptides, header=0, usecols=["SEQUENCE", "TOXICITY", "SMILES"])

# Napravimo listu (smiles, label)
listOfTuples_peptides = []
for index, row in data_peptides.iterrows():
    smiles = row["SMILES"]
    label = row["TOXICITY"]
    listOfTuples_peptides.append((smiles, label))

print("Broj peptidnih zapisa:", len(listOfTuples_peptides))

# ======================================================================
# 2) KREIRANJE GRAFSKIH OBJEKATA (StellarGraph) - ISKLJUČIVO ONE-HOT
# ======================================================================

ZeroActivity = 0
OneActivity = 0
stellarGraphAllList = []

# Umjesto da gradimo all_elements iz podataka, 
# KORISTIMO već definiran "vokabular" s 27 elemenata:
element_to_index = {
    "N": 0,
    "C": 1,
    "O": 2,
    "F": 3,
    "Cl": 4,
    "S": 5,
    "Na": 6,
    "Br": 7,
    "Se": 8,
    "I": 9,
    "Pt": 10,
    "P": 11,
    "Mg": 12,
    "K": 13,
    "Au": 14,
    "Ir": 15,
    "Cu": 16,
    "B": 17,
    "Zn": 18,
    "Re": 19,
    "Ca": 20,
    "As": 21,
    "Hg": 22,
    "Ru": 23,
    "Pd": 24,
    "Cs": 25,
    "Si": 26,
}
# Duljina one-hot vektora = 27
NUM_FEATURES = len(element_to_index)

print("\nFiksni vokabular (27 elemenata) =", element_to_index)

# Za demonstraciju, preskočit ćemo min/max normalizaciju za "degree", "aromatic" itd.
# Ovdje se pokazuje isključivo one-hot

for molecule in listOfTuples_peptides:
    smileString = molecule[0]
    smileLabel = molecule[1]

    mol = Chem.MolFromSmiles(smileString)
    atoms = mol.GetAtoms()
    edges = []
    for bond in mol.GetBonds():
        edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edges.append((bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()))

    node_features = []
    for atom in atoms:
        elem = atom.GetSymbol()
        # Ako se elem NE nalazi u element_to_index, treba odlučiti što učiniti:
        if elem not in element_to_index:
            # npr. ignorirati, ili ubaciti "UNK" (ako smo tako definirali)
            # Ovdje samo skipamo cijeli graf ili stavljamo sve nule
            # ali po mogućnosti, bolje je ne skipati,
            # recimo stavi "UNK" = [0,0,0...]
            # Ili bacimo error - ovdje ću samo staviti zero vector:
            onehot = [0] * NUM_FEATURES
        else:
            idx = element_to_index[elem]
            onehot = [0] * NUM_FEATURES
            onehot[idx] = 1

        node_features.append(onehot)

    node_features = np.array(node_features)
    edges_df = pd.DataFrame(edges, columns=["source", "target"])

    G = StellarGraph(nodes=node_features, edges=edges_df)

    if smileLabel == 1:
        OneActivity += 1
        stellarGraphAllList.append((G, smileLabel))
    elif smileLabel == 0:
        ZeroActivity += 1
        stellarGraphAllList.append((G, smileLabel))

print("ZeroActivity:", ZeroActivity)
print("OneActivity:", OneActivity)
print("Ukupno grafo-va:", len(stellarGraphAllList))

# Dalje radimo sve po starom:
graphs = []
labels = []

for triple in stellarGraphAllList:
    g_obj = triple[0]
    active = triple[1]
    graphs.append(g_obj)
    labels.append(active)

graph_labels = pd.Series(labels)
print("Distribucija labela:\n", graph_labels.value_counts().to_frame())

# ======================================================================
# 3) DEFINIRANJE NEKIH POMOĆNIH FUNKCIJA ZA METRIKE (NE TRENIRAMO JOŠ)
# ======================================================================
gm_values, precision_values, recall_values = [], [], []
f1_values, roc_auc_values, fpr_values, tpr_values = [], [], [], []
mcc_values = []

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, matthews_corrcoef

def roc_auc_metric(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    score = roc_auc_score(y_true, y_pred_bin)
    print("ROC AUC:", score)
    roc_auc_values.append(score)

def rest_of_metrics(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
    fpr = fp / (fp + tn + 1e-9)
    tpr = tp / (tp + fn + 1e-9)
    tnr = tn / (tn + fp + 1e-9)
    gm = math.sqrt(tpr * tnr)

    precision = precision_score(y_true, y_pred_bin)
    recall = recall_score(y_true, y_pred_bin)
    f1 = f1_score(y_true, y_pred_bin)

    print("GM:", gm)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)

    gm_values.append(gm)
    precision_values.append(precision)
    recall_values.append(recall)
    f1_values.append(f1)
    fpr_values.append(fpr)
    tpr_values.append(tpr)

def mcc_metric(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    mcc = matthews_corrcoef(y_true, y_pred_bin)
    print("MCC:", mcc)
    mcc_values.append(mcc)

# ======================================================================
# 4) UČITAVANJE POSTOJEĆEG MODELA + TRANSFER LEARNING (3 METODE, 10-FOLD)
# ======================================================================
PRETRAINED_MODEL_PATH = "toxicityModel25small.h5"

def load_pretrained_model():
    model_loaded = load_model(
        PRETRAINED_MODEL_PATH,
        custom_objects={
            "DeepGraphCNN": DeepGraphCNN,
            "GCNSupervisedGraphClassification": GCNSupervisedGraphClassification,
            "SortPooling": SortPooling,
            "GraphConvolution": GraphConvolution,
        }
    )
    return model_loaded

from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# =============== METODA 1 ===============
print("\n=== METODA 1: Zamrzavanje GNN slojeva + 10-fold CV ===")
fold_index = 0
for train_index, test_index in cv.split(graphs, graph_labels):
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")

    model1 = load_pretrained_model()
    # Zamrznemo GNN slojeve
    for layer in model1.layers:
        if "deep_graph_cnn" in layer.name or "graph_conv" in layer.name:
            layer.trainable = False
        else:
            layer.trainable = True

    model1.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=binary_crossentropy,
        metrics=["accuracy"]
    )

    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)

    X_train, X_test = graphs_arr[train_index], graphs_arr[test_index]
    y_train, y_test = labels_arr[train_index], labels_arr[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    gen = PaddedGraphGenerator(graphs=graphs_arr)

    train_gen = gen.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen.flow(X_test,  y_test,  batch_size=32, shuffle=False)

    history1 = model1.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        verbose=1,
        shuffle=True
    )

    y_pred = model1.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))

    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model1.save("toxicityModel25small_peptidi_freezeGNN_folded.h5")


# =============== METODA 2 ===============
print("\n=== METODA 2: Zamrzavanje READOUT/dense slojeva + 10-fold CV ===")
fold_index = 0
for train_index, test_index in cv.split(graphs, graph_labels):
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")

    model2 = load_pretrained_model()
    # Zamrznemo readout/dense
    for layer in model2.layers:
        if any(x in layer.name for x in ["dense", "dropout", "flatten", "readout"]):
            layer.trainable = False
        else:
            layer.trainable = True

    model2.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=binary_crossentropy,
        metrics=["accuracy"]
    )

    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)

    X_train, X_test = graphs_arr[train_index], graphs_arr[test_index]
    y_train, y_test = labels_arr[train_index], labels_arr[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    gen = PaddedGraphGenerator(graphs=graphs_arr)

    train_gen = gen.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen.flow(X_test,  y_test,  batch_size=32, shuffle=False)

    history2 = model2.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        verbose=1,
        shuffle=True
    )

    y_pred = model2.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))

    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model2.save("toxicityModel25small_peptidi_freezeReadout_folded.h5")


# =============== METODA 3 ===============
print("\n=== METODA 3: Zamrzavanje SVIH slojeva + novi izlazni sloj + 10-fold CV ===")
fold_index = 0
for train_index, test_index in cv.split(graphs, graph_labels):
    fold_index += 1
    print(f"\n--- Fold {fold_index} ---")

    base_model = load_pretrained_model()

    for layer in base_model.layers:
        layer.trainable = False

    # Dodajemo novi izlaz
    intermediate_output = base_model.layers[-2].output
    new_output = Dense(1, activation="sigmoid", name="new_output")(intermediate_output)

    model3 = Model(inputs=base_model.input, outputs=new_output)
    model3.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=binary_crossentropy,
        metrics=["accuracy"]
    )

    graphs_arr = np.array(graphs)
    labels_arr = np.array(graph_labels)

    X_train, X_test = graphs_arr[train_index], graphs_arr[test_index]
    y_train, y_test = labels_arr[train_index], labels_arr[test_index]

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    gen = PaddedGraphGenerator(graphs=graphs_arr)

    train_gen = gen.flow(X_train, y_train, batch_size=32, shuffle=True)
    val_gen   = gen.flow(X_val,   y_val,   batch_size=32, shuffle=False)
    test_gen  = gen.flow(X_test,  y_test,  batch_size=32, shuffle=False)

    history3 = model3.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        verbose=1,
        shuffle=True
    )

    y_pred = model3.predict(test_gen)
    y_pred = np.reshape(y_pred, (-1,))

    roc_auc_metric(y_test, y_pred)
    rest_of_metrics(y_test, y_pred)
    mcc_metric(y_test, y_pred)

model3.save("toxicityModel25small_peptidi_freezeAllNewOutput_folded.h5")

print("\n=== GOTOVO: Sve 3 metode odrađene s 10-fold CV i metrikama. ===")
