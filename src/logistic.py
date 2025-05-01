import torch
import tenseal as ts
import pandas as pd
import random
import numpy as np
import os
import matplotlib.pylab as plt 

from time import time

# Fixer les graines pour la reproductibilité
torch.random.manual_seed(73)
random.seed(73)

# ============================== #
# Chargement des données
# ============================== #

DATA_PATH = os.path.join('static', 'data', 'framingham.csv')

def load_data():
    data = pd.read_csv(DATA_PATH, delimiter=',')
    data = data.dropna()
    data = data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI"])
    
    grouped = data.groupby('TenYearCHD')
    data = grouped.apply(lambda x: x.sample(grouped.size().min(), random_state=73)).reset_index(drop=True)
    
    y = torch.tensor(data["TenYearCHD"].values).float().unsqueeze(1)
    data = data.drop("TenYearCHD", axis='columns')
    
    data = (data - data.mean()) / data.std()
    x = torch.tensor(data.values).float()
    
    return split_train_test(x, y)

def split_train_test(x, y, test_ratio=0.3):
    idxs = [i for i in range(len(x))]
    random.shuffle(idxs)
    delim = int(len(x) * test_ratio)
    test_idxs, train_idxs = idxs[:delim], idxs[delim:]
    return x[train_idxs], y[train_idxs], x[test_idxs], y[test_idxs]

# ============================== #
# Modèle de régression logistique
# ============================== #

class LR(torch.nn.Module):
    def __init__(self, n_features):
        super(LR, self).__init__()
        self.lr = torch.nn.Linear(n_features, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.lr(x))

def train(model, optim, criterion, x, y, epochs=10):
    for e in range(1, epochs + 1):
        optim.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        print(f"Loss at epoch {e}: {loss.data}")
    return model

def accuracy(model, x, y):
    out = model(x)
    correct = torch.abs(y - out) < 0.5
    return correct.float().mean()


# ============================== #
# Fonctions pour normaliser un échantillon
# ============================== #

# Pour normaliser un nouveau patient
def get_mean_std():
    raw_data = pd.read_csv(DATA_PATH, delimiter=',')
    raw_data = raw_data.dropna()
    raw_data = raw_data.drop(columns=["education", "currentSmoker", "BPMeds", "diabetes", "diaBP", "BMI", "TenYearCHD"])
    mean = raw_data.mean()
    std = raw_data.std()
    return mean, std

def donnee_standards(sample):
    mean, std = get_mean_std()
    standardized = [(s - m) / st for s, m, st in zip(sample, mean, std)]
    return torch.tensor(standardized, dtype=torch.float32)

# ============================== #
# Script principal de la prediction sur les données clairs
# ============================== #
torch.random.manual_seed(73)
random.seed(73)
x_train, y_train, x_test, y_test = load_data()
n_features = x_train.shape[1]

#print(x_train)

# On cree notre model et son evaluation 
model = LR(n_features)   
optim = torch.optim.SGD(model.parameters(), lr=1)
criterion = torch.nn.BCELoss()
model = train(model, optim, criterion, x_train, y_train)

# On donne la precision de predilection de notre modele sur les données claires
precision = accuracy(model, x_test, y_test)
print(f"La precision des performances du modeles sur les claires est : {precision}")


#print(precision)
#################### Affiche pour les donnés claires  ##########################

# def affiche(model, x_test, y_test):
#     # Prédiction des valeurs avec le modèle
#     predit_y = model(x_test).detach().numpy()  # Convertir les prédictions en un tableau NumPy pour l'affichage
    
#     plt.figure(figsize=(8, 5))

#     # Tracer les vraies valeurs
#     plt.scatter(range(len(y_test)), y_test, color='blue', label='Vraies valeurs', alpha=0.6)
    
#     # Tracer les prédictions
#     plt.scatter(range(len(predit_y)), predit_y, color='red', label='Prédictions', alpha=0.6, marker='x')
    
#     # Ajouter des labels et un titre
#     plt.xlabel('Index')
#     plt.ylabel('Charges')
#     plt.title('Comparaison entre les vraies valeurs et les prédictions')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
    
#     # Afficher le graphique
#     plt.show()

# ==========================================================#
#   Modele de la regresssion logistiques des des chiffrés   #
#===========================================================#


# Creation des parametres du CKKS pour le chiffrement
# Paramètres de CKKS
poly_mod_degree = 4096
coeff_mod_bit_sizes = [40, 20, 40]

# Créer le contexte CKKS
ctx_eval = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
ctx_eval.global_scale = 2 ** 20  # Définir l'échelle globale
ctx_eval.generate_galois_keys()  # Générer les clés Galois pour les opérations


# def encryp_list(dat):
#     return [ts.ckks_vector(ctx_eval, x.tolist()) for x in dat]

# def decrypt_list(data_enc):
#     return [vec.decrypt() for vec in data_enc] 

def encrypt(vec):
    return ts.ckks_vector(ctx_eval, vec)

def decrypt(enc_vec):

    valeur_dech_decimal = enc_vec.decrypt()
    return [round(value, 2) for value in valeur_dech_decimal]

# Définir la classe pour le modèle chiffré
class EncryptedLR:
    def __init__(self, torch_lr):
        # Prendre les poids et biais depuis le modèle PyTorch
        self.weight = torch_lr.lr.weight.data.tolist()[0]
        self.bias = torch_lr.lr.bias.data.tolist()

    def forward(self, enc_x):
        # Forward sur données chiffrées
        enc_out = enc_x.dot(self.weight) + self.bias
        return enc_out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def encrypt(self, context):
        # Chiffrer uniquement si ce n'est pas déjà chiffré
        if isinstance(self.weight, list):
            self.weight = ts.ckks_vector(context, self.weight)
        if isinstance(self.bias, list):
            self.bias = ts.ckks_vector(context, self.bias)

    def decrypt(self):
        # Déchiffrer uniquement si ce sont des vecteurs chiffrés
        if isinstance(self.weight, ts.CKKSVector):
            self.weight = self.weight.decrypt()
        if isinstance(self.bias, ts.CKKSVector):
            self.bias = self.bias.decrypt()

eelr = EncryptedLR(model)

# Precision de predilection suivant les données chiffrées

def encryp_list(dat):
    return [ts.ckks_vector(ctx_eval, x.tolist()) for x in dat]

def decrypt_list(data_enc):
    return [vec.decrypt() for vec in data_enc] 

def encrypted_evaluation(model, enc_x_test, y_test):
    t_start = time()
    
    correct = 0
    for enc_x, y in zip(enc_x_test, y_test):
        # encrypted evaluation
        enc_out = model(enc_x)
        # plain comparison
        out = enc_out.decrypt()
        out = torch.tensor(out)
        out = torch.sigmoid(out)
        if torch.abs(out - y) < 0.5:
            correct += 1
    
    t_end = time()
    
    return correct / len(x_test)

enc_x_test = [ts.ckks_vector(ctx_eval, x.tolist()) for x in x_test]
enc_y_test = encryp_list(y_test)
#print(x_test[:10])
#print(enc_x_test[:10])
#print(decrypt_list(enc_y_test[:10]))
    
precision_chiffre = encrypted_evaluation(eelr, enc_x_test, y_test)

difference = abs(precision_chiffre - precision)

print(f"La precision des performances sur les chiffrés: {precision_chiffre}")