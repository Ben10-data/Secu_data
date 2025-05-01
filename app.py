from flask import Flask, render_template, request
from sklearn.datasets import load_iris
from src import regression_line
import tenseal as ts
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import uuid
import pandas as pd
from src import logistic
import torch

app = Flask(__name__)

# Chemin absolu vers static/
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static", "data_linear")
os.makedirs(STATIC_DIR, exist_ok=True)
# chemin vers le dossier data 
csv_path = os.path.join(app.root_path, "static", "data", "framingham.csv")
df = pd.read_csv(csv_path)


# Charger les données et entraîner le modèle
iris = load_iris()
x = iris.data[:, 0]
y = iris.data[:, 2]

model = regression_line.LinearRegression()
model.fit(x, y)

# Contexte CKKS
multiplication = 1
inner = 30
outer = 60
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[outer] + [inner] * multiplication + [outer]
)
context.global_scale = 2 ** inner
@app.route("/")
def index():
	return render_template("index.html")

@app.route("/dashlayout")
def dash_layout():
	return render_template('base.html')

@app.route('/dashlayout/iris', methods=['GET', 'POST'])
def iris():
    predict_clair = decrypted_predict_y = None
    temp_pris_clair = temps_chiffre = None
    errors_img = enc_preds_img = preds_img = None
    erreur_de_prediction = None 
    texte_clair = text_chiffre = None 
    performance_du_modele_claire = performance_du_modele_chiffre = None
    diff_performance = None

    if request.method == 'POST':
        vect = request.form.get('a', type=float)
        # le texte claires
        clairs = np.array([vect])
        #notre chiffré
        encrypted_X = ts.ckks_vector(context, clairs)
        
        # le texte claire 
        texte_clair = vect 
        # le texte chiffré 
        text_chiffre = encrypted_X

        # Prédiction sur les donnés chiffrées 
        encrypted_predict_y = model.predict(encrypted_X)
        decrypted_predict_y = encrypted_predict_y.decrypt()[0]

        # Prediction sur les données clairs
        predict_clair = model.predict(clairs)[0]

        #difference d'erreurs de prediction 
        erreur_de_prediction = abs(decrypted_predict_y - predict_clair)

        # Mesure des temps
        N = 10000
        t = time.time()
        for _ in range(N): model.predict(clairs)[0]
        temp_pris_clair = time.time() - t

        t = time.time()
        for _ in range(N): model.predict(encrypted_X)
        temps_chiffre = time.time() - t

        # Générer des prédictions pour des courbes
        xs = np.array(range(-5000, 5001, 10)) / 1000
        enc_X = ts.ckks_vector(context, xs)
        enc_preds = model.predict(enc_X).decrypt()
        preds = model.predict(xs)
        errors = np.array(enc_preds) - np.array(preds)
        # performance du modele sur les claires et les chiffrés
        performance_du_modele_claire = regression_line.precision_des_claires
        performance_du_modele_chiffre = regression_line.precision_des_chiffres
        diff_performance = regression_line.differ_de_precision

        # Identifiant unique
        uid = str(uuid.uuid4())
        errors_img = f"errors_{uid}.png"
        enc_preds_img = f"enc_preds_{uid}.png"
        preds_img = f"preds_{uid}.png"

        # Sauvegarde des images
        plt.figure()
        plt.plot(xs, errors)
        plt.title("Erreur")
        plt.savefig(os.path.join(STATIC_DIR, errors_img))
        plt.close()

        plt.figure()
        plt.plot(xs, enc_preds)
        plt.title("Prédictions Chiffrées")
        plt.savefig(os.path.join(STATIC_DIR, enc_preds_img))
        plt.close()

        plt.figure()
        plt.plot(xs, preds)
        plt.title("Prédictions sur les données claires")
        plt.savefig(os.path.join(STATIC_DIR, preds_img))
        plt.close()

    return render_template("iris.html",
                           predict_clair=predict_clair,
                           decrypted_predict_y=decrypted_predict_y,
                           temp_pris_clair=temp_pris_clair,
                           temps_chiffre=temps_chiffre,
                           errors_img=errors_img,
                           enc_preds_img=enc_preds_img,
                           preds_img=preds_img, erreur_de_prediction = erreur_de_prediction,
                           texte_clair = texte_clair, text_chiffre = text_chiffre,
                           performance_du_modele_claire = performance_du_modele_claire,
                           performance_du_modele_chiffre = performance_du_modele_chiffre,
                           diff_performance = diff_performance)


@app.route('/dashlayout/logistique', methods=['GET', 'POST'])
def logistique():
    model = logistic.model
    normaliser_vecteur = logistic.donnee_standards

    prediction_clair = None  
    temps_mis_clair = None
    prediction_chiffre = None
    temps_dechiffre = None
    erreur_de_prediction = None
    Patient = None
    enc_patient = None
    precision_performance_clairs = None 
    precision_performance_chiffre = None
    difference = None 

    if request.method == 'POST':
        sexe = int(request.form['sexe'])
        age = int(request.form['age'])
        cigare = int(request.form['cigare'])
        avc = int(request.form['avc'])
        hyper = int(request.form['hyper'])
        colesterol = int(request.form['colesterol'])
        frequence0 = int(request.form['frequence0'])
        frequence1 = int(request.form['frequence1'])
        glucose = int(request.form['glucose'])
        
        # Créer la liste du patient
        Patient = [
            sexe,
            age,
            cigare,
            avc,
            hyper,
            colesterol,
            frequence0,
            frequence1,
            glucose
        ]
        
        # Normalisation du patient
        vect = normaliser_vecteur(Patient)
        
        #==============================================================#
        #     Nous travaillons ici avec les données en clairs  #
        ##=========================================================#

        t = time.time()
        # Prédiction sur le patient normalisé
        y_prevision_Patient = model(vect.unsqueeze(0))
        prediction_clair = y_prevision_Patient.item()
        temps_mis_clair = time.time() -t

        # Precision de la performance de notre modele 
        precision_performance_clairs = logistic.precision.item()

        


        #===================================================================#
        # Ici nous allons travailler avec les données chiffrées #
        #===============================================================#

        t1 = time.time()

        # Chiffrement de nos ddonnées Patient 
        # Nous allons en premier temps, importer nos parametres de chiffrement
        parametres_ckks = logistic.ctx_eval
        # Chiffrons les données de notre patient 
        enc_patient =  ts.ckks_vector(parametres_ckks, vect.tolist())
        # 
        # Nous allons import notre modele 
        model_lo = logistic.eelr
        # Nous allonns faire une prediction sur nos données chiffré 
        chiffre_predict = model_lo(enc_patient)
        dechiffe_predict = chiffre_predict.decrypt()
        prediction_chiffre = torch.sigmoid(torch.tensor(dechiffe_predict))
        temps_dechiffre = time.time() - t1
        
        # Calcul de l'erreur de prédiction
        erreur_de_prediction = abs(prediction_clair - prediction_chiffre.item())

        # Precision de la performance 
        precision_performance_chiffre = logistic.precision_chiffre
        difference = logistic.difference

    return render_template('logistique_home.html', prediction_patient_clair=prediction_clair,
                           temps_mis_clair = temps_mis_clair,
                           Patient = Patient, 
                           enc_patient = enc_patient,
                           prediction_chiffre = prediction_chiffre,
                           temps_dechiffre = temps_dechiffre,
                           erreur_de_prediction = erreur_de_prediction,
                           precision_performance_clairs = precision_performance_clairs,
                           precision_performance_chiffre = precision_performance_chiffre,
                           difference = difference)

@app.route('/dashlayout/visu_lin')
def visu_lin():
	return render_template('visu_linear.html')

@app.route('/dashlayout/visu_log')
def visu_log():
    return render_template('visu_logistique.html')


if __name__ == '__main__':
	app.run(debug=True)