import numpy as np
import tenseal as ts
from sklearn.datasets import load_iris
import time
import matplotlib.pyplot as plt
import tenseal as ts

class LinearRegression:
    def __init__(self):
          self.beta1 = None
          self.beta0 = None

    def __repr__(self):
        return f"f(x) = {self.beta0} + {self.beta1}*x"

    def fit(self,x,y):
 
        assert len(x)==len(y)
        x_avg = sum(x)/len(x)
        y_avg = sum(y)/len(y)

        xy_error = (x[0]-x_avg)*(y[0]-y_avg)

        SE = (x[0]-x_avg)**2
        for i in range(1,len(x)):
            SE += (x[i]-x_avg)**2
            xy_error += (x[i]-x_avg)*(y[i]-y_avg)
        
        self.beta1 = xy_error/SE
        self.beta0 = y_avg - self.beta1*x_avg

    def predict(self, x1):
        return self.beta0 + self.beta1*x1
    
    def get_coeff(self):
        return self.beta0, self.beta1

# Générer des prédictions pour des courbes

# contexte de chiffrement 
# initialisation des nombres de multipplications
multiplication = 1 
inner = 30
outer = 60

# Parametres Tenseal 
# il a prendre en compte que ce qui sont a l'interieur de 
# context ne doivent pas changé de nom)
# on peut changer que les valeurs des variables 
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree = 8192,
    coeff_mod_bit_sizes = [outer] + [inner] * multiplication + [outer]
)
context.global_scale = pow(2, inner)
# nos données qu'on va evaluer la precision
iris = load_iris()

x = iris.data[:, 0]
y = iris.data[:, 2]

model = LinearRegression()
model.fit(x,y)



def Precision_lr_clair(model, x_clair, y_clair):
    out = model.predict(x_clair)
    correct = abs(y_clair - out) <0.5
    return correct.mean()
    #return f" La prediction de nos données claires : {out[:5]}"

# Le chiffré qu'on a utilisé 
encrypted_X = ts.ckks_vector(context,x)

# La precision de notre modele sur les chiffrées
def Precision_lr_chiff(model, enc_x, y_clair):
    
    # prediction de notre chiffre
    encrypted_predicted_y = model.predict(enc_x)
    # dechiffroons la predictions
    dechiff_predicted_y = encrypted_predicted_y.decrypt()
    # On convertis en classe numpy 
    #dechiff_predicted_y = np.array(dechiff_predicted_y) 
   
    
    correct = abs(y_clair - dechiff_predicted_y ) <0.5
    return correct.mean()
    #return  f"données predits sur les chiffrés après dechiffrement : {dechiff_predicted_y[:5]}"
    #return dechiff_predicted_y



precision_des_claires = Precision_lr_clair(model, x, y)
precision_des_claires
precision_des_chiffres= Precision_lr_chiff(model, encrypted_X, y)
differ_de_precision = abs(precision_des_chiffres - precision_des_claires)

#print(precision_des_claires)
#print(precision_des_chiffres)
# print(f'Les donnees de x : {x}')
# print(f'les données de y : {y}')