# Neural_Network_regression.py
# Foco: Rede Neural (MLP) - Item 4 do HW

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS (Cópia exata para garantir consistência)
# ==============================================================================
print("--- CARREGANDO DADOS PARA REDE NEURAL ---")
database = pd.read_csv("winequality-red.csv", sep=";")

# 🚨 IMPORTANTE: Mesma semente do OLS e Lasso
np.random.seed(42) 

# Shuffle
ind_shuffle = np.random.permutation(len(database))         
database = database.iloc[ind_shuffle].reset_index(drop=True)

# Separação
x_database = database.drop(columns=["quality"])
y_database = database["quality"]

# Treino e Teste
n = int(np.floor(len(database)/3))
y_train = y_database[:2*n]
x_train = x_database[:2*n]
y_test = y_database[2*n:]
x_test = x_database[2*n:]

# Box-Cox
x_train_ajusted = x_train.copy()
lambdas = {}
shifts = {}
for col in x_train_ajusted.columns:
    val = x_train_ajusted[col].values
    if np.any(val <= 0):             
        shift = abs(np.min(val)) + 1e-6
        val = val + shift
        shifts[col] = shift             
    else:
        shifts[col] = 0.0
    transf_val, boxcox_lambda = boxcox(val)
    x_train_ajusted[col] = transf_val   
    lambdas[col] = boxcox_lambda     

# Z-Score
mean_train = np.mean(x_train_ajusted, axis=0)
std_train = np.std(x_train_ajusted, axis=0)
x_train_ajusted_norm = (x_train_ajusted - mean_train) / std_train

# Aplicação no Teste
x_test_ajusted = x_test.copy()
for col in x_test.columns:
    val_test = x_test_ajusted[col].values + shifts[col]
    x_test_ajusted[col] = boxcox(val_test, lmbda=lambdas[col])

x_test_ajusted_norm = (x_test_ajusted - mean_train) / std_train

print("Dados preparados. Iniciando Rede Neural...\n")

# ==============================================================================
# 2. MODELO REDE NEURAL
# ==============================================================================

print("--- REDE NEURAL (MLP Regressor) ---")

# Configuração: 1 camada oculta com 100 neurônios, max_iter alto (5000)
mlp_model = MLPRegressor(hidden_layer_sizes=(100,), 
                         activation='relu', 
                         solver='adam', 
                         max_iter=5000, 
                         random_state=42)

mlp_model.fit(x_train_ajusted_norm, y_train)

# Previsão
y_pred_mlp = mlp_model.predict(x_test_ajusted_norm)

# Métricas
rmse_mlp = np.sqrt(mean_squared_error(y_test, y_pred_mlp))
r2_mlp = r2_score(y_test, y_pred_mlp)

print(f"RMSE Rede Neural: {rmse_mlp:.5f}")
print(f"R² Rede Neural:   {r2_mlp:.5f}")