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

# =============================================================================
# 4. REDE NEURAL (MLP)
# =============================================================================
# Mudanças baseadas em Grid Search para evitar Overfitting e bater a Linear:
# 1. activation='tanh': Curvas mais suaves
# 2. alpha=1.0: Penalidade alta para evitar decorar dados
# 3. learning_rate_init=0.01: Convergência mais rápida

mlp_tunada = MLPRegressor(
    hidden_layer_sizes=(64, 32),       
    activation='tanh',                 
    solver='adam',
    alpha=1.0,                         
    learning_rate_init=0.01,           
    max_iter=5000,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)

mlp_tunada.fit(x_train_ajusted_norm, y_train)

# Predições
y_pred_train = mlp_tunada.predict(x_train_ajusted_norm)
y_pred_test = mlp_tunada.predict(x_test_ajusted_norm)

# Métricas
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
r2_train = r2_score(y_train, y_pred_train)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_test = r2_score(y_test, y_pred_test)

print("--- RESULTADOS FINAIS ---")
print(f"RMSE Treino: {rmse_train:.5f}")
print(f"R² Treino:   {r2_train:.5f}")
print("-" * 30)
print(f"RMSE Teste:  {rmse_test:.5f}")
print(f"R² Teste:    {r2_test:.5f}")

print("\n--- Tabela Comparativa (y_real vs y_pred) ---")
comparacao = pd.DataFrame({
    "y_real": y_test.values,
    "y_pred": y_pred_test
})

print(comparacao.head(20))