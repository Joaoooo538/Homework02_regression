# Lasso_regression.py
# Foco: Regularização L1 e Seleção de Variáveis

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================================================
# 1. PREPARAÇÃO DOS DADOS (Cópia exata do Native_OLS para garantir consistência)
# ==============================================================================
print("--- CARREGANDO DADOS PARA LASSO ---")
database = pd.read_csv("winequality-red.csv", sep=";")

# 🚨 IMPORTANTE: Semente fixa para o embaralhamento ser IGUAL ao do OLS
np.random.seed(42) 

# Shuffle
ind_shuffle = np.random.permutation(len(database))         
database = database.iloc[ind_shuffle].reset_index(drop=True)

# Separação
x_database = database.drop(columns=["quality"])
y_database = database["quality"]

# Treino e Teste (1/3 split)
n = int(np.floor(len(database)/3))
y_train = y_database[:2*n]
x_train = x_database[:2*n]
y_test = y_database[2*n:]
x_test = x_database[2*n:]

# Box-Cox (Treino)
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

# Z-Score (Treino)
mean_train = np.mean(x_train_ajusted, axis=0)
std_train = np.std(x_train_ajusted, axis=0)
x_train_ajusted_norm = (x_train_ajusted - mean_train) / std_train

# Aplicação no Teste (Usando parâmetros do Treino)
x_test_ajusted = x_test.copy()
for col in x_test.columns:
    val_test = x_test_ajusted[col].values + shifts[col]
    x_test_ajusted[col] = boxcox(val_test, lmbda=lambdas[col])

x_test_ajusted_norm = (x_test_ajusted - mean_train) / std_train

print("Dados preparados. Iniciando Lasso...\n")

# ==============================================================================
# 2. MODELO LASSO (O Código Novo)
# ==============================================================================

print("--- LASSO REGRESSION ---")

# LassoCV faz a validação cruzada automática (cv=5) para achar o melhor alpha
lasso_model = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_model.fit(x_train_ajusted_norm, y_train)

# Previsão
y_pred_lasso = lasso_model.predict(x_test_ajusted_norm)

# Métricas
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"Melhor Alpha (Penalidade): {lasso_model.alpha_:.6f}")
print(f"RMSE Lasso: {rmse_lasso:.5f}")
print(f"R² Lasso:   {r2_lasso:.5f}")

# Análise de Variáveis Zeradas (Obrigatório no HW)
print("\n>>> Variáveis eliminadas pelo Lasso:")
colunas_originais = x_database.columns
coeficientes = pd.Series(lasso_model.coef_, index=colunas_originais)

zerados = coeficientes[coeficientes == 0]
if len(zerados) > 0:
    print(f"O Lasso zerou {len(zerados)} variáveis: {list(zerados.index)}")
else:
    print("Nenhuma variável foi zerada.")