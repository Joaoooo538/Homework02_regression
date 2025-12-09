# Primeiro análise e pré processamento e então aplicação do modelo PCR implemnetado a mão
# Vamos fazer com 7 folders, o treino é feito com todos os folds, exceto o que virou teste na rodada
# Normalizações Z-score e boxcox foram feitas pra colocar na mesma escala e diminuir o skewness
# ✅ Para setores terminados e ❌ para incompletos

import os
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

#COM PCA

# ✅ Carregamento do csv  =============================
database = pd.read_csv("winequality-red.csv", sep=";")
#print(database.columns)
#print(len(database))
print("\nDimensões do conjunto completo: ",database.shape) 
#=====================================================



# ✅ Separação de features e target ===========
x_database = database.drop(columns=["quality"])
y_database = database["quality"]
#==============================================



# ✅ Configuração como as folds vao se dividr ====================================================================
k = 7                                                     # Número de Folds
# Kfolds com embaralhamento, dividido em k vezes 
k_folds = KFold(n_splits=k, shuffle=True)       # Criação do objeto KFold e dos indices train_index e test_index
rmse_folds = []                                 # Variaveis pra armazenar o resultado de erro de cada k fold (RMSE)
r2_folds = []                                   # (R^2)

#Retirar apenas os valores do objeto dataframe
X = x_database.values
y = y_database.values
#==================================================================================================================



# ✅ Configuração como as folds vao se dividr=====================================================================================
for train_index, test_index in k_folds.split(X):        # K-fold cria os indices train_index e test_index em formato array
    X_train, X_test = X[train_index], X[test_index]     # separa o array X em treino e teste tmb arrays
    y_train, y_test = y[train_index], y[test_index]     # Separa o array y em treino e teste

    
    
    # ============================= ✨ PRE PROCESSAMENTO DENTRO DE CADA CONFIGURAÇÃO DE FOLDS ✨ ================================
    # BOX-COX ---------------------------------------------------------------------------------------------------------------------
    X_train_df = pd.DataFrame(X_train, columns=x_database.columns)  # Remonta o df com as colunas dos features da original
    X_train_ajusted = X_train_df.copy()                             # O novo é a copia do dataframe, mesmas dimensoes
    lambdas = {}
    shifts = {}
    for col in X_train_df.columns:
        val = X_train_df[col].values                                # Pega denovo os valores só da coluna
        if np.any(val <= 0):                                        # Shitif caso necessario e guarda depois
            shift = abs(np.min(val)) + 1e-6
            val = val + shift
            shifts[col] = shift
        else:
            shifts[col] = 0.0
        transf_val, boxcox_lambda = boxcox(val)                     # Boxcox que recebe valores  e guarda o lambda e o val trasnf.
        X_train_ajusted[col] = transf_val                           # Joga os valores pra cópia
        lambdas[col] = boxcox_lambda                                # Guarda os lambda pra cada rodagem
    
    # Z-SCORE ---------------------------------------------------------------------------------------------------------------------
    mean_train = np.mean(X_train_ajusted, axis=0)                   # Padrão
    std_train = np.std(X_train_ajusted, axis=0)                     # lembrar de usar os mesmos std e mean no fim
    X_train_norm = (X_train_ajusted - mean_train) / std_train

    # PCA -------------------------------------------------------------------------------------------------------------------------
    n_pcs = 9
    pca = PCA(n_components=n_pcs)
    X_train_pca = pca.fit_transform(X_train_norm)
    
    #=============================================🎯 INICIO DA FASE DE TREINO 🎯=====================================================
    X_train_pca_bias = np.column_stack([np.ones(X_train_pca.shape[0]), X_train_pca])                # Adiciona o intercepto
    betas = np.linalg.inv(X_train_pca_bias.T @ X_train_pca_bias) @ (X_train_pca_bias.T @ y_train)   # Usa o modelo na mão
    #=================================================================================================================================



    #=============================================🧪 INICIO DA FASE DE TESTES 🧪=====================================================
    # Preparar X_test com os parâmetros do treino ---
    X_test_df = pd.DataFrame(X_test, columns=x_database.columns)
    X_test_ajusted = X_test_df.copy()
    for col in X_test_df.columns:
        val_test = X_test_df[col].values + shifts[col]
        X_test_ajusted[col] = boxcox(val_test, lmbda=lambdas[col])
    
    X_test_norm = (X_test_ajusted - mean_train) / std_train
    X_test_pca = pca.transform(X_test_norm)
    X_test_pca_bias = np.column_stack([np.ones(X_test_pca.shape[0]), X_test_pca])
    
    # --- Previsão e métricas ---
    y_pred = X_test_pca_bias @ betas
    rmse_fold = np.sqrt(np.mean((y_test - y_pred)**2))
    r2_fold = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    rmse_folds.append(rmse_fold)
    r2_folds.append(r2_fold)
    #==================================================================================================================================

# --- Resultados finais ---
print("RMSE médio:", np.mean(rmse_folds))
print("R² médio:", np.mean(r2_folds))
