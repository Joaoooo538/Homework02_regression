# Primeiro análise e pré processamento, aplicação de PCA para redução de dimensionalidade e então aplicação do modelo
# Na primeira rodagem só se faz o modelo simples separando treino e teste usualmente (2/3 e 1/3)
# Separados em Treino e Teste, porém como vamos aplicar Cross validation, vamos dividir os K-folds logo, igualmente (5 a 10)
# Vamos fazer com 10 folders, o treino é feito com todos os folds, exceto o que virou teste na rodada
# Treino: [F1 F2 F3 F6] Teste: [F5],Treino: [F2 F3 F5 F6] Teste: [F1]...
# ✅ Para setores terminados e ❌ para incompletos


import pandas as pd
import numpy as np
from scipy.stats import boxcox
from sklearn.decomposition import PCA



# ✅ Carregamento do csv  =============================
database = pd.read_csv("winequality-red.csv", sep=";")
#print(database.columns)
#print(len(database))
print("\nDimensões do conjunto completo: ",database.shape) 
#=====================================================



# ✅ Shuffle pra embaralhar as entradas ================================================================
ind_shuffle = np.random.permutation(len(database))              #cria os indicies
database = database.iloc[ind_shuffle].reset_index(drop=True)    #embaralha e reseta a posicao do indice
#=======================================================================================================



# ✅ Separação de features e target ===========
x_database = database.drop(columns=["quality"])
y_database = database["quality"]
#==============================================



# ✅= Cálculo do Skewness pra ajeitar =
skew = x_database.skew()
#print(skew)
#=====================================



# ✅ Separação de treino e teste ========================================================================
n = int(np.floor(len(database)/3))   #um terço do tamanho arredondado pra baixo
data_train = database[:2*n]
data_test = database[2*n:]
#print("Tamanho do conjunto de treino: ",len(data_train),"Tamanho do conjunto de teste :",len(data_test))
#========================================================================================================



# ✅ Separação de features e target de treino e teste ==
y_train = y_database[:2*n]
x_train = x_database[:2*n]
print("\nDimensões dos preditores de treino: ",x_train.shape) 
y_test = y_database[2*n:]
x_test = x_database[2*n:]
print("\nDimensões dos preditores de teste: ",x_test.shape)
#======================================================


# ------------> ADICIONAR UM VIOLIN PLOT ANTES E DEPOIS <------------- 

# ✅ Ajuste para diminuir o skewness com Box-Cox ===================================================================================================
#GARANTIR QUE USE O BOXCOX NO TREINO, GUARDE OS LAMBDAS E APLIQUE ELES NO TESTE, FAZER TUDO JUNTO PODE VAZAR TENDENCIA DO 
# TREINO PRO TESTE ATRAVES DO PARAMETRO LAMBDA DE ACORDO COM A DISTRIBUIÇÃO DOS DADOS, JÁ É UM MODELO ESTATISTICO TREINADO
#  E NAO PODE SABER DA DISTRIBUIÇÃO DO CONJUNTO DE TESTES.
x_train_ajusted = x_train.copy()
lambdas = {}    # para aplicar no teste depois os mesmos lambdas no boxcox
shifts = {}     # para aplicar no teste depois os shifts no canto correto
for col in x_train_ajusted.columns:
    val = x_train_ajusted[col].values
    # Box-Cox precisa de valores > 0
    if np.any(val <= 0):                  # se tiver um valor negativo, ajeita pra positivo
        shift = abs(np.min(val)) + 1e-6   # calculo do shift para avançar todos os valores negativos e 0, maiores que 0
        val = val + shift                 # deslocamento necessario pra ficar positivo se tiver algum valor menor que zero nos valores da coluna
        shifts[col] = shift
    else:
        shifts[col] = 0.0

    transf_val, boxcox_lambdas = boxcox(val)    # salva os lambdas pra poder usar no conjunto de teste mais tarde e os valores transformados
    x_train_ajusted[col] = transf_val           # monta denovo o a matriz
    lambdas[col] = boxcox_lambdas               # grava os lambdas
# print("\nNovos skewness pós box-cox do conjunto de treino\n",x_train_ajusted.skew())
# para setar um lambda mais afrente no conjunto de teste usaremos transf_val = boxcox(val[col], lmbda=lambdas[col])
#==================================================================================================================================================



# ✅ (Z-SCORE) Inicio da aplicação do PCA para redução da dimensionalidade ======================================================
# Normalização média 0 e desvio padrao 1 para realizar o calculo correto da matriz de covariância.
mean_train = np.mean(x_train_ajusted, axis=0)
std_train = np.std(x_train_ajusted, axis=0)
x_train_ajusted_norm = (x_train_ajusted - mean_train) / std_train   #Padronização Z-score com a media e o desvio padrao.
# print (x_train_ajusted_norm)
# IMPORTANTE USAR OS VALORES DO MODELO DE TREINO DE STD E MEAN PARA O CONJUNTO DE TESTE QUANDO FOR FAZER O Z-SCORE DENOVO E O PCA.
#===============================================================================================================================



# ✅ Calculo do PCA nativo =====================================================================================================
# Com o calculo nativo nao precisamos passar a matriz de covariância, apenas os dados, ele calculará da mesma forma e fará a
# mesma checagem de posto.
pca = PCA()
pca.fit(x_train_ajusted_norm)           # PCA completo pra retorno de todos os PCS e 
print ("\n \nVariância explicada por componente principal", pca.explained_variance_ratio_.cumsum(),"\n\n")    # variância explicada cumulativa
# Escolha de 9 features para 97% de variância explicada aceitavel
n_pcs = 9
pca_reduzido = PCA(n_components=n_pcs)      # Determinação do PCA com retorno só dos 9 primeiros PC's
pca_reduzido.fit(x_train_ajusted_norm)  # Cálculo efetivo do PCA no conjunto com 9 PC's

x_train_pca = pca_reduzido.transform(x_train_ajusted_norm)          # Trasnformação do conjunto com os PC's escolhidos
print("\n Dimensionalidade de treino trasnf.: ",x_train_pca.shape)  # Dimensionalidade esperada: 1066(amostras de treino) x 9(PC's)
#================================================================================================================================


#=============================================🎯 INICIO DA FASE DE TREINO 🎯====================================================
# ✅ Regressão multivariada via Mínimos quadrados (OLS) =========================================================================
# Y = b0 + b1x1 + b2x2 ... + bnxn + e , sendo x cada uma das features
# ou seja temos 9 + 1 colunas, contando com o b0, adicionamos entao essa coluna
x_train_pca_bias = np.column_stack([np.ones(x_train_pca.shape[0]), x_train_pca])  #adiciona uma coluna de 1's para representar o b0

# aplicação da fórmula do OLS [(Xt*X)^-1 * Xt * Y]
betas = np.linalg.inv(  x_train_pca_bias.T     @       x_train_pca_bias )     @       (x_train_pca_bias.T    @   y_train)
print("Betas resultantes: ",betas)      # Esperados 10 betas

# cálculo do erro médio quadrado (RMSE) de validação no treino sqrt(   mean( (Y - Ypred)^2 )  )
y_predict_train =  x_train_pca_bias @ betas
MSE_train = np.sqrt(np.mean((y_train - y_predict_train)**2))
print("\nRMSE de treino:", MSE_train)

# Cálculo do R^2
y_mean_train = np.mean(y_train)
r2_train = 1 - np.sum((y_train - y_predict_train)**2) / np.sum((y_train - y_mean_train)**2) #o sum vai somar os numeradores entre si e os denominadores
print("\nR2 de treino:", r2_train)

#[np.linalg.inv: é a inversa no python]
#[      @      : é o multiplicador matricial do python]
#=================================================================================================================================



#=============================================🧪 INICIO DA FASE DE TESTES 🧪=====================================================
# ✅ Adaptação do conjunto de testes e aplicação do modelo =======================================================================
# Aplicando todas as trasnformações no conjunto de testes denovo com os todos os parametros obtidos do TREINO (media, dp, lambdas, 
# PCA, shifts e betas)

#boxcox com lambdas salvos
x_test_ajusted = x_test.copy()
for i in range(x_test.shape[1]):
    col = x_test.columns[i]
    val_test = x_test_ajusted.values[:,i] + shifts[col]                   #soma os shifts iguais aos do treino
    x_test_ajusted[col] = boxcox(val_test, lmbda=lambdas[col])  #aplica o boxcox com os lambdas iguais aos do treino
    # Utilizar o mesmo shift que no treino pra nao dar data-leakage, checar se nao 
    # fica nada negativo, caso fique aumentar a parcela de soma fixa do shift no treino
    
# Z-score com mean e std salvos
x_test_ajusted_norm = (x_test_ajusted - mean_train) / std_train     #os mesmos do treino pra evitar o leakage

# PCA (aplica a mesma matriz do treino, apenas transforma os dados direto)
x_test_pca = pca_reduzido.transform(x_test_ajusted_norm)

# Bias (nada muda)
x_test_pca = np.column_stack([np.ones(x_test_pca.shape[0]), x_test_pca])

# Aplicação do modelo para validação e (MSE) de validação no teste
y_predict_test = x_test_pca @ betas
MSE_test = np.sqrt(np.mean((y_test - y_predict_test)**2))
print("\nRMSE de teste :", MSE_test)

#(R2)
y_mean_test = np.mean(y_test)
r2_test = 1 - np.sum((y_test - y_predict_test)**2) / np.sum((y_test - y_mean_test)**2)
print("\nR2 de teste:", r2_test)

#Sobre as estatísticas
#Treino baixo + Teste baixo -> ✅ BOM
#Treino baixo + Teste alto  -> ❌ OVERFITTING
#Treino alto + Teste alto   -> ❌ UNDERFITTING
#Treino alto + Teste baixo  -> (erro no código)
#Modelo atual puxa sempre pro "Meio", nao tem complexidade suficiente pra se adequar as nao liearidades dos dados.
#=================================================================================================================================


# COMPARACAO MANUAL DAS PREDIÇÕES
#comparacao = pd.DataFrame({
#    "y_real": y_test.values[:10].flatten(),
#    "y_pred": y_predict_test[:10].flatten()
#})
#print(comparacao)


#modelo pra implementação futura

#=============================== K-fold implementation ============================
#from sklearn.model_selection import KFold
#
#kf = KFold(n_splits=5, shuffle=True, random_state=42)
#
#for train_idx, test_idx in kf.split(X):
#    X_train, X_test = X[train_idx], X[test_idx]
#    y_train, y_test = y[train_idx], y[test_idx]
#
#    # treina o modelo
#==================================================================================