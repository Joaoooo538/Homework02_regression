import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

print("--- ETAPA 0: PREPARAÇÃO E PRÉ-PROCESSAMENTO ---")

# 1. CARREGAMENTO:

script_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(script_dir, 'winequality-red.csv')

try:
    df = pd.read_csv(file_path, sep=';') 

    print("Dados carregados com sucesso!")

except FileNotFoundError:

    print("ERRO: Arquivo csv não encontrado na pasta HW2.")

    exit()

X = df.drop('quality', axis=1)

Y = df['quality']

# Separamos 25% para teste
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras\n")

# 3. ASSIMETRIA
print("--- 3.1 Corrigindo Assimetria (Skewness) ---")

# Verificamos quais colunas são "tortas":

skew_antes = X_train.skew()

# Se for maior que 0.75, consideramos assimétrico:

colunas_assimetricas = skew_antes[abs(skew_antes) > 0.75].index

print(f"Colunas que receberão Log: {list(colunas_assimetricas)}")

# Aplicamos Log para "consertar" a distribuição:

#Precisa ser corrigido.......

for col in colunas_assimetricas:

    X_train[col] = np.log1p(X_train[col])

    X_test[col]  = np.log1p(X_test[col])

# 4. MATRIZ DE CORRELAÇÃO

print("\n--- 3.2 Visualizando Correlações ---")

plt.figure(figsize=(10, 8))

sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm', fmt=".2f")

plt.title("Matriz de Correlação (Dados de Treino)")

plt.show()

print("\n--- 3.3 Padronização (StandardScaler) ---")

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled  = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X.columns)

X_test  = pd.DataFrame(X_test_scaled, columns=X.columns)

print("Dados padronizados e prontos.")

print("-" * 50)