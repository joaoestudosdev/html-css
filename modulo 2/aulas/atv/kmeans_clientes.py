# Importação das bibliotecas necessárias
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Criando a base de dados manualmente com as informações dos clientes
dados = {
    "idade": [19,21,22,23,24,25,26,27,28,29,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,69],
    "gasto_mensal": [120,150,180,210,250,270,300,320,350,370,400,420,450,480,520,550,600,630,680,720,760,800,850,900,950,1000,1050,1100,1150,1200],
    "frequencia_compras_mes": [8,7,6,6,5,5,4,4,4,3,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1]
}

# Transformando os dados em um DataFrame
df = pd.DataFrame(dados)

# Selecionando apenas as variáveis que serão usadas no K-Means
X = df[["idade", "gasto_mensal", "frequencia_compras_mes"]]

# Definindo o número de clusters (neste caso 3 grupos)
kmeans = KMeans(n_clusters=3, random_state=42)

# Aplicando o algoritmo nos dados
kmeans.fit(X)

# Criando uma nova coluna no DataFrame com o número do cluster de cada cliente
df["cluster"] = kmeans.labels_

# Mostrando a tabela final com os grupos encontrados
print(df)

# Criando um gráfico para visualizar os clusters
plt.scatter(df["gasto_mensal"], df["frequencia_compras_mes"], c=df["cluster"])

# Nome dos eixos do gráfico
plt.xlabel("Gasto mensal")
plt.ylabel("Frequência de compras no mês")

# Título do gráfico
plt.title("Segmentação de clientes usando K-Means")

# Exibindo o gráfico
plt.show()