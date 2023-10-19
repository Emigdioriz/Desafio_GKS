import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

df_raw = pd.read_csv('sti_movemento1.csv')

#TESE GERAL

df = df_raw
# Codificar variáveis categóricas usando One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
categorical_columns = ['UF', 'Nacionalidade', 'Classificacao']
encoded_data = encoder.fit_transform(df[categorical_columns])

# Criar um novo DataFrame com as variáveis codificadas
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

# Concatenar o DataFrame codificado com as colunas 'Pessoas' 
X = pd.concat([encoded_df, df[['Pessoas']]], axis=1)
y = df['Pessoas']

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o modelo de regressão linear
model = LinearRegression()

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nResultados para a Tese Geral \n')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

if mse < 0.1 and r2 > 0.9:
    print("A predição foi excelente!")
elif mse < 0.5 and r2 > 0.7:
    print("A predição foi boa.")
else:
    print("A predição não atendeu aos critérios desejados.")

print('\n######################################################################################')

# TESE DA ARENTINA

#retificando quaisquer espaços vazios que tenham nos dados
for coluna in df_raw.columns:
    if df_raw[coluna].dtype == 'object':
        df_raw[coluna] = df_raw[coluna].str.strip()

uf_filtrado = ['RS', 'SC', 'PR', 'SP']
tipo_filtrado = ['Entrada']
classificacao_filtrada = ['Permanente']
nacionalidade_filtrada = ['Argentina']

# filtrar o dataset original para as variáveis desejadas para o estudo
df1 = df_raw[(df_raw['UF'].isin(uf_filtrado)) & (df_raw['Tipo'].isin(tipo_filtrado)) & (df_raw['Classificacao'].isin(classificacao_filtrada)) & (df_raw['Nacionalidade'].isin(nacionalidade_filtrada)) ]


encoder = OneHotEncoder(sparse=False)
categorical_columns = ['UF', 'Nacionalidade', 'Classificacao']
encoded_data = encoder.fit_transform(df1[categorical_columns])


encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))


encoded_df = encoded_df.reset_index(drop=True)
df1 = df1.reset_index(drop=True)

X = pd.concat([encoded_df, df1[['Pessoas']]], axis=1)
y = df1['Pessoas']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Resultados para a Tese Argentina\n')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

if mse < 0.1 and r2 > 0.9:
    print("A predição foi excelente!")
elif mse < 0.5 and r2 > 0.7:
    print("A predição foi boa.")
else:
    print("A predição não atendeu aos critérios desejados.")

print('\n######################################################################################')


# TESE DE RORAIMA

uf_filtrado = ['RR']
tipo_filtrado = ['Entrada']
classificacao_filtrada = ['Visitante', 'Permanente']

# filtrar o dataset original para as variáveis desejadas para o estudo
df2 = df_raw[(df_raw['UF'].isin(uf_filtrado)) & (df_raw['Tipo'].isin(tipo_filtrado)) & (df_raw['Classificacao'].isin(classificacao_filtrada))]


encoder = OneHotEncoder(sparse=False)
categorical_columns = ['UF', 'Tipo', 'Nacionalidade', 'Classificacao', 'Data']
encoded_data = encoder.fit_transform(df2[categorical_columns])


encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
encoded_df = encoded_df.reset_index(drop=True)
df2 = df2.reset_index(drop=True)


X = pd.concat([encoded_df, df2[['Pessoas']]], axis=1)
y = df2['Pessoas']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('\nResultados para a Tese Roraima \n')
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
if mse < 0.1 and r2 > 0.9:
    print("A predição foi excelente!")
elif mse < 0.5 and r2 > 0.7:
    print("A predição foi boa.")
else:
    print("A predição não atendeu aos critérios desejados.")