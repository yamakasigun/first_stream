import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

data1 = pd.read_csv('Expresso_churn_dataset.csv')
print(data1)


# Remplacons_les_valeurs_manquantes_par_les_moyennes_des_colonnes
def nan_replacement_by_mean(data1):
    for column in data1.columns:
        if pd.api.types.is_numeric_dtype(data1[column]):
            mean_column = data1[column].mean()
            data1[column].fillna(mean_column)


nan_replacement_by_mean(data1)


# Outliers handling
def detect_outliers(df):
    # Calcul de l'IQR pour chaque colonne
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    # Détermination des limites pour détecter les valeurs aberrantes
    bornes_inf = q1 - 1.5 * iqr
    # Identifier les valeurs aberrantes
    aberrations = (df < bornes_inf)
    return aberrations


outliers = detect_outliers(data1)

df_no_outliers = data1[~outliers.any(axis=1)]

# Categorical features encoding
one_hot1 = pd.get_dummies(df_no_outliers['REGION'])
one_hot2 = pd.get_dummies(df_no_outliers['MRG'])
new_data1 = df_no_outliers.drop('user_id', axis=1)
new_data2 = new_data1.drop('REGION', axis=1)
new_data2 = new_data2.join(one_hot1)
new_data3 = new_data2.drop('TENURE', axis=1)
new_data4 = new_data3.drop('MRG', axis=1)
new_data4 = new_data4.join(one_hot2)
new_data = new_data4.drop('TOP_PACK', axis=1)

# DATA SEPARATION
# We're going to predict the churn
y = new_data['CHURN']

# DATA SEPARATION
# We will use the rest of the variables as inputs
X = new_data.drop('CHURN', axis=1)

# DATA SPLITTING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Build our LinearRegression model
lr = LinearRegression()

# fitting training data
lr.fit(X_train, y_train)

# testing model’s performance
y_lr_train_pre = lr.predict(X_train)
y_lr_test_pre = lr.predict(X_test)

# Model Evaluation
lr_train_mean = mean_squared_error(y_train, y_lr_train_pre)
lr_train_r2 = r2_score(y_train, y_lr_train_pre)
lr_test_mean = mean_squared_error(y_test, y_lr_test_pre)
lr_test_r2 = r2_score(y_test, y_lr_test_pre)

# APPLICATION

st.title('PREDICTION DU CHURN')

# Entrées des données qui serviront pour la prédiction
montant = st.number_input('MONTANT')
frequence_rech = st.number_input('FREQUENCE_RECH')
revenue = st.number_input('REVENUE')
arpu_segment = st.number_input('ARPU_SEGMENT')
frequence = st.number_input('FREQUENCE')
volume_data = st.number_input('DATA_VOLUME')
on_net = st.number_input('ON_NET')
orange = st.number_input('ORANGE')
tigo = st.number_input('TIGO')
zone_1 = st.number_input('ZONE1')
zone_2 = st.number_input('ZONE2')
regularity = st.number_input('REGULARITY')
freq_top_pack = st.number_input('FREQ_TOP_PACK')
villes = ['DAKAR', 'DIOURBEL', 'FATICK', 'KAFFRINE', 'KAOLACK', 'KEDOUGOU', 'KOLDA', 'LOUGA', 'M', 'MATAM',
          'SAINT-LOUIS', 'SEDHIOU', 'TAMBACOUNDA', 'THIES', 'ZIGUINCHOR', 'NO']
ville_selectionnee = st.selectbox('selectionnez une ville:', villes)
if ville_selectionnee:
    ville_selectionnee = 1
pass
# Faire la prédiction
if st.button('PREDIRE'):
    data = [[montant, frequence_rech, revenue, arpu_segment, frequence, volume_data, on_net, orange, tigo, zone_1,
             zone_2, regularity, freq_top_pack, ville_selectionnee]]
    predict = lr.predict(data)
    st.success(f"La prédiction est {predict}")
