
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://localhost:5000")

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def log_mlflow_model(modelo, nombre_modelo, X_train, y_train, X_test, y_test, etiqueta):
    with mlflow.start_run(run_name=nombre_modelo):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_params({
            "modelo": modelo.__class__.__name__,
            "tipo_modelo": etiqueta
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "roc_auc": auc
        })

        mlflow.sklearn.log_model(modelo, f"modelo_{etiqueta}")

        print(f"\n✅ Modelo {etiqueta} registrado en MLflow:")
        print(f"AUC: {auc:.4f} | Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, digits=4))


import re
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import prince
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from ydata_profiling import ProfileReport
import joblib
from dash import Dash, html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
from pyngrok import ngrok

datos = pd.read_excel('base_original.xlsx')

print("Dimensiones:", datos.shape)

datos.head(5)

datos['Tipo_Cliente'] = '0'

datos['Tipo_Cliente'] = np.where(
    datos['Altura_mora'].isin([0, -99]),
    '1',
    datos['Tipo_Cliente']
)

datos['Tipo_Cliente'] = np.where(
    (datos['Estado_Acuerdo_Pago_1'] == 'Pagado Pte PyS') &
    (datos['Tipo_Cliente'] == ''),
    '1',
    datos['Tipo_Cliente']
)

datos['Tipo_Cliente'] = np.where(
    (~datos['Fecha_Primer_contacto'].isnull()) &
    (datos['Tipo_Cliente'] == '0'),
    '1',
    datos['Tipo_Cliente']
)

datos['Tipo_Cliente'] = datos['Tipo_Cliente'].replace('', np.nan)
datos['Tipo_Cliente'] = datos['Tipo_Cliente'].fillna('0')

datos['Tipo_Cliente'].value_counts()

datos['Base']=np.where(datos['Tipo_Cliente']=='0', 'Aplicación', 'Desarrollo')

embudo = pd.DataFrame(dict(
    etapa = ['Total_Inicial', 'No_Acuerdos_Recientes', 'No_Saldos_Altos', 'No_Montos_Bajos',
            'Total_Inicial', 'No_Acuerdos_Recientes', 'No_Saldos_Altos', 'No_Montos_Bajos'],
    numero = [194994, 194994, 188650, 175513,
              146713, 130757, 127514, 123194],
    Base = ['Aplicación', 'Aplicación', 'Aplicación', 'Aplicación',
                'Desarrollo', 'Desarrollo', 'Desarrollo', 'Desarrollo']))

fig = px.funnel(embudo, x = 'numero', y = 'etapa', color = 'Base', color_discrete_map = {'Aplicación': 'dimgray', 'Desarrollo': 'steelblue'}
                )
fig.update_traces(textfont = {'color': 'white'})

fig.show()

datos.dtypes.value_counts()

msno.bar(datos, color="steelblue", sort="descending")
plt.title("Valores nulos por variable")
plt.show()

porcentaje_nulos = datos.isnull().mean().sort_values(ascending=False)
porcentaje_nulos[porcentaje_nulos > 0.2]


columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos > 0.2].index
datos_limpios = datos.drop(columns=columnas_a_eliminar)

print("Columnas eliminadas:")
print(list(columnas_a_eliminar))
print("Nueva forma del dataset:", datos_limpios.shape)

datos_limpios.columns

nulos_restantes = datos_limpios.isnull().sum()
nulos_restantes[nulos_restantes > 0].sort_values(ascending=False)

datos_limpios.describe()

datos_limpios=datos_limpios[~datos_limpios['Saldo_Total_cliente']<0]
datos_limpios=datos_limpios[~datos_limpios['Saldo_Capital_cliente']<0]


perfil = ProfileReport(datos_limpios, title="Reporte EDA de BASEU", explorative=True)

# Guarda el reporte en HTML
perfil.to_file("reporte_eda_baseu.html")

# Abrir automáticamente el archivo HTML en el navegador (opcional)
import plotly.offline as pyo
pyo.plot(fig, filename='grafico_interactivo.html')

#### Variables numéricas
df_base = datos_limpios.copy()
for col in df_base.select_dtypes(include=[np.number]).columns:
    if df_base[col].isnull().sum() > 0:
        mediana = df_base[col].median()
        df_base[col].fillna(mediana, inplace=True)

#### Variables categoricas
def imputar_categorica_aleatoria(df, col):
    valores = df[col].dropna().value_counts(normalize=True)
    df[col] = df[col].apply(
        lambda x: np.random.choice(valores.index, p=valores.values) if pd.isnull(x) else x
    )

categoricas = df_base.select_dtypes(include=['object', 'category']).columns

for col in categoricas:
    if df_base[col].isnull().sum() > 0:
        imputar_categorica_aleatoria(df_base, col)

nulos_restantes = df_base.isnull().sum()
nulos_restantes[nulos_restantes > 0].sort_values(ascending=False)

def convertir_puntaje(valor):
    if pd.isnull(valor):
        return np.nan
    valor_str = str(valor).strip().upper()

    if any(x in valor_str for x in ['EXCLUSIONES', 'NO APLICA', '-1', 'SIN DATO']):
        return np.nan

    match_mayor = re.search(r'MAYOR(?:ES)? A (\d+)', valor_str)
    if match_mayor:
        return float(match_mayor.group(1)) + 10

    match_rango = re.match(r'(\d+)\s*-\s*(\d+)', valor_str)
    if match_rango:
        inf, sup = map(float, match_rango.groups())
        return (inf + sup) / 2

    match_num = re.match(r'^\d{2,4}$', valor_str)
    if match_num:
        return float(match_num.group(0))

    return np.nan

def convertir_ingresos(valor):
    if pd.isnull(valor):
        return np.nan
    valor_str = str(valor).strip().upper().replace('$', '').replace(',', '')

    if any(x in valor_str for x in ['EXCLUSIONES', 'NO APLICA', 'SIN DATO']):
        return np.nan

    match_mayor = re.search(r'MAYOR(?:ES)? A (\d+)', valor_str)
    if match_mayor:
        return (float(match_mayor.group(1)) + 500) * 1000

    match_rango = re.match(r'(\d+)\s*-\s*(\d+)', valor_str)
    if match_rango:
        inf, sup = map(float, match_rango.groups())
        return (inf + sup) / 2 * 1000

    match_num = re.match(r'^\d+(\.\d+)?$', valor_str)
    if match_num:
        valor_float = float(valor_str)
        if valor_float < 100000:
            return valor_float * 1000
        else:
            return valor_float

    return np.nan

df_base['PUNTAJE_1_VALOR'] = df_base['PUNTAJE_1'].apply(convertir_puntaje)
df_base['INGRESOS_VALOR'] = df_base['INGRESOS_ESTIMADOS_DATACREDITO_1'].apply(convertir_ingresos)

df_base['PUNTAJE_1_VALOR'].fillna(df_base['PUNTAJE_1_VALOR'].median(), inplace=True)
df_base['INGRESOS_VALOR'].fillna(df_base['INGRESOS_VALOR'].median(), inplace=True)

df_base.columns

SMMLV = 1400000
if 'CUOTAS_MERCADO_1' not in df_base.columns:
    df_base['CUOTAS_MERCADO_1'] = 0

df_base['INGRESO_DISPONIBLE'] = df_base['INGRESOS_VALOR'] - df_base['CUOTAS_MERCADO_1']

if 'Saldo_Capital_cliente' in df_base.columns:
    df_base['SMMV_Saldo_Capital_cliente'] = df_base['Saldo_Capital_cliente'] / SMMLV
    df_base['Ingreso_Saldo_Capital_cliente'] = df_base['Saldo_Capital_cliente'] / df_base['INGRESOS_VALOR']
    df_base['RANGO_SALDO_CAPITAL'] = np.select([
        df_base['Saldo_Capital_cliente'] < 2500000,
        df_base['Saldo_Capital_cliente'].between(2500000, 10000000),
        df_base['Saldo_Capital_cliente'].between(10000000, 30000000),
        df_base['Saldo_Capital_cliente'] > 30000000],
        ['MONTO_BAJO', 'MONTO_MEDIO', 'MONTO_MEDIO_ALTO', 'MONTO_ALTO'], default='SIN CLASIFICAR')

if 'Saldo_Total_cliente' in df_base.columns:
    df_base['SMMV_Saldo_Total_cliente'] = df_base['Saldo_Total_cliente'] / SMMLV
    df_base['Ingreso_Saldo_Total_cliente'] = df_base['Saldo_Total_cliente'] / df_base['INGRESOS_VALOR']

if 'SALDO_TOTAL_DATACREDITO' in df_base.columns:
    df_base['SMMV_SALDO_TOTAL_DATACREDITO'] = df_base['SALDO_TOTAL_DATACREDITO'] / SMMLV
    df_base['Ingreso_SALDO_TOTAL_DATACREDITO'] = df_base['SALDO_TOTAL_DATACREDITO'] / df_base['INGRESOS_VALOR']

if 'SALDO_CASTIGO_DATACREDITO' in df_base.columns:
    df_base['SMMV_SALDO_CASTIGO_DATACREDITO'] = df_base['SALDO_CASTIGO_DATACREDITO'] / SMMLV
    df_base['Ingreso_SALDO_CASTIGO_DATACREDITO'] = df_base['SALDO_CASTIGO_DATACREDITO'] / df_base['INGRESOS_VALOR']

subregiones=pd.read_excel('Subregiones.xlsx', sheet_name='Hoja3')

df_base[(~df_base['Ciudad'].isin(['MEDELLIN', 'CALI', 'SIN INFORMACION','BOGOTA D.C.' ]))]['Ciudad'].value_counts(sort = True)

df_base['Ciudad2']=np.where( df_base['Ciudad']=='BOGOTA D.C.', 'BOGOTA D.C.',
                       np.where(df_base['Ciudad']=='CALI', 'CALI',
                                np.where(df_base['Ciudad']=='SIN INFORMACION', 'SIN INFORMACION',
                                          np.where(df_base['Ciudad']=='MEDELLIN', 'MEDELLIN',
                                                    np.where(df_base['Ciudad']=='BARRANQUILLA', 'BARRANQUILLA', '')))))

df_base['Ciudad']=np.where(df_base['Ciudad']=='GUADALAJARA DE BUGA','BUGA',
             np.where(df_base['Ciudad']=='EL CARMEN DE BOLIVAR', 'CARMEN DE BOLÍVAR',
             np.where(df_base['Ciudad']=='PUEBLOVIEJO',  'PUEBLO VIEJO',
             np.where(df_base['Ciudad']=='SAN ANDRES DE TUMACO',  'ARCHIPIELAGO DE SAN ANDRES',
             np.where(df_base['Ciudad']=='ISTMINA',  'SAN PABLO',
             np.where(df_base['Ciudad']=='EL CARMEN DE VIBORAL', 'CARMEN DE VIBORAL',
             np.where(df_base['Ciudad']=='VILLA DE SAN DIEGO DE UBATE', 'UBATE',
             np.where(df_base['Ciudad']=='SANTACRUZ', 'SANTA CRUZ',
             np.where(df_base['Ciudad']=='SANTA ROSA DE VITERBO', 'SAN ROSA VITERBO',
             np.where(df_base['Ciudad']=='EL SANTUARIO', 'SANTUARIO',
             np.where(df_base['Ciudad']=='CUBARRAL', 'SAN LUIS DE CUBARRAL',
             np.where(df_base['Ciudad']=='VILLAGARZON', 'VILLA GARZON',
             np.where(df_base['Ciudad']=='URIBE', 'LA URIBE',
             np.where(df_base['Ciudad']=='SAN CARLOS DE GUAROA', 'SAN CARLOS GUAROA',
             np.where(df_base['Ciudad']=='CURILLO', '',
             np.where(df_base['Ciudad']=='VISTAHERMOSA', 'VISTA HERMOSA',
             np.where(df_base['Ciudad']=='SAN ANTONIO DEL TEQUENDAMA', 'SAN ANTONIO DE TEQUENDAMA',
             np.where(df_base['Ciudad']=='PEÃOL', 'PEÑOL',
             np.where(df_base['Ciudad']=='CHACHAGSI',   'CHACHAGUI',
             np.where(df_base['Ciudad']=='SAN JUAN DE BETULIA', 'SAN JUAN BETULIA',
             np.where(df_base['Ciudad']=='SAN LUIS DE SINCE', 'SINCELEJO',
             np.where(df_base['Ciudad']=='GUACHENE' ,'GUACHETÁ',
             np.where(df_base['Ciudad']=='EL CANTON DEL SAN PABLO', 'CANTON DE SAN PABLO',
             np.where(df_base['Ciudad']=='TOGSI','TOGÜÍ',
             np.where(df_base['Ciudad']=='SAN PABLO DE BORBUR', 'SAN PABLO BORBUR',df_base['Ciudad']

             )))))))))))))))))))))))))

df_base['Ciudad']=df_base['Ciudad'].str.replace('Ã‘', 'Ñ')

subregiones['NOMBRE_MPIO']=subregiones['NOMBRE_MPIO'].str.upper()
subregiones['NOMBRE_DEPTO']=subregiones['NOMBRE_DEPTO'].str.upper()
df_base['Ciudad']=df_base['Ciudad'].str.upper()
df_base['Ciudad2']=df_base['Ciudad2'].str.upper()

subregiones['NOMBRE_MPIO']=subregiones['NOMBRE_MPIO'].str.replace('Á', 'A').str.replace('É', 'E').str.replace('Í', 'I').str.replace('Ó', 'O').str.replace('Ú', 'U')
subregiones['NOMBRE_DEPTO']=subregiones['NOMBRE_DEPTO'].str.replace('Á', 'A').str.replace('É', 'E').str.replace('Í', 'I').str.replace('Ó', 'O').str.replace('Ú', 'U')
df_base['Ciudad']=df_base['Ciudad'].str.replace('Á', 'A').str.replace('É', 'E').str.replace('Í', 'I').str.replace('Ó', 'O').str.replace('Ú', 'U')
df_base['Ciudad2']=df_base['Ciudad2'].str.replace('Á', 'A').str.replace('É', 'E').str.replace('Í', 'I').str.replace('Ó', 'O').str.replace('Ú', 'U')

df_base=pd.merge(df_base, subregiones, left_on='Ciudad', right_on='NOMBRE_MPIO', how='left')

df_base['CIUDAD_FINAL']=np.where(df_base['Ciudad2']=='', df_base['NOMBRE_DEPTO'],  df_base['Ciudad2'])

df_base['CIUDAD_FINAL']=np.where(df_base['CIUDAD_FINAL'].isnull(), 'SIN INFORMACION',  df_base['CIUDAD_FINAL'])

len(df_base['CIUDAD_FINAL'].value_counts())

df_base.columns

fig, axes = plt.subplots(2,2, figsize=(20, 15))

pd.crosstab(df_base['RANGO_EDAD_1'], df_base['Base']).plot(kind="bar",  ax=axes[0, 0], ylabel='Cantidad de clientes', color=['steelblue', 'peru'], linestyle=':', title='A. Rango Edad', xlabel='')
pd.crosstab(df_base['CIUDAD_FINAL'], df_base['Base']).plot(kind="line",  ax=axes[1, 0], ylabel='Cantidad de clientes', color=['steelblue', 'peru'], linestyle=':',  title='B. Ciudad', xlabel='')
pd.crosstab(df_base['RANGO_SALDO_CAPITAL'], df_base['Base']).plot(kind="line",  ax=axes[0, 1], ylabel='Cantidad de clientes', color=['steelblue', 'peru'], linestyle=':',  title='C. Rango Saldo Capital', xlabel='')
pd.crosstab(df_base['Tipo_Cliente'], df_base['Base']).plot(kind="line",  ax=axes[1, 1], ylabel='Cantidad de clientes', color=['steelblue', 'peru'],  title='D. Tipo Cliente', xlabel='')

df_base = df_base.drop(columns=['Ciudad', 'Ciudad2', 'NOMBRE_MPIO', 'NOMBRE_DEPTO','Contacto__c','INGRESOS_ESTIMADOS_DATACREDITO','PUNTAJE','PUNTAJE_1','INGRESOS_ESTIMADOS_DATACREDITO_1','FECHA_RECEPCION','FECHA_RECEPCION_1','Operado_Por__c','RANGO_EDAD','Base'])

cat_cols = df_base.select_dtypes(include=['object', 'category']).columns

categorias_por_variable = df_base[cat_cols].nunique().sort_values(ascending=False)


print("Número de categorías únicas por variable categórica:")
print(categorias_por_variable)

plt.figure(figsize=(10,6))
categorias_por_variable.plot(kind='barh', color='steelblue')
plt.title('Número de categorías por variable categórica')
plt.xlabel('Cantidad de categorías')
plt.ylabel('Variable')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# Variables numericas
vars_numericas = df_base.select_dtypes(include=[np.number]).columns
print("Variables numéricas:", list(vars_numericas))

df_base_outliers = df_base.copy()

for col in vars_numericas:
    q1 = df_base_outliers[col].quantile(0.25)
    q3 = df_base_outliers[col].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    media_sin_outliers = df_base_outliers[(df_base_outliers[col] >= limite_inf) & (df_base_outliers[col] <= limite_sup)][col].mean()

    df_base_outliers[col] = np.where(df_base_outliers[col] > limite_sup, media_sin_outliers, df_base_outliers[col])
    df_base_outliers[col] = np.where(df_base_outliers[col] < limite_inf, media_sin_outliers, df_base_outliers[col])

for col in vars_numericas[:]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(y=df_base[col], ax=ax[0], color='peru')
    ax[0].set_title(f'Antes: {col}')
    sns.boxplot(y=df_base_outliers[col], ax=ax[1], color='steelblue')
    ax[1].set_title(f'Después: {col}')
    plt.tight_layout()
    plt.show()

df_base.columns

df_base_outliers['SMMV_Saldo_Total_cliente'].min()

num_cols = df_base_outliers.select_dtypes(include=np.number).columns
X = df_base_outliers[num_cols]

#MATRIZ DE CORRELACION
corr_matrix = X.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación de variables numéricas")
plt.show()

# Variables originales en pesos
variables_en_pesos = [
    'Saldo_Capital_cliente',
    'Saldo_Total_cliente',
    'SALDO_TOTAL_DATACREDITO',
    'SALDO_CASTIGO_DATACREDITO',
    'INGRESOS_VALOR'
]

# Variables proporcionales respecto a ingresos
variables_proporcionales = [
    'Ingreso_Saldo_Capital_cliente',
    'Ingreso_Saldo_Total_cliente',
    'Ingreso_SALDO_TOTAL_DATACREDITO',
    'Ingreso_SALDO_CASTIGO_DATACREDITO'
]

variables_a_eliminar = variables_en_pesos + variables_proporcionales

df_base_outliers = df_base_outliers.drop(columns=variables_a_eliminar)

print("Variables eliminadas correctamente.")
print("Columnas actuales:", df_base_outliers.columns.tolist())


num_cols = df_base_outliers.select_dtypes(include=np.number).columns
X = df_base_outliers[num_cols]

#MATRIZ DE CORRELACION
corr_matrix = X.corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Matriz de correlación de variables numéricas")
plt.show()

X = df_base_outliers[num_cols].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_df = pd.DataFrame()
vif_df["Variable"] = num_cols
vif_df["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

vif_df = vif_df.sort_values(by="VIF", ascending=False)

print(vif_df)

X = df_base_outliers.drop(columns='Tipo_Cliente')
y = df_base_outliers['Tipo_Cliente'].astype(int)

X_num = X.select_dtypes(include=np.number)
X_cat = X.select_dtypes(include=['object', 'category']).astype(str)

df_corr = pd.concat([X_num, y], axis=1).corr()

df_corr['Tipo_Cliente'].sort_values(ascending=False)

for col in X_cat.columns:
    tabla = pd.crosstab(X_cat[col], y, normalize='index')
    print(f"\nVariable: {col}")
    print(tabla.head())

df_base_outliers = df_base_outliers.drop(columns=['estado','localizado_historico','Motivo_Renuencia_Cliente'])

df_base_outliers_model_1 = df_base_outliers.copy()

X = df_base_outliers_model_1.drop(columns='Tipo_Cliente')
y = df_base_outliers_model_1['Tipo_Cliente'].astype(int)

X_num = X.select_dtypes(include=np.number)
X_cat = X.select_dtypes(include=['object', 'category']).astype(str)


X_cat_dummies = pd.get_dummies(X_cat, drop_first=True)

X_completo = pd.concat([X_num.reset_index(drop=True), X_cat_dummies.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
X_escalado = scaler.fit_transform(X_completo)

X_train, X_test, y_train, y_test = train_test_split(
    X_escalado, y, test_size=0.1, stratify=y, random_state=42
)


#### PRIMER MODELO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import mlflow
import mlflow.sklearn

# Asegúrate de que estas variables estén definidas antes:
# X_train, X_test, y_train, y_test

def evaluar_modelo_con_roc_mlflow(modelo, nombre_modelo, color, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=nombre_modelo):
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = None

        # Reporte
        print(f"\n📊 Reporte del modelo: {nombre_modelo}")
        print(classification_report(y_test, y_pred, digits=4))

        # Métricas y MLflow
        acc = accuracy_score(y_test, y_pred)
        mlflow.log_param("modelo", modelo.__class__.__name__)
        mlflow.log_param("tipo", nombre_modelo)
        mlflow.log_metric("accuracy", acc)

        if y_prob is not None:
            auc = roc_auc_score(y_test, y_prob)
            mlflow.log_metric("roc_auc", auc)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{nombre_modelo} (AUC={auc:.4f})', color=color)
        else:
            plt.plot([], [], label=f'{nombre_modelo} (no probas)', color=color)

        mlflow.sklearn.log_model(modelo, f"modelo_{nombre_modelo.replace(' ', '_').lower()}")

# ============================
# Entrenamiento y evaluación
# ============================
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')

evaluar_modelo_con_roc_mlflow(
    LogisticRegression(max_iter=1000),
    "Regresión Logística Escenario 1",
    color='blue',
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

evaluar_modelo_con_roc_mlflow(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    "Árbol de Clasificación Escenario 1",
    color='orange',
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

evaluar_modelo_con_roc_mlflow(
    RandomForestClassifier(n_estimators=100, random_state=42),
    "Random Forest Escenario 1",
    color='green',
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test
)

plt.title("Curvas ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True)
plt.legend()
plt.show()

# Lista de las 10 ciudades/departamentos con mayor frecuencia
ciudades_top = [
    'BOGOTA D.C.', 'SIN INFORMACION', 'CALI', 'SANTANDER', 'MEDELLIN',
    'CUNDINAMARCA', 'ANTIOQUIA', 'BARRANQUILLA', 'VALLE DEL CAUCA', 'BOLIVAR'
]

df_base_outliers['CIUDAD_FINAL_REAGRUPADA'] = df_base_outliers['CIUDAD_FINAL'].apply(
    lambda x: x if x in ciudades_top else 'OTRAS CIUDADES'
)

X_pca = df_base_outliers.drop(columns='Tipo_Cliente')
y_pca = df_base_outliers['Tipo_Cliente'].astype(int)

X_num = X_pca.select_dtypes(include=np.number)

X_num_train, X_num_test, y_train_pca, y_test_pca = train_test_split(
    X_num, y_pca, test_size=0.1, stratify=y_pca, random_state=42
)

scaler = StandardScaler()
X_num_train_scaled = scaler.fit_transform(X_num_train)
X_num_test_scaled = scaler.transform(X_num_test)

pca = PCA(n_components=0.85)
X_pca_train = pca.fit_transform(X_num_train_scaled)
X_pca_test = pca.transform(X_num_test_scaled)

plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', color='darkblue')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.title('Varianza Explicada Acumulada por PCA')
plt.grid(True)
plt.tight_layout()
plt.show()

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_num_train.columns
)

plt.figure(figsize=(12, 6))
sns.heatmap(loadings, annot=True, cmap='Spectral', center=0)
plt.title("Matriz de carga (loadings) - 9 componentes principales (train)")
plt.tight_layout()
plt.show()

from sklearn.cluster import KMeans

df_pca_train = pd.DataFrame(X_pca_train[:, :9], columns=[f'PC{i+1}' for i in range(9)], index=X_num_train.index)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters_train = kmeans.fit_predict(df_pca_train)

df_pca_train['Segmento_PCA'] = clusters_train

df_final_train = df_base_outliers.loc[X_num_train.index].copy()
df_final_train['Segmento_PCA'] = df_pca_train['Segmento_PCA'].values

print(df_pca_train['Segmento_PCA'].value_counts())
df_final_train.head()


# Componentes principales del test
df_pca_test = pd.DataFrame(X_pca_test[:, :9], columns=[f'PC{i+1}' for i in range(9)], index=X_num_test.index)

clusters_test = kmeans.predict(df_pca_test)

df_pca_test['Segmento_PCA'] = clusters_test

df_final_test = df_base_outliers.loc[X_num_test.index].copy()
df_final_test['Segmento_PCA'] = df_pca_test['Segmento_PCA'].values

print(df_pca_test.shape, df_final_test.shape)

df_final_train

categoricas_mca = [
    'Entidad_principal',
    'Genero__c',
    'RANGO_EDAD_1',
    'RANGO_SALDO_CAPITAL',
    'CIUDAD_FINAL_REAGRUPADA'
]

X_cat_mca = df_final_train[categoricas_mca].astype(str).copy()

mca = prince.MCA(n_components=3, n_iter=5, random_state=42)
X_mca = mca.fit_transform(X_cat_mca)

df_mca = pd.DataFrame(
    X_mca.values,
    columns=[f'MCA{i+1}' for i in range(X_mca.shape[1])],
    index=X_cat_mca.index
)

df_mca_plot = df_mca.copy()
df_mca_plot['Tipo_Cliente'] = df_final_train['Tipo_Cliente'].reset_index(drop=True)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df_mca_plot['MCA1'],
    y=df_mca_plot['MCA2'],
    hue=df_mca_plot['Tipo_Cliente'],
    palette='Set2',
    alpha=0.7
)
plt.title("Dispersión MCA (Categorías Reagrupadas)")
plt.xlabel("MCA 1")
plt.ylabel("MCA 2")
plt.grid(True)
plt.legend(title='Tipo Cliente')
plt.tight_layout()
plt.show()


pca_cols = [col for col in df_pca_train.columns if col.startswith("PC")]
df_pca_only = df_pca_train[pca_cols]
segmentos = df_pca_train[['Segmento_PCA']]

X_train_reducido = pd.concat([df_pca_only, df_mca, segmentos], axis=1)

# Entrenamiento
X_cat_mca_train = df_final_train[categoricas_mca].astype(str).copy()
mca = prince.MCA(n_components=3, random_state=42)
X_mca_train = mca.fit_transform(X_cat_mca_train)

df_mca = pd.DataFrame(
    X_mca_train.values,
    columns=[f'MCA{i+1}' for i in range(X_mca_train.shape[1])],
    index=X_cat_mca_train.index
)

# Test: asegurar columnas alineadas con las del fit

# 🔹 1. Asegurar consistencia en las variables categóricas
for col in categoricas_mca:
    df_final_train[col] = df_final_train[col].astype(str).str.strip()
    df_final_test[col] = df_final_test[col].astype(str).str.strip()

# 🔹 2. One-hot para controlar columnas
X_cat_mca_train = pd.get_dummies(df_final_train[categoricas_mca], drop_first=False)
X_cat_mca_test = pd.get_dummies(df_final_test[categoricas_mca], drop_first=False)

# 🔹 3. Alineación de columnas
X_cat_mca_test = X_cat_mca_test.reindex(columns=X_cat_mca_train.columns, fill_value=0)

# 🔹 4. Entrenar y transformar con MCA
mca = prince.MCA(n_components=3, random_state=42)
X_mca_train = mca.fit_transform(X_cat_mca_train)
X_mca_test = mca.transform(X_cat_mca_test)

# 🔹 5. Convertir resultados a DataFrame
df_mca = pd.DataFrame(X_mca_train.values, columns=[f'MCA{i+1}' for i in range(X_mca_train.shape[1])], index=X_cat_mca_train.index)
df_mca_test = pd.DataFrame(X_mca_test.values, columns=[f'MCA{i+1}' for i in range(X_mca_test.shape[1])], index=X_cat_mca_test.index)

# 🔹 6. Confirmar que no hay nulos
print("¿Hay nulos en el entrenamiento después de MCA?")
print(df_mca.isnull().sum())


# Modelo con los componentes de PCA y MCA

X_train_reducido = pd.concat([
    df_pca_train.drop(columns=['Segmento_PCA']).reset_index(drop=True),
    X_mca.reset_index(drop=True)
], axis=1)

X_test_reducido = pd.concat([
    df_pca_test.reset_index(drop=True),
    df_mca_test.reset_index(drop=True)
], axis=1)

X_train_reducido['Segmento_PCA'] = df_pca_train['Segmento_PCA'].values
X_test_reducido['Segmento_PCA'] = df_pca_test['Segmento_PCA'].values

y_train_reducido = df_final_train['Tipo_Cliente'].astype(int).values
y_test_reducido = df_final_test['Tipo_Cliente'].astype(int).values

df_mca = df_mca.loc[df_final_train.index]

X_train_reducido = pd.concat([df_pca_only, df_mca, segmentos], axis=1)

X_train_reducido
X_test_reducido = X_test_reducido[X_train_reducido.columns]

### MODELO 2 

modelos = {
    'Regresión Logística Escenario 2': LogisticRegression(max_iter=1000, solver='lbfgs'),
    'Árbol de Clasificación Escenario 2': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest Escenario 2': RandomForestClassifier(n_estimators=100, random_state=42)
}

plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')

for nombre, modelo in modelos.items():
    with mlflow.start_run(run_name=nombre):
        modelo.fit(X_train_reducido, y_train_reducido)
        y_pred = modelo.predict(X_test_reducido)
        y_prob = modelo.predict_proba(X_test_reducido)[:, 1]

        # Reporte
        print(f"\n📊 Modelo: {nombre}")
        print("Classification Report:")
        print(classification_report(y_test_reducido, y_pred, digits=4))

        # Métricas
        auc = roc_auc_score(y_test_reducido, y_prob)
        acc = accuracy_score(y_test_reducido, y_pred)

        mlflow.log_params({
            "modelo": modelo.__class__.__name__,
            "tipo": "PCA_MCA_CLUSTER"
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "roc_auc": auc
        })
        mlflow.sklearn.log_model(modelo, f"modelo_{nombre.lower().replace(' ', '_')}")

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test_reducido, y_prob)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC = {auc:.3f})')

# Gráfico final
plt.title('Curvas ROC - Modelos con PCA + MCA + Cluster')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

df_base_outliers=df_base_outliers.drop(columns='CIUDAD_FINAL_REAGRUPADA')

df_feature_selection = df_base_outliers.copy()


X = df_feature_selection.drop(columns=['Tipo_Cliente'])
y = df_feature_selection['Tipo_Cliente'].astype(int)

X_cat = X.select_dtypes(include='object').astype(str)
X_num = X.select_dtypes(include=np.number)
X_dummies = pd.get_dummies(X_cat, drop_first=True)
X_full = pd.concat([X_num.reset_index(drop=True), X_dummies.reset_index(drop=True)], axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, stratify=y, random_state=42)

selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
selector.fit(X_train, y_train)

X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

import mlflow
import mlflow.sklearn
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Asegúrate de tener:
# - X_train_sel, X_test_sel, y_train, y_test

def evaluar_modelo_mlflow(modelo, nombre, color):
    with mlflow.start_run(run_name=f"{nombre} (Feature Selection)"):
        modelo.fit(X_train_sel, y_train)
        y_pred = modelo.predict(X_test_sel)
        y_prob = modelo.predict_proba(X_test_sel)[:, 1]

        # Reporte
        print(f"\n📊 Modelo: {nombre}")
        print(classification_report(y_test, y_pred, digits=4))

        # Métricas
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_params({
            "modelo": modelo.__class__.__name__,
            "tipo": "feature_selection"
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "roc_auc": auc
        })
        mlflow.sklearn.log_model(modelo, f"modelo_{nombre.lower().replace(' ', '_')}")

        # Curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{nombre} (AUC={auc:.3f})', color=color)

# =======================
# Ejecución
# =======================
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--', label='Aleatorio')

evaluar_modelo_mlflow(LogisticRegression(max_iter=1000), "Regresión Logística Escenario 3", color='blue')
evaluar_modelo_mlflow(DecisionTreeClassifier(max_depth=5, random_state=42), "Árbol de Clasificación Escenario 3", color='orange')
evaluar_modelo_mlflow(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest Escenario 3", color='green')

plt.title("Curvas ROC - Modelos con Feature Selection")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============== Scorecard de probabilidades (Random Forest final)
modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train_sel, y_train)
y_prob_rf = modelo_rf.predict_proba(X_test_sel)[:, 1]

df_score = pd.DataFrame({
    'probabilidad': y_prob_rf,
    'real': y_test.reset_index(drop=True)
})

df_score['decil'] = pd.qcut(df_score['probabilidad'], 10, labels=False) + 1  # 1-10

scorecard = df_score.groupby('decil').agg(
    Total=('real', 'count'),
    Buenos=('real', 'sum')
).reset_index()

scorecard['% Buenos'] = (scorecard['Buenos'] / scorecard['Total'] * 100).round(2)

scorecard = scorecard.sort_values('decil', ascending=False).reset_index(drop=True)

print("Scorecard por Deciles (Random Forest):")
print(scorecard)

plt.figure(figsize=(10, 6))
sns.barplot(x='decil', y='% Buenos', data=scorecard, palette='Blues_r')
plt.title('% de Buenos por Decil (Random Forest)')
plt.xlabel('Decil (10 = Mayor probabilidad de ser Bueno)')
plt.ylabel('% Buenos')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X_completo, y, test_size=0.1, stratify=y, random_state=42
)

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

print(np.unique(y_train, return_counts=True))

clientes_sin_contacto = df_base_outliers_model_1[df_base_outliers_model_1['Tipo_Cliente'] == '0'].copy()

X_sin = clientes_sin_contacto.drop(columns='Tipo_Cliente')
X_sin_num = X_sin.select_dtypes(include=np.number)
X_sin_cat = X_sin.select_dtypes(include=['object', 'category']).astype(str)

X_sin_cat_dummies = pd.get_dummies(X_sin_cat, drop_first=True)

X_sin_completo = pd.concat([X_sin_num.reset_index(drop=True), X_sin_cat_dummies.reset_index(drop=True)], axis=1)
X_sin_completo = X_sin_completo.reindex(columns=X_completo.columns, fill_value=0)

X_sin_escalado = scaler.transform(X_sin_completo)

y_prob_sin = modelo_rf.predict_proba(X_sin_escalado)[:, 1]

df_score_sin = pd.DataFrame({
    'probabilidad': y_prob_sin
}, index=clientes_sin_contacto.index)

df_score_sin['decil'] = pd.qcut(df_score_sin['probabilidad'], 10, labels=False, duplicates='drop') + 1

scorecard_sin = df_score_sin.groupby('decil').agg(
    Total=('probabilidad', 'count'),
    Probabilidad_Promedio=('probabilidad', 'mean')
).reset_index().sort_values('decil', ascending=False).reset_index(drop=True)

scorecard_sin['Probabilidad_Promedio'] = (scorecard_sin['Probabilidad_Promedio'] * 100).round(2)

print("Scorecard de Priorización (Clientes Sin Contacto - Random Forest):")
print(scorecard_sin)

fig, ax1 = plt.subplots(figsize=(10, 6))

sns.barplot(x='decil', y='Total', data=scorecard_sin, palette='Blues_r', ax=ax1)
ax1.set_ylabel('Cantidad de Clientes', color='navy')
ax1.set_xlabel('Decil (10 = Mayor probabilidad)')
ax1.tick_params(axis='y', labelcolor='navy')

ax2 = ax1.twinx()
sns.lineplot(x='decil', y='Probabilidad_Promedio', data=scorecard_sin, color='crimson', marker='o', linewidth=2, ax=ax2)
ax2.set_ylabel('Probabilidad Promedio (%)', color='crimson')
ax2.tick_params(axis='y', labelcolor='crimson')

plt.title('Scorecard de Priorización - Clientes Sin Contacto')
plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

joblib.dump(modelo_rf, "modelo_final.pkl")

import subprocess


subprocess.run(["ngrok", "config", "add-authtoken", "2xsAceL1D0yWwBayJNlaMqTELeo_5b9Tpb1Y4tDRWytdytdok"])


# ========== PARÁMETROS DE IMÁGENES ==========
IMG_LEFT_SRC = "https://data.org/wp-content/uploads/2022/02/Universidad-de-los-Andes.png"
IMG_RIGHT_SRC = "https://colombiafintech.co/static/uploads/5e45c3a34dda773c0791515c_QNT.png"
IMG_LEFT_WIDTH = "200px"
IMG_LEFT_HEIGHT = "auto"
IMG_RIGHT_WIDTH = "200px"
IMG_RIGHT_HEIGHT = "auto"

# ========== CARGUE MODELO ==========
modelo = joblib.load("modelo_final.pkl")

# ========== PREPARACION DATOS ==========
df_general = df_base_outliers.copy()
df_con_score = df_base_outliers.copy()
df_con_score = df_con_score.join(df_score_sin[['probabilidad', 'decil']], how='left')
df_con_score = df_con_score[df_con_score['decil'].notna()].copy()
df_con_score['decil'] = df_con_score['decil'].astype(int)

umbrales = df_score_sin['probabilidad'].quantile(np.linspace(0, 1, 11)).values

scorecard_sin = (
    df_score_sin.groupby('decil')
    .agg(
        Total=('probabilidad', 'count'),
        Probabilidad_Promedio=('probabilidad', 'mean')
    )
    .reset_index()
)

scorecard_sin['Probabilidad_Promedio'] = (scorecard_sin['Probabilidad_Promedio'] * 100).round(2)

df_coordenadas = pd.DataFrame([
    {'CIUDAD_FINAL': 'BOGOTA D.C.', 'lat': 4.7110, 'lon': -74.0721},
    {'CIUDAD_FINAL': 'SIN INFORMACION', 'lat': 4.5, 'lon': -74.1},
    {'CIUDAD_FINAL': 'CALI', 'lat': 3.4516, 'lon': -76.5320},
    {'CIUDAD_FINAL': 'MEDELLIN', 'lat': 6.2442, 'lon': -75.5812},
    {'CIUDAD_FINAL': 'BARRANQUILLA', 'lat': 10.9639, 'lon': -74.7964},
    {'CIUDAD_FINAL': 'CUNDINAMARCA', 'lat': 4.7541, 'lon': -74.0928},
    {'CIUDAD_FINAL': 'ANTIOQUIA', 'lat': 6.5581, 'lon': -75.8258},
    {'CIUDAD_FINAL': 'TOLIMA', 'lat': 4.4389, 'lon': -75.2322},
    {'CIUDAD_FINAL': 'MAGDALENA', 'lat': 10.4167, 'lon': -74.6667},
    {'CIUDAD_FINAL': 'ATLANTICO', 'lat': 10.6966, 'lon': -74.8741},
    {'CIUDAD_FINAL': 'HUILA', 'lat': 2.5359, 'lon': -75.5277},
    {'CIUDAD_FINAL': 'CESAR', 'lat': 9.3370, 'lon': -73.6536},
    {'CIUDAD_FINAL': 'RISARALDA', 'lat': 5.3150, 'lon': -75.9928},
    {'CIUDAD_FINAL': 'CORDOBA', 'lat': 8.75, 'lon': -75.88},
    {'CIUDAD_FINAL': 'BOYACA', 'lat': 5.4545, 'lon': -73.3620},
    {'CIUDAD_FINAL': 'CALDAS', 'lat': 5.0711, 'lon': -75.5138},
    {'CIUDAD_FINAL': 'NARIÑO', 'lat': 1.2136, 'lon': -77.2811},
    {'CIUDAD_FINAL': 'CASANARE', 'lat': 5.3544, 'lon': -71.9226},
    {'CIUDAD_FINAL': 'CAUCA', 'lat': 2.7, 'lon': -76.7},
    {'CIUDAD_FINAL': 'SUCRE', 'lat': 9.3047, 'lon': -75.3978},
    {'CIUDAD_FINAL': 'QUINDIO', 'lat': 4.5339, 'lon': -75.6820},
    {'CIUDAD_FINAL': 'LA GUAJIRA', 'lat': 11.3548, 'lon': -72.5200},
    {'CIUDAD_FINAL': 'CAQUETA', 'lat': 0.8694, 'lon': -73.8419},
    {'CIUDAD_FINAL': 'CHOCO', 'lat': 5.6832, 'lon': -76.6611},
    {'CIUDAD_FINAL': 'ARAUCA', 'lat': 7.0903, 'lon': -70.7617},
    {'CIUDAD_FINAL': 'AMAZONAS', 'lat': -1.4436, 'lon': -71.5724},
    {'CIUDAD_FINAL': 'PUTUMAYO', 'lat': 0.5, 'lon': -76.5},
    {'CIUDAD_FINAL': 'ARCHIPIELAGO DE SAN ANDRES', 'lat': 12.5847, 'lon': -81.7006},
    {'CIUDAD_FINAL': 'GUAVIARE', 'lat': 2.0, 'lon': -72.6},
    {'CIUDAD_FINAL': 'VICHADA', 'lat': 5.0, 'lon': -69.3},
    {'CIUDAD_FINAL': 'GUAINIA', 'lat': 2.57, 'lon': -68.76},
    {'CIUDAD_FINAL': 'VAUPES', 'lat': 0.85, 'lon': -70.8}
])
conteo_ciudades = df_general['CIUDAD_FINAL'].value_counts().reset_index()
conteo_ciudades.columns = ['CIUDAD_FINAL', 'count']
ciudades_geo_original = pd.merge(conteo_ciudades, df_coordenadas, on='CIUDAD_FINAL', how='left')

df_entidad_general = df_general['Entidad_principal'].value_counts().reset_index()
df_entidad_general.columns = ['Entidad', 'count']
fig_entidad_general = px.bar(df_entidad_general, x='Entidad', y='count', title='Entidad Principal')


