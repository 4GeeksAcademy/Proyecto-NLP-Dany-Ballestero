#from utils import db_connect
#engine = db_connect()

# your code here

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Descargar diccionarios de NLTK necesarios
nltk.download('stopwords')
nltk.download('wordnet')

# PASO 1: Carga del conjunto de datos
# Nota: Si el enlace original falla, puedes usar el repositorio oficial de 4Geeks
url_data = "https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv"
df = pd.read_csv(url_data)

# PASO 2: Preprocesamiento de los enlaces
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_url(url):
    # Segmentar por signos de puntuación y pasar a minúsculas
    words = re.split(r'\W+', str(url).lower())
    # Eliminar stopwords, lematizar y quitar caracteres vacíos
    clean_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return " ".join(clean_words)

# Aplicar limpieza a la columna 'url'
df['clean_url'] = df['url'].apply(preprocess_url)

# Convertir el texto limpio a números (Vectorización) para que el SVM lo entienda
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_url'])
y = df['is_spam'] # Etiqueta objetivo: True(Spam) o False(No Spam)

# Dividir en train (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PASO 3: Construir y entrenar un SVM (Parámetros por defecto)
model = SVC(random_state=42)
model.fit(X_train, y_train)

# Evaluar modelo base
y_pred = model.predict(X_test)
print(f"Precisión del modelo base: {accuracy_score(y_test, y_pred):.4f}")

# PASO 4: Optimizar el modelo anterior (Grid Search)
# Probamos distintas combinaciones de hiperparámetros
param_grid = {
    'C': [0.1, 1, 10], 
    'kernel': ['linear', 'rbf']
}

print("Iniciando optimización (esto puede tardar un poco)...")
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=3)
grid.fit(X_train, y_train)

print(f"Mejores hiperparámetros encontrados: {grid.best_params_}")
print(f"Precisión del modelo optimizado: {grid.best_score_:.4f}")

# PASO 5: Guardar el modelo
# Guardamos tanto el modelo optimizado como el vectorizador
best_model = grid.best_estimator_
joblib.dump(best_model, 'svm_spam_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("¡Modelo y vectorizador guardados con éxito!")