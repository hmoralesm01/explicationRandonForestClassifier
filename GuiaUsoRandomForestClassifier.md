# Guía de uso de RandomForestClassifier

## Introducción

**RandomForestClassifier** es un algoritmo de clasificación basado en árboles de decisión que combina múltiples árboles para mejorar la precisión y reducir el sobreajuste (*overfitting*).

Se basa en la técnica de **bagging (Bootstrap Aggregating)**, donde cada árbol aprende a partir de un subconjunto aleatorio de los datos y de las características. La predicción final se obtiene mediante **votación mayoritaria** entre todos los árboles.

---

## Funcionamiento del algoritmo

### 1. Construcción del bosque

- Se crean **N árboles de decisión** (`n_estimators`).
- Cada árbol se entrena con un subconjunto aleatorio del dataset (**muestreo con reemplazo**).

### 2. Selección aleatoria de características

- En cada nodo, solo se considera un subconjunto de variables (`max_features`).
- Esto aumenta la diversidad entre los árboles y reduce la correlación entre ellos.

### 3. Predicción

- **Clasificación:** cada árbol vota por una clase y se devuelve la clase con más votos.
- **Regresión:** se promedia la salida de todos los árboles.

---

## Hiperparámetros clave

| Hiperparámetro       | Descripción                                      | Impacto |
|----------------------|--------------------------------------------------|---------|
| `n_estimators`       | Número de árboles en el bosque                   | Más árboles → mayor precisión, más tiempo |
| `max_depth`          | Profundidad máxima de cada árbol                 | Controla el sobreajuste |
| `min_samples_split`  | Mínimo de muestras para dividir un nodo          | Evita divisiones demasiado específicas |
| `min_samples_leaf`   | Mínimo de muestras en una hoja                   | Reduce la memorización |
| `max_features`       | Features usadas en cada división                 | Diversifica los árboles |
| `bootstrap`          | Muestreo con reemplazo                           | `True` por defecto (bagging) |

---

## Consideraciones a tener en cuenta

### Datos numéricos y categóricos
- RandomForest admite ambos tipos, pero las variables categóricas deben codificarse (por ejemplo, `OneHotEncoder`). Es decir, los datos que vamos a procesar deben de ser numericos, las clases que tenemos que adivinar pueden ser strings.

### Computación
- Un mayor número de árboles incrementa el consumo de memoria y tiempo de entrenamiento.

### Sobreajuste
- Menos frecuente que en un árbol único, pero puede aparecer si `max_depth=None`.

### Interpretabilidad
- Menos interpretable que un solo árbol.
- Se puede analizar la importancia de las variables con `feature_importances_`.

### Aleatoriedad
- Cada ejecución puede variar.
- Usar `random_state` para garantizar reproducibilidad.

---

## Ejemplo de implementación

# Grid de hyperparametros. Va a probar todas las combinaciones.
    # n_estimators → cuántos árboles hay
    # max_depth → qué tan profundos pueden crecer
    # min_samples_leaf → evita árboles “demasiado específicos”
    # max_features → cuántas variables ve cada árbol

rf_param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 40],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

# A traves de GridSearchCV, vamos a instaciar nuestro modelo. EL primer parametro es el tipo de modelo. El segundo la semilla. CV para la validacion cruzada, en este caso seria el 20%
gs_rf = GridSearchCV(
    RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ),
    param_grid=rf_param_grid,
    cv=5,
    verbose=True
)
# Entreno y comprobacion de puntuacion.
gs_rf.fit(X_train, y_train)


