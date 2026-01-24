Guía de uso de RandomForestClassifier
1. Introducción

RandomForestClassifier es un algoritmo de clasificación basado en árboles de decisión, que combina muchos árboles para mejorar la precisión y reducir el sobreajuste (overfitting).

Se basa en la técnica de bagging (Bootstrap Aggregating), donde cada árbol aprende de un subconjunto aleatorio de los datos y características, y la predicción final se realiza por votación mayoritaria entre todos los árboles.

2. Funcionamiento del algoritmo

Construcción del bosque

Se crean N árboles de decisión (n_estimators).

Cada árbol se entrena con un subconjunto aleatorio de muestras del dataset (con reemplazo).

Selección de características aleatorias

En cada nodo de cada árbol, se consideran solo un subconjunto de features (max_features) para dividir.

Esto hace que los árboles sean diversos y no todos aprendan lo mismo.

Predicción

Para clasificación, cada árbol vota por una clase.

La clase con mayoría de votos se devuelve como predicción final.

Para regresión, se promedian los resultados de todos los árboles.


3. Hiperparámetros clave
| Hiperparámetro      | Descripción                                    | Impacto                                                         |
| ------------------- | ---------------------------------------------- | --------------------------------------------------------------- |
| `n_estimators`      | Número de árboles en el bosque                 | Más árboles = mayor precisión, pero más tiempo de entrenamiento |
| `max_depth`         | Profundidad máxima de cada árbol               | Controla el sobreajuste                                         |
| `min_samples_split` | Mínimo número de muestras para dividir un nodo | Evita divisiones muy específicas                                |
| `min_samples_leaf`  | Mínimo número de muestras en una hoja          | Evita que los árboles memoricen datos                           |
| `max_features`      | Número de features a considerar en cada split  | Diversifica los árboles y reduce correlación entre ellos        |
| `bootstrap`         | Si se toman muestras con reemplazo             | True por defecto (bagging)                                      |



4. Consideraciones a tener en cuenta

Datos numéricos y categóricos

RandomForest funciona con ambos, pero las categorías deben estar codificadas (p.ej., OneHotEncoder).

Escalado

No requiere normalización de features, a diferencia de SVM o LogisticRegression.

Computación

Más árboles → más precisión → mayor consumo de memoria y tiempo.

Sobreajuste

Menos frecuente que con un solo árbol, pero puede ocurrir si los árboles son demasiado profundos (max_depth=None).

Interpretabilidad

Es más difícil interpretar que un solo árbol; se puede usar feature_importances_ para entender la importancia de cada variable.

Randomness

Cada ejecución puede generar resultados ligeramente distintos; usar random_state para reproducibilidad.


5. Ejemplo básico de implementación
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Definir el modelo
rf = RandomForestClassifier(random_state=42)

# Definir hiperparámetros a probar
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 20, 40],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt"]
}

# GridSearchCV para seleccionar los mejores hiperparámetros
gs_rf = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy")
gs_rf.fit(X_train, y_train)

# Mejor modelo
best_rf = gs_rf.best_estimator_

# Validación y métricas
accuracy_test = best_rf.score(X_test, y_test)
print("Accuracy en test:", accuracy_test)
6. Recomendaciones para su uso

Comenzar con un número moderado de árboles (100-200) y aumentar si hay tiempo.

Probar distintos valores de max_depth y min_samples_leaf para evitar sobreajuste.

Revisar importancia de variables con best_rf.feature_importances_.

Usar RandomizedSearchCV si el grid es muy grande para ahorrar tiempo.

Siempre usar random_state para reproducibilidad en entornos educativos o de investigación.

![ilustracion](randomForest.png)