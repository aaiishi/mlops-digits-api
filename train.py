from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Charger les données
digits = load_digits()
X = digits.data
y = digits.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Évaluer
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Sauvegarder
joblib.dump(model, "model.joblib")