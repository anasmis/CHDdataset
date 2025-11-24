# 🫀 Prédiction des Maladies Coronariennes (CHD) - Projet ML

Ce projet implémente un système de machine learning complet pour prédire les maladies coronariennes (CHD) basé sur des données physiologiques et démographiques.

## 📋 Description

L'application utilise des techniques avancées de machine learning incluant:
- Préprocessing automatisé des données
- Analyse en Composantes Principales (ACP)
- Comparaison de modèles (Régression Logistique vs KNN)
- Interface web interactive avec Streamlit

## 🚀 Installation et Exécution

### Prérequis
- Python 3.8+
- pip

### Installation locale
```bash
# Cloner le repository
git clone [votre-repo]
cd chd-prediction

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run streamlit_app.py
```

### Déploiement Cloud

#### Streamlit Cloud
1. Fork ce repository sur GitHub
2. Aller sur [share.streamlit.io](https://share.streamlit.io/)
3. Connecter votre compte GitHub
4. Sélectionner ce repository et `streamlit_app.py`
5. Déployer

#### Heroku
```bash
# Créer une application Heroku
heroku create votre-app-name

# Déployer
git push heroku main
```

## 📊 Structure du Projet

```
chd-prediction/
├── serie4.ipynb          # Notebook Jupyter complet
├── streamlit_app.py      # Application web Streamlit
├── Model.pkl             # Modèle ML entraîné
├── model_info.json       # Métadonnées du modèle
├── requirements.txt      # Dépendances Python
├── Procfile             # Configuration Heroku
├── setup.sh             # Script de configuration
└── README.md            # Documentation
```

## 🔬 Méthodologie

### Données
- **Source**: Dataset CHD (463 échantillons)
- **Variables**: sbp, ldl, adiposity, famhist, obesity, age
- **Cible**: Présence/absence de maladie coronarienne

### Pipeline ML
1. **Préprocessing**:
   - Variables numériques: Imputation (médiane) + Standardisation
   - Variables catégorielles: Uniformisation + One-Hot Encoding

2. **Modélisation**:
   - Régression Logistique avec/sans ACP
   - KNN avec SMOTE pour rééquilibrage
   - Optimisation des hyperparamètres (GridSearchCV)

3. **Validation**:
   - Division train/test (67%/33%)
   - Validation croisée à 5 plis
   - Métriques: Précision, Rappel, F1-score

### Performances
- **Précision générale**: 70.6%
- **Réduction de dimension**: 6 → 5 variables (95.8% variance conservée)
- **Modèle final**: Régression Logistique avec ACP

## 🎯 Utilisation de l'Application

### Interface Web
L'application Streamlit propose:
- **Saisie intuitive** des données patient
- **Prédiction en temps réel** du risque CHD
- **Visualisation** des probabilités
- **Recommandations** personnalisées

### Variables d'Entrée
- **Pression Artérielle Systolique** (90-250 mmHg)
- **Cholestérol LDL** (0-15 mmol/L)
- **Adiposité** (5-50)
- **Obésité (BMI)** (15-50)
- **Âge** (15-80 ans)
- **Antécédents Familiaux** (Absent/Present)

## ⚠️ Avertissement

Cette application est développée à des fins **éducatives uniquement**. Elle ne doit pas être utilisée pour des diagnostics médicaux réels. Consultez toujours un professionnel de santé pour toute question médicale.

## 📧 Contact

Pour toute question ou suggestion concernant ce projet, n'hésitez pas à ouvrir une issue sur GitHub.

---
*Développé dans le cadre du cours de Machine Learning - EMI 2025*
