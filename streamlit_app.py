
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.base import BaseEstimator, TransformerMixin

# Classe personnalisée nécessaire pour le chargement du modèle
class FamhistUniformizer(BaseEstimator, TransformerMixin):
    """Transformateur personnalisé pour uniformiser la variable famhist."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        if 'famhist' in X_copy.columns:
            X_copy['famhist'] = X_copy['famhist'].str.capitalize()
        return X_copy

# Configuration de la page
st.set_page_config(
    page_title="Prédiction des Maladies Coronariennes (CHD)",
    page_icon="❤️",
    layout="wide"
)

# Titre principal
st.title("🫀 Prédiction des Maladies Coronariennes (CHD)")
st.markdown("---")

# Chargement du modèle et des informations
@st.cache_resource
def load_model_and_info():
    """Charge le modèle et les informations."""
    try:
        model = joblib.load('Model.pkl')
        with open('model_info.json', 'r', encoding='utf-8') as f:
            info = json.load(f)
        return model, info
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None, None

# Chargement
model, model_info = load_model_and_info()

if model is not None and model_info is not None:

    # Sidebar avec informations sur le modèle
    st.sidebar.header("📊 Informations sur le Modèle")
    st.sidebar.info(f"""
    **Type:** {model_info['model_type']}

    **Variables d'entrée:** {len(model_info['numeric_features']) + len(model_info['categorical_features'])}
    - Numériques: {len(model_info['numeric_features'])}
    - Catégorielles: {len(model_info['categorical_features'])}

    **Classes prédites:** {model_info['target_classes']}
    """)

    # Interface principale
    st.header("📝 Saisie des Données Patient")

    # Création de deux colonnes pour l'interface
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Variables Physiologiques")

        # Variables numériques avec valeurs par défaut réalistes
        sbp = st.number_input(
            "🩺 Pression Artérielle Systolique (mmHg)",
            min_value=90, max_value=250, value=140,
            help="Pression artérielle systolique normale: 90-140 mmHg"
        )

        ldl = st.number_input(
            "🧪 LDL Cholesterol (mmol/L)",
            min_value=0.0, max_value=15.0, value=3.5,
            help="Taux de cholestérol LDL (mauvais cholestérol)"
        )

        adiposity = st.number_input(
            "📏 Adiposité",
            min_value=5.0, max_value=50.0, value=25.0,
            help="Mesure de l'adiposité corporelle"
        )

    with col2:
        st.subheader("Données Démographiques & Style de Vie")

        obesity = st.number_input(
            "⚖️ Obésité (BMI)",
            min_value=15.0, max_value=50.0, value=26.0,
            help="Indice de masse corporelle"
        )

        age = st.number_input(
            "🎂 Âge (années)",
            min_value=15, max_value=80, value=45,
            help="Âge du patient"
        )

        # Variable catégorielle
        famhist = st.selectbox(
            "👨‍👩‍👧‍👦 Antécédents Familiaux de Maladie Cardiaque",
            options=["Absent", "Present"],
            help="Présence d'antécédents familiaux de maladie coronarienne"
        )

    # Bouton de prédiction
    st.markdown("---")

    if st.button("🔍 Effectuer la Prédiction", type="primary"):

        # Construction du dataframe d'entrée
        input_data = pd.DataFrame({
            'sbp': [sbp],
            'ldl': [ldl],
            'adiposity': [adiposity],
            'famhist': [famhist],
            'obesity': [obesity],
            'age': [age]
        })

        try:
            # Prédiction
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            # Affichage des résultats
            st.markdown("---")
            st.header("📊 Résultats de la Prédiction")

            # Colonnes pour les résultats
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                if prediction == 1:
                    st.error("⚠️ **RISQUE ÉLEVÉ DE MALADIE CORONARIENNE**")
                    st.markdown("Le modèle prédit un **risque élevé** de développer une maladie coronarienne.")
                else:
                    st.success("✅ **RISQUE FAIBLE DE MALADIE CORONARIENNE**")
                    st.markdown("Le modèle prédit un **risque faible** de développer une maladie coronarienne.")

            with result_col2:
                st.metric(
                    label="Probabilité de Maladie Coronarienne",
                    value=f"{probability[1]:.1%}",
                    delta=f"Confiance: {max(probability):.1%}"
                )

            # Graphique des probabilités
            st.subheader("📈 Probabilités Détaillées")
            prob_data = pd.DataFrame({
                'Classe': ['Pas de CHD', 'CHD'],
                'Probabilité': probability
            })

            st.bar_chart(prob_data.set_index('Classe'))

            # Recommandations
            st.subheader("💡 Recommandations")
            if prediction == 1:
                st.warning("""
                **Recommandations pour réduire le risque:**
                - Consulter un médecin pour une évaluation complète
                - Surveiller régulièrement la pression artérielle
                - Adopter une alimentation pauvre en cholestérol
                - Pratiquer une activité physique régulière
                - Éviter le tabac et l'alcool en excès
                """)
            else:
                st.info("""
                **Recommandations préventives:**
                - Maintenir un mode de vie sain
                - Contrôles médicaux réguliers
                - Alimentation équilibrée
                - Activité physique régulière
                """)

        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {e}")

    # Informations supplémentaires
    st.markdown("---")
    st.subheader("ℹ️ À propos de cette Application")

    with st.expander("En savoir plus"):
        st.markdown("""
        ### Méthodologie
        - **Préprocessing**: Standardisation des variables numériques, encodage One-Hot des variables catégorielles
        - **Modèle**: Régression Logistique avec réduction de dimension par ACP
        - **Validation**: Validation croisée et test sur données indépendantes

        ### Variables Utilisées
        - **sbp**: Pression artérielle systolique
        - **ldl**: Cholestérol LDL
        - **adiposity**: Adiposité corporelle
        - **famhist**: Antécédents familiaux
        - **obesity**: Indice de masse corporelle
        - **age**: Âge du patient

        ### Avertissement
        ⚠️ Cette application est à des fins éducatives uniquement. 
        Elle ne remplace pas un avis médical professionnel.
        """)

else:
    st.error("❌ Impossible de charger le modèle. Veuillez vérifier que les fichiers Model.pkl et model_info.json sont présents.")

