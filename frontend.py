import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import altair as alt 
import shap
import requests

# --- 0. CONFIGURATION MLOPS & STREAMLIT ---
# 🌍 L'adresse de ton API à travers le Vigile Nginx
# Mets "http://nginx:80" si Streamlit tourne DANS Docker, ou "http://localhost:8080" si Streamlit tourne sur ton PC
API_URL = "http://nginx:80"

st.set_page_config(
    page_title="Trustpilot Sentiment IA",
    page_icon="⭐",
    layout="wide"
)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 🧠 MÉMOIRE DE STREAMLIT (Auth) ---
if "token" not in st.session_state:
    st.session_state["token"] = None
if "role" not in st.session_state:
    st.session_state["role"] = None
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# --- 1. CHARGEMENT DES RESSOURCES (CACHE) ---
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'punkt_tab']
    for res in resources:
        try:
            nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

download_nltk_resources()

# On charge le modèle localement UNIQUEMENT pour l'explicabilité (SHAP) et le Bulk CSV
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load('trustpilot_lgbm_model.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        return None, None

model, vectorizer = load_model_assets()

# --- 2. PIPELINE DE NETTOYAGE (Pour SHAP local) ---
stop_words = set(stopwords.words('english'))
stop_words.update([",", ".", "``", "@", "*", "(", ")", "[","]", "...", "-", "_", ">", "<", ":", "/", "//", "///", "=", "--", "©", "~", ";", "\\", "\\\\", '"', "'","''", '""' "'m", "'ve", "n't","!","?", "'re", "rd", "'s", "%"])
lemmatizer = WordNetLemmatizer()

def processing_pipeline(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"\.+", '', text)
    text = re.sub(r"/", ' ', text)
    text = re.sub(r"[0-9]+", '', text)
    try:
        tokens = word_tokenize(text, language='english')
    except:
        tokens = text.split()
    
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            cleaned_tokens.append(lemma)
    return " ".join(cleaned_tokens)

# --- 3. SIDEBAR : AUTHENTIFICATION & INFOS ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
    
    if st.session_state["token"] is None:
        choix = st.radio("Accès Application", ["Connexion", "Inscription"])
        st.divider()
        
        if choix == "Inscription":
            st.header("📝 Créer un compte")
            new_username = st.text_input("Nouveau pseudo")
            new_password = st.text_input("Nouveau mot de passe", type="password")
            
            if st.button("S'inscrire"):
                if new_username and new_password:
                    payload = {"username": new_username, "password": new_password}
                    res_creation = requests.post(f"{API_URL}/login", json=payload)
                    if res_creation.status_code == 200:
                        st.success("✅ Compte créé ! Connectez-vous.")
                    elif res_creation.status_code == 400:
                        st.error("❌ Ce nom d'utilisateur existe déjà.")
                    else:
                        st.error("❌ Erreur serveur.")
                else:
                    st.warning("Remplissez tous les champs.")

        elif choix == "Connexion":
            st.header("🔒 Connexion")
            username = st.text_input("Pseudo")
            password = st.text_input("Mot de passe", type="password")
            
            if st.button("Se connecter"):
                response = requests.post(f"{API_URL}/token_API", json={"username": username, "password": password})
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["token"] = data["access_token"]
                    st.session_state["role"] = data["role"]
                    st.success("Connexion réussie !")
                    st.rerun()
                else:
                    st.error("❌ Identifiants incorrects.")
    else:
        st.success(f"Connecté en tant que : {st.session_state['role'].upper()}")
        if st.button("Se déconnecter"):
            st.session_state["token"] = None
            st.session_state["role"] = None
            st.rerun()

    st.markdown("---")
    st.header("🔍 Infos du Modèle")
    st.info("Modèle : LightGBM")
    st.write("Vectorisation : TF-IDF")
    st.metric(label="Précision (Accuracy)", value="71.8%", delta="+vs Baseline")
    st.markdown("---")
    st.caption("Projet DataScientest\nLakdar & Aurore")

# --- 4. TITRE PRINCIPAL ---
st.title("🛍️ Analyse de Sentiment & Expérience Client")
st.markdown("Application de démonstration pour la prédiction de satisfaction à partir d'avis textuels.")

# --- 5. L'APPLICATION (Sécurisée) ---
if st.session_state["token"]:
    tab_demo, tab_data, tab_model = st.tabs(["🚀 Démo Live", "📊 Jeu de Données", "🤖 Performance Modèle"])

    # ==============================================================================
    # ONGLET 1 : DÉMO LIVE
    # ==============================================================================
    with tab_demo:
        def set_text(text):
            st.session_state.text_input = text

        st.subheader("Testez l'IA en temps réel")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("😡 Négatif", on_click=set_text, args=["Horrible service, I waited 2 weeks and the package is broken. Never again!"], use_container_width=True)
        with col2:
            st.button("😐 Neutre", on_click=set_text, args=["The product is okay but shipping was a bit slow. Not bad, not great."], use_container_width=True)
        with col3:
            st.button("😍 Positif", on_click=set_text, args=["Absolutely amazing! Best purchase of the year, highly recommended."], use_container_width=True)

        user_input = st.text_area("Votre commentaire :", value=st.session_state.text_input, height=100)

        # --- PREDICTION VIA L'API ---
        if st.button("Lancer l'analyse", type="primary"):
            if user_input.strip():
                with st.spinner('Validation de sécurité et analyse en cours...'):
                    
                    # 1. Requête à l'API via Nginx
                    headers = {"X-API-Key": st.session_state["token"]}
                    payload = {"text": user_input}
                    res = requests.post(f"{API_URL}/predict", headers=headers, json=payload)
                    
                    # 2. Gestion des réponses MLOps (Succès, Quota, Spam)
                    if res.status_code == 200:
                        resultat = res.json()
                        sentiment_api = resultat["sentiment"]
                        confiance_api = resultat["prediction_score"]
                        pred_class = resultat["class_id"]
                        
                        # Couleurs personnalisées
                        color_map = {"Négatif": "red", "Neutre": "orange", "Positif": "green"}
                        emoji_map = {"Négatif": "😞", "Neutre": "😐", "Positif": "😃"}
                        
                        color = color_map.get(sentiment_api, "gray")
                        emoji = emoji_map.get(sentiment_api, "")
                        label_text = f"{sentiment_api} {emoji}"

                        st.divider()
                        c1, c2 = st.columns([1, 2])
                        
                        with c1:
                            st.markdown("### Verdict API :")
                            st.markdown(f":{color}[**{label_text}**]")
                            st.metric("Confiance", confiance_api)
                        
                        with c2:
                            # Pour le graphique de proba, on utilise le modèle local si dispo
                            if model and vectorizer:
                                clean_text = processing_pipeline(user_input)
                                vec_input = vectorizer.transform([clean_text])
                                input_array = vec_input.toarray()
                                pred_proba = model.predict_proba(input_array)[0]
                                
                                st.markdown("#### Probabilités")
                                df_chart = pd.DataFrame({"Sentiment": ["Négatif", "Neutre", "Positif"], "Probabilité": pred_proba})
                                c = alt.Chart(df_chart).mark_bar().encode(
                                    x=alt.X('Sentiment', sort=None),
                                    y='Probabilité',
                                    color=alt.Color('Sentiment', scale=alt.Scale(domain=["Négatif", "Neutre", "Positif"], range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None)
                                )
                                st.altair_chart(c, use_container_width=True)
                            else:
                                st.info("Modèle local non trouvé pour afficher les probabilités détaillées.")
                        
                        # --- VRAI SHAP SIMPLE (Généré en local) ---
                        if model and vectorizer:
                            st.markdown("---")
                            st.subheader("🧠 Analyse SHAP (Impact réel)")
                            st.write(f"Contribution des mots à la décision : **{label_text}**")

                            try:
                                explainer = shap.TreeExplainer(model)
                                shap_values = explainer.shap_values(input_array)
                                
                                vals = None
                                if isinstance(shap_values, list):
                                    if len(shap_values) > pred_class:
                                        vals = shap_values[pred_class][0]
                                    else:
                                        vals = shap_values[0][0]
                                else:
                                    if len(shap_values.shape) == 3 and shap_values.shape[2] > pred_class:
                                        vals = shap_values[0, :, pred_class]
                                    else:
                                        vals = shap_values[0]

                                feature_names = vectorizer.get_feature_names_out()
                                df_shap = pd.DataFrame({
                                    "Mot": feature_names,
                                    "SHAP Value": vals,
                                    "TFIDF": input_array[0]
                                })
                                df_shap = df_shap[df_shap["TFIDF"] > 0]
                                
                                if df_shap.empty:
                                    st.info("Aucun mot-clé connu n'a été détecté dans ce texte.")
                                else:
                                    df_shap["Abs_Value"] = df_shap["SHAP Value"].abs()
                                    df_shap_top = df_shap.sort_values(by="Abs_Value", ascending=False).head(10)
                                    
                                    chart_shap = alt.Chart(df_shap_top).mark_bar().encode(
                                        x=alt.X('SHAP Value', title='Impact sur la décision'),
                                        y=alt.Y('Mot', sort='-x', title='Mots détectés'),
                                        color=alt.condition(
                                            alt.datum["SHAP Value"] > 0,
                                            alt.value("#FF4B4B"),
                                            alt.value("#1E88E5")
                                        ),
                                        tooltip=['Mot', 'SHAP Value']
                                    )
                                    st.altair_chart(chart_shap, use_container_width=True)
                                    st.caption("🟥 Rouge : Pousse vers ce sentiment | 🟦 Bleu : S'y oppose")

                            except Exception as e:
                                st.warning(f"Impossible d'afficher le détail SHAP : {e}")

                    # Gestion des erreurs Nginx / API
                    elif res.status_code == 403:
                        st.error("🛑 Nginx a bloqué la requête : Quota quotidien atteint (5/5) ou accès non autorisé.")
                    elif res.status_code == 503:
                        st.error("🐌 Nginx a bloqué la requête : Vous allez trop vite (Anti-Spam) !")
                    else:
                        st.error(f"Erreur technique de l'API : {res.status_code}")
            else:
                st.warning("Veuillez entrer un texte à analyser.")

        # --- CSV BULK (Seulement pour les Admins) ---
        st.markdown("---")
        st.subheader("📂 Analyse de masse (Fichier CSV)")
        if st.session_state["role"] == "admin":
            if model is None:
                st.error("⚠️ Les modèles locaux sont nécessaires pour le traitement en masse.")
            else:
                csv_template = "text\nExemple: Super produit !\nExemple: Livraison trop longue..."
                st.download_button("📥 Télécharger modèle CSV", csv_template, "modele_avis.csv", "text/csv")
                
                uploaded_file = st.file_uploader("Déposez votre fichier ici", type=["csv"])
                if uploaded_file is not None:
                    try:
                        df = pd.read_csv(uploaded_file)
                        cols = [c for c in df.columns if 'text' in c.lower() or 'review' in c.lower()]
                        if cols:
                            if st.button(f"Analyser {len(df)} avis"):
                                target_col = cols[0]
                                df['clean'] = df[target_col].astype(str).apply(processing_pipeline)
                                vecs = vectorizer.transform(df['clean'])
                                preds = model.predict(vecs.toarray())
                                mapping = {0: "Négatif", 1: "Neutre", 2: "Positif"}
                                df['Prediction'] = [mapping[p] for p in preds]
                                
                                st.dataframe(df[[target_col, 'Prediction']], use_container_width=True)
                                st.download_button("📥 Télécharger résultats", df.to_csv(index=False).encode('utf-8'), "resultats.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Erreur CSV : {e}")
        else:
            st.warning("🔒 L'analyse de masse est réservée aux Administrateurs pour limiter la charge serveur.")

    # ==============================================================================
    # ONGLET 2 : JEU DE DONNÉES
    # ==============================================================================
    with tab_data:
        st.header("📚 Le Jeu de Données : Amazon Electronics")
        col_d1, col_d2 = st.columns([1, 2])
        with col_d1:
            st.markdown("### Source & Pourquoi ?")
            st.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=100)
            st.info("**Amazon Reviews Dataset**")
            st.write("- **Structure identique** : Texte + Note")
            st.write("- **Focus Electronics** : Vocabulaire riche")
            st.write("- **Période** : 2010 - 2018")
        
        with col_d2:
            st.markdown("### Volumétrie & Nettoyage")
            metrics_df = pd.DataFrame({"Métrique": ["Avis bruts", "Avis après filtrage", "Langue"], "Valeur": ["~1.2 Millions", "572 950", "Anglais"]})
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
            c1, c2, c3 = st.columns(3)
            c1.error("❌ **Prix**\n\nSupprimé (trop de NAs)")
            c2.warning("⚠️ **Votes**\n\nImputé à 0 (NAs)")
            c3.info("🖼️ **Image**\n\nBooléen (Y/N)")

        st.divider()
        st.subheader("📋 Aperçu des données brutes (Exemple)")
        example_data = {
            "overall": [5, 1, 3, 5, 2],
            "summary": ["Amazing sound", "Waste of money", "Average", "Great service", "Disappointed"],
            "reviewText": ["This headphone is amazing! The bass is deep.", "Terrible quality, stopped working.", "It's okay for the price.", "Works perfectly, fast delivery.", "Poor screen resolution."],
            "brand": ["Bose", "Generic", "Sony", "Samsung", "LG"]
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)

        st.markdown("---")
        st.subheader("Distribution des classes (Équilibrée)")
        chart_balance = alt.Chart(pd.DataFrame({"Sentiment": ["Négatif", "Neutre", "Positif"], "Nombre": [190983, 190983, 190983]})).mark_bar().encode(
            x=alt.X('Sentiment', sort=None), y='Nombre',
            color=alt.Color('Sentiment', scale=alt.Scale(range=["#6D6D6D", "#FFB7B2", "#FF69B4"]), legend=None)
        )
        st.altair_chart(chart_balance, use_container_width=True)

    # ==============================================================================
    # ONGLET 3 : PERFORMANCE MODÈLE
    # ==============================================================================
    with tab_model:
        st.header("⚙️ Performance LightGBM")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy Globale", "71.8%")
        m2.metric("F1-Score", "0.72")
        m3.metric("Vocabulaire", "5 000 mots")

        st.subheader("Matrice de Confusion")
        confusion_data = pd.DataFrame(
            [[8303, 2155, 558], [2303, 6734, 1979], [551, 1765, 8700]],
            columns=["Prédit Négatif", "Prédit Neutre", "Prédit Positif"],
            index=["Réel Négatif", "Réel Neutre", "Réel Positif"]
        )
        st.dataframe(confusion_data, use_container_width=True)
        
        st.success("✅ **Observation :** Très bonne détection des avis positifs et négatifs.")
        st.warning("⚠️ **Limite :** La classe 'Neutre' (au centre) est celle qui génère le plus de confusion.")
        
        st.markdown("---")
        st.subheader("Global Feature Importance")
        st.write("Mots les plus impactants pour le modèle (Global) :")
        col_feat1, col_feat2 = st.columns(2)
        with col_feat1:
            st.error("📉 **Négatif** : bad, poor, waste, return, money")
        with col_feat2:
            st.success("📈 **Positif** : great, love, good, easy, perfect")

else:
    # Message d'accueil quand on n'est pas connecté
    st.info("👈 Veuillez vous connecter ou créer un compte dans le menu de gauche pour accéder à l'application.")
    
    # On peut quand même afficher les onglets Data et Modèle (Optionnel, c'est stylé pour le jury)
    st.markdown("### 🔒 Accès Restreint")
    st.write("Cette interface est protégée par notre API Gateway. Connectez-vous pour utiliser le modèle prédictif et accéder aux explications SHAP.")