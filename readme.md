# 🧠 Analyse de Sentiment Trustpilot - Pipeline MLOps End-to-End

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Nginx](https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)
![LightGBM](https://img.shields.io/badge/LightGBM-Machine_Learning-yellow?style=for-the-badge)

Ce projet est une application complète d'analyse de sentiment (Positif, Neutre, Négatif) basée sur des avis clients Trustpilot. Il ne s'agit pas d'un simple script de Data Science, mais d'une **architecture MLOps de production**, incluant une API REST sécurisée, une API Gateway (Nginx), et une interface utilisateur.

---

## ✨ Fonctionnalités Principales

* **🤖 Modèle de Machine Learning** : Prédiction du sentiment via un modèle **LightGBM** et un vectoriseur **TF-IDF** pré-entraînés.
* **🚀 API REST** : Développée avec **FastAPI**, offrant des routes de prédiction et de gestion des utilisateurs.
* **🛡️ API Gateway & Sécurité (Nginx)** : 
  * **Rate Limiting** : Protection anti-spam configurée à 2 requêtes/seconde.
  * **Role-Based Access Control (RBAC)** : Nginx utilise le module `auth_request` pour vérifier les rôles avant d'autoriser l'accès au modèle.
* **🚦 Gestion des Quotas Métiers** : Les utilisateurs standards (`user`) sont limités à 5 prédictions par jour. Les administrateurs (`admin`) ont un accès illimité.
* **🎨 Interface Graphique** : Frontend interactif développé avec **Streamlit**, incluant un système de connexion et d'inscription complet.
* **🐳 Conteneurisation** : L'ensemble des services (API, Proxy, Frontend) est orchestré par **Docker Compose** pour un déploiement "Plug & Play".

---

## 🏗️ Architecture du Projet

Le projet est divisé en 3 microservices communiquant au sein d'un réseau Docker isolé :

1. **Frontend (Streamlit)** : Interface utilisateur (Port `8501`).
2. **Reverse Proxy (Nginx)** : Point d'entrée sécurisé de l'API (Port `8080`). Intercepte les requêtes, gère le flux (Rate Limiting) et vérifie les autorisations.
3. **Backend API (FastAPI)** : Coeur de l'application (Port `80` en interne). Gère la logique métier, la base utilisateurs (`users.json`), et l'inférence du modèle IA.

---

## 🚀 Installation & Démarrage (Local ou VPS)

### Prérequis
* [Docker](https://docs.docker.com/get-docker/) et [Docker Compose](https://docs.docker.com/compose/install/) installés sur votre machine ou serveur.

### Étapes

1. **Cloner le repository :**
   ```bash
   git clone [https://github.com/VOTRE_NOM/VOTRE_REPO.git](https://github.com/VOTRE_NOM/VOTRE_REPO.git)
   cd VOTRE_REPO