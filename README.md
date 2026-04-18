# OrientIA MTL 🏥

**Le bon soin, au bon endroit, au bon moment.**

Système d'orientation intelligent pour les urgences de Montréal, Projet académique TECH 60711 H2026, HEC Montréal.

## Description

OrientIA guide le patient vers le bon point de service en combinant une interface conversationnelle en langage naturel, une évaluation clinique basée sur l'échelle CTAS, et des données d'achalandage en temps réel.

## Démo en ligne

👉 [Accéder au prototype](https://orientIA-MTL.onrender.com)

> Le prototype nécessite une clé API Groq gratuite. Obtenez-la en 2 minutes sur [console.groq.com](https://console.groq.com)

## Architecture IA — 4 couches

| Couche | Technologie | Rôle |
|---|---|---|
| NLP | LLaMA 3.1 via Groq API | Interprétation des symptômes en langage naturel |
| Classification | Random Forest (scikit-learn) | Niveau d'urgence CTAS 1→5 |
| Prédiction | XGBoost | Durée de séjour estimée |
| Recommandation | TOPSIS (Python) | Top 3 établissements |

## Structure du projet

```
orientIA/
├── index.html              ← Prototype frontend complet (PWA)
├── README.md
├── .gitignore
├── models/
│   ├── ctas_engine.py      ← Moteur de règles CTAS
│   ├── ctas_ml_pipeline.py ← Pipeline Random Forest
│   ├── ctas_rf_model.pkl   ← Modèle entraîné (accuracy 93.8%)
│   ├── recommendation_engine.py ← Moteur TOPSIS
│   └── points_de_service.py
└── data/
    ├── Base_CTAS_TriageAI_v3.xlsx     ← 71 présentations cliniques
    └── Etablissements_Montreal_TriageAI.xlsx ← 45 établissements
```

## Utilisation du prototype

1. Ouvrir le lien de démo
2. Entrer votre clé API Groq gratuite
3. Suivre le parcours : profil → sécurité → chatbot → recommandations

## Sources de données

| Source | Usage | Accès |
|---|---|---|
| ICIS / SNISA | Entraînement Random Forest + XGBoost | Sur demande institutionnelle |
| CSV MSSS | Achalandage urgences temps réel | [Public](https://www.msss.gouv.qc.ca) |
| BDOESS Statistique Canada | Coordonnées GPS établissements | [Public](https://www.statcan.gc.ca) |
| Documentation CTAS (ACMU) | Base de règles cliniques | [Public](https://caep.ca/resources/ctas) |

## Stack technique

- **Frontend** : HTML + CSS + JavaScript (PWA)
- **Chatbot** : API Groq — LLaMA 3.1 8B Instant
- **Modèles ML** : scikit-learn, XGBoost, Python 3.11
- **Déploiement** : Render (site statique)

## Avertissement

⚠️ **Prototype académique uniquement.** Ne remplace pas un professionnel de santé.
En cas d'urgence, appelez le **911**. Pour Info-Santé, appelez le **811**.

---

