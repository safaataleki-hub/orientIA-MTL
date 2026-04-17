"""
Pipeline ML CTAS — Random Forest
==================================
Étape 1 : Génération de 10 000 patients synthétiques cliniquement cohérents
Étape 2 : Entraînement d'un Random Forest multi-classe (CTAS 1-5)
Étape 3 : Évaluation (accuracy, rapport de classification, matrice de confusion)
Étape 4 : Sauvegarde du modèle pour intégration dans l'orchestrateur

Stratégie de génération :
  - Chaque patient est généré à partir d'un profil de base par niveau CTAS
  - Du bruit gaussien est ajouté pour simuler la variabilité clinique réelle
  - Les distributions sont calibrées sur les seuils du guide CTAS officiel
  - Le label final est validé par le moteur de règles (CTASEngine) pour
    garantir la cohérence entre les données et la logique métier
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Importer le moteur de règles pour valider les labels
import sys
sys.path.insert(0, '/home/claude')
from ctas_engine import CTASEngine, PatientInput

np.random.seed(42)


# ─────────────────────────────────────────────
#  PROFILS CLINIQUES PAR NIVEAU CTAS
#  Chaque profil définit les distributions des features
#  (moyenne, écart-type) calibrées sur le guide officiel
# ─────────────────────────────────────────────

PROFILS_CTAS = {
    1: {
        "description": "Réanimation — menace vitale immédiate",
        "poids": 0.08,   # ~8% des visites aux urgences
        "age":        {"mean": 58, "std": 20},
        "spo2":       {"mean": 82, "std": 6,  "clip": (50, 89)},
        "fc":         {"mean": 135,"std": 25, "clip": (30, 200)},
        "tension":    {"mean": 78, "std": 18, "clip": (40, 100)},
        "temperature":{"mean": 38.2,"std": 1.2,"clip": (35, 42)},
        "glycemie":   {"mean": 8.0, "std": 6.0,"clip": (1.0, 35)},
        "douleur":    {"mean": 9.0, "std": 1.0,"clip": (6, 10)},
        "flags": {
            "inconscient":      0.55,
            "convulsion":       0.20,
            "choc":             0.65,
            "confusion":        0.60,
            "saignement":       0.30,
            "immunosupprime":   0.10,
            "sirs":             {"mean": 3.0, "std": 0.3, "clip": (2, 3)},
        },
        # CTAS 1 : plaintes spécifiques + plaintes ambiguës avec signes vitaux critiques
        "plaintes_possibles": [
            "P1_01","P1_02","P1_03","P1_04","P1_05",
            "P1_06","P1_07","P1_08","P1_09","P1_10",
            "P1_11","P1_12","P1_13",
            # Plaintes ambiguës qui deviennent CTAS 1 via signes vitaux critiques
            "P2_03","P3_03","P3_06","P4_06",
        ],
    },
    2: {
        "description": "Emergent — menace potentielle",
        "poids": 0.18,
        "age":        {"mean": 55, "std": 18},
        "spo2":       {"mean": 91, "std": 2,  "clip": (88, 93)},
        "fc":         {"mean": 118,"std": 15, "clip": (95, 150)},
        "tension":    {"mean": 155,"std": 35, "clip": (85, 230)},
        "temperature":{"mean": 38.8,"std": 0.8,"clip": (36, 41)},
        "glycemie":   {"mean": 7.0, "std": 5.0,"clip": (1.5, 25)},
        "douleur":    {"mean": 7.5, "std": 1.5,"clip": (5, 10)},
        "flags": {
            "inconscient":      0.05,
            "convulsion":       0.05,
            "choc":             0.20,
            "confusion":        0.35,
            "saignement":       0.20,
            "immunosupprime":   0.15,
            "sirs":             {"mean": 2.2, "std": 0.5, "clip": (1, 3)},
        },
        "plaintes_possibles": [
            "P2_01","P2_02","P2_03","P2_04","P2_05",
            "P2_06","P2_07","P2_08","P2_09","P2_10",
            # Plaintes ambiguës upgradées par signes vitaux
            "P3_01","P3_03","P3_04","P3_06","P3_07",
            "P4_02","P4_03","P4_06",
        ],
    },
    3: {
        "description": "Urgent — problème sérieux potentiel",
        "poids": 0.32,
        "age":        {"mean": 42, "std": 18},
        "spo2":       {"mean": 94, "std": 2,  "clip": (91, 97)},
        "fc":         {"mean": 98, "std": 14, "clip": (70, 135)},
        "tension":    {"mean": 138,"std": 22, "clip": (100, 200)},
        "temperature":{"mean": 38.0,"std": 0.8,"clip": (36.5, 40)},
        "glycemie":   {"mean": 6.2, "std": 3.0,"clip": (3.0, 20)},
        "douleur":    {"mean": 6.0, "std": 1.5,"clip": (3, 9)},
        "flags": {
            "inconscient":      0.01,
            "convulsion":       0.02,
            "choc":             0.05,
            "confusion":        0.10,
            "saignement":       0.08,
            "immunosupprime":   0.10,
            "sirs":             {"mean": 1.2, "std": 0.6, "clip": (0, 2)},
        },
        "plaintes_possibles": [
            "P3_01","P3_02","P3_03","P3_04",
            "P3_05","P3_06","P3_07","P3_08",
            # Plaintes CTAS 4/5 upgradées par contexte clinique
            "P4_01","P4_02","P4_03","P4_04","P4_05","P4_06",
            "P5_03","P5_04",
        ],
    },
    4: {
        "description": "Moins urgent — stable",
        "poids": 0.28,
        "age":        {"mean": 35, "std": 18},
        "spo2":       {"mean": 97, "std": 1.5,"clip": (94, 100)},
        "fc":         {"mean": 85, "std": 12, "clip": (60, 115)},
        "tension":    {"mean": 122,"std": 18, "clip": (95, 175)},
        "temperature":{"mean": 37.4,"std": 0.7,"clip": (36.5, 39.5)},
        "glycemie":   {"mean": 5.5, "std": 1.5,"clip": (3.5, 12)},
        "douleur":    {"mean": 3.5, "std": 1.5,"clip": (1, 7)},
        "flags": {
            "inconscient":      0.00,
            "convulsion":       0.00,
            "choc":             0.01,
            "confusion":        0.02,
            "saignement":       0.03,
            "immunosupprime":   0.05,
            "sirs":             {"mean": 0.4, "std": 0.5, "clip": (0, 1)},
        },
        "plaintes_possibles": [
            "P4_01","P4_02","P4_03","P4_04","P4_05","P4_06",
            # Plaintes CTAS 5 avec léger contexte
            "P5_01","P5_02","P5_03","P5_04",
            # Plaintes CTAS 3 stables
            "P3_02","P3_05","P3_07",
        ],
    },
    5: {
        "description": "Non urgent — mineur",
        "poids": 0.14,
        "age":        {"mean": 30, "std": 16},
        "spo2":       {"mean": 98, "std": 1.0,"clip": (96, 100)},
        "fc":         {"mean": 76, "std": 10, "clip": (55, 100)},
        "tension":    {"mean": 115,"std": 12, "clip": (90, 145)},
        "temperature":{"mean": 37.0,"std": 0.4,"clip": (36.5, 38.4)},
        "glycemie":   {"mean": 5.2, "std": 1.0,"clip": (4.0, 8.0)},
        "douleur":    {"mean": 2.0, "std": 1.0,"clip": (0, 4)},
        "flags": {
            "inconscient":      0.00,
            "convulsion":       0.00,
            "choc":             0.00,
            "confusion":        0.00,
            "saignement":       0.01,
            "immunosupprime":   0.03,
            "sirs":             {"mean": 0.1, "std": 0.3, "clip": (0, 1)},
        },
        "plaintes_possibles": [
            "P5_01","P5_02","P5_03","P5_04",
            "P4_03","P4_04","P4_05","P4_06",
        ],
    },
}

# Mapping plainte → catégorie numérique pour le modèle
PLAINTE_CATEGORIES = {
    "P1_01":0,"P1_02":1,"P1_03":2,"P1_04":3,"P1_05":4,
    "P1_06":5,"P1_07":6,"P1_08":7,"P1_09":8,"P1_10":9,
    "P1_11":10,"P1_12":11,"P1_13":12,
    "P2_01":13,"P2_02":14,"P2_03":15,"P2_04":16,"P2_05":17,
    "P2_06":18,"P2_07":19,"P2_08":20,"P2_09":21,"P2_10":22,
    "P3_01":23,"P3_02":24,"P3_03":25,"P3_04":26,
    "P3_05":27,"P3_06":28,"P3_07":29,"P3_08":30,
    "P4_01":31,"P4_02":32,"P4_03":33,"P4_04":34,"P4_05":35,"P4_06":36,
    "P5_01":37,"P5_02":38,"P5_03":39,"P5_04":40,
}


# ─────────────────────────────────────────────
#  GÉNÉRATEUR DE DONNÉES SYNTHÉTIQUES
# ─────────────────────────────────────────────

def generer_patient(ctas_cible: int, profil: dict) -> dict:
    """Génère un patient synthétique pour un niveau CTAS donné."""

    def sample(params):
        val = np.random.normal(params["mean"], params["std"])
        return float(np.clip(val, params["clip"][0], params["clip"][1]))

    def sample_bool(prob):
        return int(np.random.random() < prob)

    age  = int(np.clip(np.random.normal(profil["age"]["mean"], profil["age"]["std"]), 1, 100))
    sexe = int(np.random.random() < 0.5)  # 0=F, 1=M

    spo2      = sample(profil["spo2"])
    fc        = sample(profil["fc"])
    tension   = sample(profil["tension"])
    temp      = sample(profil["temperature"])
    glycemie  = sample(profil["glycemie"])
    douleur   = sample(profil["douleur"])

    flags     = profil["flags"]
    inconscient   = sample_bool(flags["inconscient"])
    convulsion    = sample_bool(flags["convulsion"])
    choc          = sample_bool(flags["choc"])
    confusion     = sample_bool(flags["confusion"])
    saignement    = sample_bool(flags["saignement"])
    immunosupprime= sample_bool(flags["immunosupprime"])
    sirs          = int(np.clip(
        np.random.normal(flags["sirs"]["mean"], flags["sirs"]["std"]),
        flags["sirs"]["clip"][0], flags["sirs"]["clip"][1]
    ))

    douleur_centrale   = int(np.random.random() < (0.7 if ctas_cible <= 2 else 0.4))
    douleur_aigue      = int(np.random.random() < (0.9 if ctas_cible <= 3 else 0.5))

    plainte = np.random.choice(profil["plaintes_possibles"])
    plainte_cat = PLAINTE_CATEGORIES[plainte]

    return {
        "age": age,
        "sexe": sexe,
        "spo2": round(spo2, 1),
        "frequence_cardiaque": int(fc),
        "tension_systolique": int(tension),
        "temperature": round(temp, 1),
        "glycemie": round(glycemie, 1),
        "douleur_intensite": round(douleur, 1),
        "douleur_centrale": douleur_centrale,
        "douleur_aigue": douleur_aigue,
        "inconscient": inconscient,
        "convulsion_active": convulsion,
        "choc_hemodynamique": choc,
        "confusion_soudaine": confusion,
        "saignement_abondant": saignement,
        "immunosupprime": immunosupprime,
        "sirs_criteres": sirs,
        "presentation_id": plainte,
        "plainte_categorie": plainte_cat,
        "ctas_label": ctas_cible,
    }


def generer_dataset(n_total: int = 10000) -> pd.DataFrame:
    """Génère le dataset complet avec distribution réaliste par niveau CTAS."""
    print(f"Génération de {n_total} patients synthétiques...")
    records = []

    for ctas_niveau, profil in PROFILS_CTAS.items():
        n_niveau = int(n_total * profil["poids"])
        print(f"  CTAS {ctas_niveau} ({profil['description']}) : {n_niveau} patients")
        for _ in range(n_niveau):
            records.append(generer_patient(ctas_niveau, profil))

    # Compléter si arrondi
    while len(records) < n_total:
        records.append(generer_patient(3, PROFILS_CTAS[3]))

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Total généré : {len(df)} patients\n")
    return df


# ─────────────────────────────────────────────
#  FEATURES ET ENTRAÎNEMENT
# ─────────────────────────────────────────────

FEATURES = [
    "age", "sexe",
    "spo2", "frequence_cardiaque", "tension_systolique",
    "temperature", "glycemie",
    "douleur_intensite", "douleur_centrale", "douleur_aigue",
    "inconscient", "convulsion_active", "choc_hemodynamique",
    "confusion_soudaine", "saignement_abondant",
    "immunosupprime", "sirs_criteres",
    "plainte_categorie",
]

TARGET = "ctas_label"


def entrainer_modele(df: pd.DataFrame):
    """Entraîne le Random Forest et retourne le modèle + métriques."""

    X = df[FEATURES]
    y = df[TARGET]

    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Entraînement du Random Forest...")
    print(f"  Train : {len(X_train)} patients")
    print(f"  Test  : {len(X_test)} patients\n")

    model = RandomForestClassifier(
        n_estimators=200,       # 200 arbres — bon équilibre perf/temps
        max_depth=15,           # éviter l'overfitting
        min_samples_leaf=5,
        class_weight="balanced",# compenser le déséquilibre CTAS 1 vs 5
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ── Évaluation ──
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("=" * 55)
    print("  RÉSULTATS D'ÉVALUATION")
    print("=" * 55)
    print(f"\n  Accuracy globale : {acc:.3f} ({acc*100:.1f}%)\n")

    print("  Rapport par niveau CTAS :")
    print(classification_report(
        y_test, y_pred,
        target_names=[f"CTAS {i}" for i in range(1, 6)]
    ))

    print("  Matrice de confusion :")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Réel CTAS {i}" for i in range(1, 6)],
        columns=[f"Prédit CTAS {i}" for i in range(1, 6)]
    )
    print(cm_df.to_string())

    # ── Validation croisée ──
    print(f"\n  Validation croisée (5-fold) :")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    print(f"  Scores  : {[round(s, 3) for s in cv_scores]}")
    print(f"  Moyenne : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Importance des features ──
    print(f"\n  Top 10 features les plus importantes :")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    importances = importances.sort_values(ascending=False)
    for feat, imp in importances.head(10).items():
        bar = "█" * int(imp * 100)
        print(f"    {feat:<25} {imp:.3f}  {bar}")

    print("\n" + "=" * 55)

    return model, {
        "accuracy": acc,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "feature_importances": importances.to_dict(),
        "classes": list(model.classes_),
    }


# ─────────────────────────────────────────────
#  SAUVEGARDE
# ─────────────────────────────────────────────

def sauvegarder_modele(model, features: list, path: str = "/home/claude/ctas_rf_model.pkl"):
    """Sauvegarde le modèle + métadonnées pour l'orchestrateur."""
    payload = {
        "model": model,
        "features": features,
        "classes": list(model.classes_),
        "plainte_categories": PLAINTE_CATEGORIES,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Modèle sauvegardé → {path}")


# ─────────────────────────────────────────────
#  FONCTION DE PRÉDICTION (pour l'orchestrateur)
# ─────────────────────────────────────────────

def predire_ctas(patient: PatientInput, model_path: str = "/home/claude/ctas_rf_model.pkl") -> dict:
    """
    Prédit les probabilités CTAS à partir d'un PatientInput.
    Retourne un dict compatible avec le placeholder de l'orchestrateur.
    """
    with open(model_path, "rb") as f:
        payload = pickle.load(f)

    model   = payload["model"]
    features= payload["features"]
    plainte_cat = payload["plainte_categories"]

    # Construire le vecteur de features
    row = {
        "age":                  getattr(patient, "age", 40) or 40,
        "sexe":                 getattr(patient, "sexe", 0) or 0,
        "spo2":                 patient.spo2 or 98.0,
        "frequence_cardiaque":  patient.frequence_cardiaque or 80,
        "tension_systolique":   patient.tension_systolique or 120,
        "temperature":          patient.temperature or 37.0,
        "glycemie":             patient.glycemie or 5.5,
        "douleur_intensite":    patient.douleur_intensite or 0,
        "douleur_centrale":     int(patient.douleur_centrale or False),
        "douleur_aigue":        int(patient.douleur_aigue or False),
        "inconscient":          int(patient.inconscient),
        "convulsion_active":    int(patient.convulsion_active),
        "choc_hemodynamique":   int(patient.choc_hemodynamique),
        "confusion_soudaine":   int(patient.confusion_soudaine),
        "saignement_abondant":  int(patient.saignement_abondant),
        "immunosupprime":       int(patient.immunosupprime),
        "sirs_criteres":        patient.sirs_criteres or 0,
        "plainte_categorie":    plainte_cat.get(patient.presentation_id, 25),
    }

    X = pd.DataFrame([row])[features]
    probas = model.predict_proba(X)[0]
    classes = payload["classes"]

    return {
        "statut": "ok",
        "probabilites": {int(cls): round(float(p), 3) for cls, p in zip(classes, probas)},
        "prediction": int(classes[probas.argmax()]),
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Générer le dataset
    df = generer_dataset(n_total=10000)

    # Aperçu de la distribution
    print("Distribution par niveau CTAS :")
    dist = df["ctas_label"].value_counts().sort_index()
    for ctas, count in dist.items():
        pct = count / len(df) * 100
        print(f"  CTAS {ctas} : {count:5d} patients ({pct:.1f}%)")
    print()

    # Sauvegarder le dataset
    df.to_csv("/home/claude/ctas_dataset.csv", index=False)
    print("  Dataset sauvegardé → /home/claude/ctas_dataset.csv\n")

    # 2. Entraîner le modèle
    model, metriques = entrainer_modele(df)

    # 3. Sauvegarder
    sauvegarder_modele(model, FEATURES)

    # 4. Test de la fonction de prédiction
    print("\n  Test de prédiction sur un cas réel :")
    test_patient = PatientInput(
        presentation_id="P3_03",
        douleur_intensite=7,
        douleur_centrale=True,
        douleur_aigue=True,
        temperature=38.8,
        frequence_cardiaque=105,
        spo2=96.0,
    )
    resultat = predire_ctas(test_patient)
    print(f"  Patient : douleur abdominale 7/10, fièvre 38.8°C")
    print(f"  Prédiction ML : CTAS {resultat['prediction']}")
    print(f"  Probabilités  :")
    for niveau, prob in sorted(resultat["probabilites"].items()):
        bar = "█" * int(prob * 40)
        print(f"    CTAS {niveau} : {prob:.3f}  {bar}")
