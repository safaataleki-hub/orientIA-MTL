"""
Moteur de règles CTAS (Canadian Triage and Acuity Scale)
=========================================================
Implémente l'arbre décisionnel officiel du guide Prehospital CTAS v2.0 :
    Étape 1 — Quick Look         : signes vitaux critiques → CTAS 1 immédiat
    Étape 2 — Présentation       : plainte principale → niveau de base
    Étape 3 — Modificateurs 1er ordre généraux : overrides physiologiques
    Étape 4 — Modificateurs 2e ordre spécifiques : raffinement par plainte

Règle fondamentale : toujours retenir le CTAS le plus élevé (chiffre le plus bas).
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  STRUCTURES DE DONNÉES
# ─────────────────────────────────────────────

@dataclass
class PatientInput:
    """
    Données d'entrée du patient.
    Toutes les valeurs sont optionnelles (None = non renseigné).
    """
    # Signes vitaux
    gcs: Optional[int] = None            # Glasgow Coma Scale (3-15)
    spo2: Optional[float] = None         # Saturation O2 en %
    tension_systolique: Optional[int] = None   # mmHg
    frequence_cardiaque: Optional[int] = None  # bpm
    frequence_respiratoire: Optional[int] = None  # /min
    temperature: Optional[float] = None  # °C
    glycemie: Optional[float] = None     # mmol/L

    # Présentation principale (ID de la feuille 01)
    presentation_id: Optional[str] = None

    # Douleur
    douleur_intensite: Optional[int] = None   # 0-10
    douleur_centrale: Optional[bool] = None   # True=centrale, False=périphérique
    douleur_aigue: Optional[bool] = None      # True=aiguë, False=chronique

    # Flags cliniques (questions Oui/Non)
    inconscient: bool = False
    arret_respiratoire: bool = False
    convulsion_active: bool = False
    choc_hemodynamique: bool = False       # peau froide, pouls faible, hypotension
    confusion_soudaine: bool = False
    saignement_abondant: bool = False
    detresse_respiratoire_moderee: bool = False
    reaction_allergique_severe: bool = False

    # Modificateurs spécifiques par présentation
    vomit_sang: bool = False              # pour P3_03 douleur abdominale
    enceinte: bool = False                # pour P3_03
    douleur_irradie_bras_machoire: bool = False  # pour P3_01 douleur thoracique atypique
    deformation_osseuse: bool = False     # pour P4_03 entorse
    fievre_avec_itu: bool = False         # pour P4_02 infection urinaire

    # Contexte complémentaire
    immunosupprime: bool = False          # chimio, greffé, stéroïdes
    sirs_criteres: int = 0               # nb de critères SIRS positifs (0-3)
    avc_delai_heures: Optional[float] = None  # heures depuis début symptômes AVC


@dataclass
class CTASResult:
    """Résultat du triage CTAS."""
    niveau: int                          # 1 à 5
    label: str                           # ex. "CTAS 2 – Emergent"
    raison_principale: str               # déclencheur principal
    raisons_secondaires: list = field(default_factory=list)
    etape_declenchement: str = ""        # Quick Look / Présentation / Modificateur
    recommandation: str = ""


# ─────────────────────────────────────────────
#  DONNÉES DE RÉFÉRENCE
# ─────────────────────────────────────────────

LABELS_CTAS = {
    1: "CTAS 1 – Réanimation",
    2: "CTAS 2 – Emergent",
    3: "CTAS 3 – Urgent",
    4: "CTAS 4 – Moins urgent",
    5: "CTAS 5 – Non urgent",
}

RECOMMANDATIONS = {
    1: "Dirigez-vous immédiatement vers l'urgence la plus proche.",
    2: "Consultez une urgence rapidement (dans les 15 minutes).",
    3: "Urgence ou clinique sans rendez-vous (dans les 30 minutes).",
    4: "Clinique sans rendez-vous ou GMF (dans les 1-2 heures).",
    5: "Clinique ou pharmacie (dans les 24 heures).",
}

# Présentation → niveau CTAS de base
PRESENTATIONS = {
    # CTAS 1
    "P1_01": (1, "Arrêt cardiaque"),
    "P1_02": (1, "Arrêt respiratoire"),
    "P1_03": (1, "Inconscient profond (GCS ≤ 9)"),
    "P1_04": (1, "Convulsion active prolongée"),
    "P1_05": (1, "Polytraumatisme majeur"),
    "P1_06": (1, "Traumatisme crânien grave"),
    "P1_07": (1, "Brûlure majeure (>25% surface)"),
    "P1_08": (1, "Choc hémodynamique"),
    "P1_09": (1, "Choc septique"),
    "P1_10": (1, "Détresse respiratoire sévère"),
    "P1_11": (1, "Anaphylaxie sévère"),
    "P1_12": (1, "Hémorragie massive"),
    "P1_13": (1, "Hypoglycémie sévère avec perte de conscience"),
    # CTAS 2
    "P2_01": (2, "Douleur thoracique suspecte SCA"),
    "P2_02": (2, "AVC suspect"),
    "P2_03": (2, "Dyspnée modérée"),
    "P2_04": (2, "Trauma à haut risque"),
    "P2_05": (2, "Hypoglycémie symptomatique"),
    "P2_06": (2, "Hyperglycémie symptomatique"),
    "P2_07": (2, "Sepsis suspect"),
    "P2_08": (2, "Saignement en grossesse"),
    "P2_09": (2, "Idées suicidaires actives avec plan"),
    "P2_10": (2, "Fièvre chez nourrisson < 3 mois"),
    # CTAS 3
    "P3_01": (3, "Douleur thoracique atypique"),
    "P3_02": (3, "Dyspnée légère"),
    "P3_03": (3, "Douleur abdominale aiguë"),
    "P3_04": (3, "Céphalée sévère"),
    "P3_05": (3, "Fracture suspectée stable"),
    "P3_06": (3, "Fièvre avec comorbidité"),
    "P3_07": (3, "Colique néphrétique"),
    "P3_08": (3, "Douleur en grossesse (2e-3e trimestre)"),
    # CTAS 4
    "P4_01": (4, "Vomissements simples sans déshydratation"),
    "P4_02": (4, "Infection urinaire simple"),
    "P4_03": (4, "Entorse"),
    "P4_04": (4, "Plaie superficielle"),
    "P4_05": (4, "Otite simple"),
    "P4_06": (4, "Toux simple sans détresse"),
    # CTAS 5
    "P5_01": (5, "Renouvellement d'ordonnance"),
    "P5_02": (5, "Éruption bénigne sans fièvre"),
    "P5_03": (5, "Douleur chronique stable"),
    "P5_04": (5, "Rhume simple"),
}


# ─────────────────────────────────────────────
#  MOTEUR DE RÈGLES
# ─────────────────────────────────────────────

class CTASEngine:
    """
    Moteur de règles CTAS.
    Applique l'arbre décisionnel en 4 étapes selon le guide officiel.
    """

    def evaluer(self, patient: PatientInput) -> CTASResult:
        niveau_final = 5
        raison_principale = "Aucune présentation critique détectée"
        raisons_secondaires = []
        etape = "Présentation"

        # ══════════════════════════════════════════
        # ÉTAPE 1 — QUICK LOOK (signes vitaux critiques)
        # Surpasse tout : si déclenché → CTAS 1 immédiat
        # ══════════════════════════════════════════
        quick_look = self._quick_look(patient)
        if quick_look:
            niveau_final = 1
            raison_principale = quick_look[0]
            raisons_secondaires = quick_look[1:]
            etape = "Quick Look"
            return self._construire_resultat(
                niveau_final, raison_principale, raisons_secondaires, etape
            )

        # ══════════════════════════════════════════
        # ÉTAPE 2 — PRÉSENTATION PRINCIPALE
        # ══════════════════════════════════════════
        if patient.presentation_id and patient.presentation_id in PRESENTATIONS:
            niveau_base, desc = PRESENTATIONS[patient.presentation_id]
            if niveau_base < niveau_final:
                niveau_final = niveau_base
                raison_principale = f"Présentation : {desc}"
                etape = "Présentation"

        # ══════════════════════════════════════════
        # ÉTAPE 3 — MODIFICATEURS 1ER ORDRE (physiologiques)
        # Peuvent upgrade le niveau même si présentation est basse
        # ══════════════════════════════════════════
        mods_1er = self._modificateurs_premier_ordre(patient)
        for niveau_mod, desc_mod in mods_1er:
            if niveau_mod < niveau_final:
                niveau_final = niveau_mod
                raison_principale = f"Modificateur : {desc_mod}"
                etape = "Modificateur 1er ordre"
            elif niveau_mod == niveau_final:
                raisons_secondaires.append(desc_mod)

        # ══════════════════════════════════════════
        # ÉTAPE 4 — MODIFICATEURS 2E ORDRE (spécifiques à la présentation)
        # Ne peuvent PAS downgrader un niveau déjà établi
        # ══════════════════════════════════════════
        if patient.presentation_id:
            mods_2e = self._modificateurs_second_ordre(patient)
            for niveau_mod, desc_mod in mods_2e:
                # Cas spécial -1 : AVC hors fenêtre → forcer CTAS 3 si pas déjà plus critique
                if niveau_mod == -1:
                    if niveau_final == 2:
                        niveau_final = 3
                        raisons_secondaires.append(desc_mod)
                        etape = "Modificateur 2e ordre"
                elif niveau_mod < niveau_final:
                    niveau_final = niveau_mod
                    raisons_secondaires.append(f"Modificateur spécifique : {desc_mod}")
                    etape = "Modificateur 2e ordre"

        return self._construire_resultat(
            niveau_final, raison_principale, raisons_secondaires, etape
        )

    # ──────────────────────────────────────────
    #  ÉTAPE 1 : QUICK LOOK
    # ──────────────────────────────────────────
    def _quick_look(self, p: PatientInput) -> list:
        """Retourne une liste de raisons si CTAS 1 détecté, sinon liste vide."""
        raisons = []

        # Conscience
        if p.inconscient or (p.gcs is not None and p.gcs <= 9):
            raisons.append(f"Inconscience / GCS ≤ 9 (GCS={p.gcs})")

        # Respiration
        if p.arret_respiratoire:
            raisons.append("Arrêt respiratoire")
        if p.spo2 is not None and p.spo2 < 90:
            raisons.append(f"SpO2 critique : {p.spo2}% (<90%)")

        # Circulation
        if p.choc_hemodynamique:
            raisons.append("Choc hémodynamique (hypotension + hypoperfusion)")
        if p.tension_systolique is not None and p.tension_systolique < 90:
            raisons.append(f"Tension systolique critique : {p.tension_systolique} mmHg")

        # Convulsion
        if p.convulsion_active:
            raisons.append("Convulsion active en cours")

        # Anaphylaxie
        if p.reaction_allergique_severe:
            raisons.append("Réaction allergique sévère (anaphylaxie)")

        return raisons

    # ──────────────────────────────────────────
    #  ÉTAPE 3 : MODIFICATEURS 1ER ORDRE
    # ──────────────────────────────────────────
    def _modificateurs_premier_ordre(self, p: PatientInput) -> list:
        """
        Retourne une liste de tuples (niveau_ctas, description).
        Applique : SpO2, FC, température, glycémie, GCS, douleur, saignement.
        """
        mods = []

        # --- SpO2 (Table 1 du guide) ---
        if p.spo2 is not None:
            if p.spo2 < 90:
                mods.append((1, f"SpO2 {p.spo2}% — détresse respiratoire sévère"))
            elif p.spo2 < 92:
                mods.append((2, f"SpO2 {p.spo2}% — détresse respiratoire modérée"))
            elif p.spo2 <= 94:
                mods.append((3, f"SpO2 {p.spo2}% — détresse respiratoire légère"))

        # --- Fréquence cardiaque ---
        if p.frequence_cardiaque is not None:
            if p.frequence_cardiaque > 130:
                mods.append((2, f"Tachycardie sévère : {p.frequence_cardiaque} bpm"))

        # --- Tension systolique ---
        if p.tension_systolique is not None:
            if p.tension_systolique >= 220:
                if p.confusion_soudaine or (p.douleur_intensite and p.douleur_intensite > 3):
                    mods.append((2, f"HTA sévère symptomatique : {p.tension_systolique} mmHg"))
                else:
                    mods.append((3, f"HTA sévère asymptomatique : {p.tension_systolique} mmHg"))
            elif p.tension_systolique >= 200:
                if p.confusion_soudaine or (p.douleur_intensite and p.douleur_intensite > 3):
                    mods.append((3, f"HTA modérée symptomatique : {p.tension_systolique} mmHg"))
                else:
                    mods.append((4, f"HTA modérée asymptomatique : {p.tension_systolique} mmHg"))

        # --- Température (Table 4 du guide) ---
        if p.temperature is not None and p.temperature >= 38.5:
            if p.immunosupprime:
                mods.append((2, f"Fièvre {p.temperature}°C chez immunosupprimé"))
            elif p.sirs_criteres >= 3 or p.choc_hemodynamique or p.confusion_soudaine:
                mods.append((2, f"Fièvre {p.temperature}°C — aspect septique (SIRS ≥3)"))
            elif p.sirs_criteres >= 2:
                mods.append((3, f"Fièvre {p.temperature}°C — patient paraît mal en point"))
            else:
                mods.append((4, f"Fièvre {p.temperature}°C — patient stable"))

        # --- Glycémie (Table 8 du guide) ---
        if p.glycemie is not None:
            if p.glycemie < 3.0:
                if p.confusion_soudaine or p.convulsion_active:
                    mods.append((2, f"Hypoglycémie {p.glycemie} mmol/L avec symptômes"))
                else:
                    mods.append((3, f"Hypoglycémie {p.glycemie} mmol/L asymptomatique"))
            elif p.glycemie >= 18.0:
                if p.detresse_respiratoire_moderee or p.choc_hemodynamique:
                    mods.append((2, f"Hyperglycémie {p.glycemie} mmol/L avec symptômes"))
                else:
                    mods.append((3, f"Hyperglycémie {p.glycemie} mmol/L asymptomatique"))

        # --- GCS (Table 3 du guide) ---
        if p.gcs is not None:
            if 10 <= p.gcs <= 13:
                mods.append((2, f"Conscience altérée : GCS {p.gcs}"))
            # GCS ≤9 déjà géré dans Quick Look

        # --- Confusion soudaine ---
        if p.confusion_soudaine and (p.gcs is None or p.gcs > 13):
            mods.append((2, "Confusion soudaine — altération neurologique"))

        # --- Détresse respiratoire modérée (flags cliniques) ---
        if p.detresse_respiratoire_moderee:
            mods.append((2, "Détresse respiratoire modérée"))

        # --- Saignement abondant ---
        if p.saignement_abondant:
            mods.append((2, "Hémorragie active abondante"))

        # --- Douleur (Table 5 du guide) ---
        # Logique : sévérité + localisation + durée
        if p.douleur_intensite is not None:
            niveau_douleur = self._evaluer_douleur(p)
            if niveau_douleur:
                mods.append(niveau_douleur)

        return mods

    def _evaluer_douleur(self, p: PatientInput):
        """Applique la Table 5 du guide CTAS (Pain modifier)."""
        if p.douleur_intensite is None:
            return None

        centrale = p.douleur_centrale if p.douleur_centrale is not None else True
        aigue = p.douleur_aigue if p.douleur_aigue is not None else True
        i = p.douleur_intensite
        loc = "centrale" if centrale else "périphérique"
        duree = "aiguë" if aigue else "chronique"

        if i >= 8:       # sévère
            if centrale and aigue:
                return (2, f"Douleur sévère {i}/10 {loc} {duree}")
            elif centrale and not aigue:
                return (3, f"Douleur sévère {i}/10 {loc} {duree}")
            elif not centrale and aigue:
                return (3, f"Douleur sévère {i}/10 {loc} {duree}")
            else:
                return (4, f"Douleur sévère {i}/10 {loc} {duree}")
        elif i >= 4:     # modérée
            if centrale and aigue:
                return (3, f"Douleur modérée {i}/10 {loc} {duree}")
            elif centrale and not aigue:
                return (4, f"Douleur modérée {i}/10 {loc} {duree}")
            elif not centrale and aigue:
                return (4, f"Douleur modérée {i}/10 {loc} {duree}")
            else:
                return (5, f"Douleur modérée {i}/10 {loc} {duree}")
        else:            # légère
            if centrale and aigue:
                return (4, f"Douleur légère {i}/10 {loc} {duree}")
            else:
                return (5, f"Douleur légère {i}/10 {loc} {duree}")

    # ──────────────────────────────────────────
    #  ÉTAPE 4 : MODIFICATEURS 2E ORDRE
    # ──────────────────────────────────────────
    def _modificateurs_second_ordre(self, p: PatientInput) -> list:
        """
        Modificateurs spécifiques à la présentation (feuille 03).
        Ne peuvent PAS downgrader un niveau déjà établi par les étapes précédentes.
        """
        mods = []
        pid = p.presentation_id

        # Douleur abdominale aiguë (P3_03)
        if pid == "P3_03":
            if p.vomit_sang:
                mods.append((2, "Hématémèse — hémorragie digestive suspectée"))
            if p.enceinte:
                mods.append((2, "Douleur abdominale chez femme enceinte"))

        # Douleur thoracique atypique (P3_01)
        if pid == "P3_01":
            if p.douleur_irradie_bras_machoire:
                mods.append((2, "Irradiation bras/mâchoire — SCA suspecté"))

        # AVC suspect (P2_02) — délai depuis symptômes
        # Le délai est le seul cas où un modificateur 2e ordre peut descendre
        # une présentation CTAS 2 à CTAS 3 (règle explicite du guide)
        if pid == "P2_02":
            if p.avc_delai_heures is not None and p.avc_delai_heures >= 3.5:
                mods.append((-1, f"AVC hors fenêtre thérapeutique ({p.avc_delai_heures}h ≥ 3.5h) → CTAS 3"))
            elif p.avc_delai_heures is not None and p.avc_delai_heures < 3.5:
                mods.append((2, f"AVC dans la fenêtre thérapeutique ({p.avc_delai_heures}h < 3.5h)"))

        # Entorse (P4_03)
        if pid == "P4_03":
            if p.deformation_osseuse:
                mods.append((3, "Déformation osseuse — fracture déplacée suspectée"))

        # Infection urinaire simple (P4_02)
        if pid == "P4_02":
            if p.fievre_avec_itu:
                mods.append((3, "Fièvre avec ITU — pyélonéphrite suspectée"))

        return mods

    # ──────────────────────────────────────────
    #  CONSTRUCTION DU RÉSULTAT
    # ──────────────────────────────────────────
    def _construire_resultat(
        self, niveau, raison_principale, raisons_secondaires, etape
    ) -> CTASResult:
        return CTASResult(
            niveau=niveau,
            label=LABELS_CTAS[niveau],
            raison_principale=raison_principale,
            raisons_secondaires=list(dict.fromkeys(raisons_secondaires)),  # dédoublonner
            etape_declenchement=etape,
            recommandation=RECOMMANDATIONS[niveau],
        )


# ─────────────────────────────────────────────
#  TESTS DE VALIDATION
# ─────────────────────────────────────────────

def run_tests():
    engine = CTASEngine()
    tests = [
        # (description, PatientInput, niveau_attendu)
        (
            "Arrêt cardiaque",
            PatientInput(presentation_id="P1_01", inconscient=True, arret_respiratoire=True),
            1
        ),
        (
            "AVC < 3.5h",
            PatientInput(presentation_id="P2_02", avc_delai_heures=2.0,
                         gcs=13, tension_systolique=225),
            2
        ),
        (
            "AVC > 3.5h → downgrade CTAS 3",
            PatientInput(presentation_id="P2_02", avc_delai_heures=5.0, gcs=14),
            3
        ),
        (
            "Douleur thoracique atypique + irradiation bras → upgrade CTAS 2",
            PatientInput(presentation_id="P3_01", douleur_irradie_bras_machoire=True,
                         douleur_intensite=7, douleur_centrale=True, douleur_aigue=True),
            2
        ),
        (
            "SpO2 88% → CTAS 1 (Quick Look)",
            PatientInput(presentation_id="P4_06", spo2=88.0),
            1
        ),
        (
            "SpO2 91% → upgrade CTAS 2",
            PatientInput(presentation_id="P4_06", spo2=91.0),
            2
        ),
        (
            "Fièvre 39°C immunosupprimé → CTAS 2",
            PatientInput(presentation_id="P3_06", temperature=39.0, immunosupprime=True),
            2
        ),
        (
            "Hypoglycémie 2.5 mmol/L avec confusion → CTAS 2",
            PatientInput(glycemie=2.5, confusion_soudaine=True),
            2
        ),
        (
            "Entorse simple → CTAS 4",
            PatientInput(presentation_id="P4_03"),
            4
        ),
        (
            "Entorse + déformation osseuse → upgrade CTAS 3",
            PatientInput(presentation_id="P4_03", deformation_osseuse=True),
            3
        ),
        (
            "Rhume simple → CTAS 5",
            PatientInput(presentation_id="P5_04"),
            5
        ),
        (
            "Douleur abdominale + vomit sang → CTAS 2",
            PatientInput(presentation_id="P3_03", vomit_sang=True),
            2
        ),
        (
            "HTA 230 mmHg symptomatique → CTAS 2",
            PatientInput(tension_systolique=230, douleur_intensite=6,
                         douleur_centrale=True, douleur_aigue=True),
            2
        ),
        (
            "Douleur sévère 9/10 centrale aiguë sans présentation → CTAS 2",
            PatientInput(douleur_intensite=9, douleur_centrale=True, douleur_aigue=True),
            2
        ),
    ]

    print("=" * 65)
    print("  TESTS DE VALIDATION DU MOTEUR CTAS")
    print("=" * 65)
    passed = 0
    failed = 0
    for desc, patient, attendu in tests:
        result = engine.evaluer(patient)
        status = "✅ PASS" if result.niveau == attendu else f"❌ FAIL (obtenu CTAS {result.niveau})"
        if result.niveau == attendu:
            passed += 1
        else:
            failed += 1
        print(f"\n[{status}] {desc}")
        print(f"         Attendu: CTAS {attendu} | Obtenu: CTAS {result.niveau}")
        print(f"         Étape  : {result.etape_declenchement}")
        print(f"         Raison : {result.raison_principale}")
        if result.raisons_secondaires:
            for r in result.raisons_secondaires:
                print(f"                  + {r}")

    print("\n" + "=" * 65)
    print(f"  Résultats : {passed}/{len(tests)} tests passés", 
          "🎉" if failed == 0 else f"({failed} échec(s))")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
