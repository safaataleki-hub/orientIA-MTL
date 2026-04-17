"""
Moteur de Recommandation — TriageAI MTL
========================================
Algorithme de scoring pondéré à 3 étapes :
    Étape 1 — Filtre CTAS       : éligibilité stricte par niveau de triage
    Étape 2 — Filtre capacité   : éligibilité par catégorie de plainte
    Étape 3 — Scoring pondéré   : attente (0.40) + capacité (0.35) + distance (0.25)

Retourne le top 3 des meilleures options avec justification détaillée.

Note sur les poids :
    Les poids ont été définis de façon heuristique pour le prototype en
    reflétant les priorités cliniques de l'équipe (attente > capacité > distance).
    Une version de production devrait les valider via la méthode AHP
    (Analytic Hierarchy Process, Saaty 1980) avec des professionnels de santé.
"""

from dataclasses import dataclass, field
from typing import Optional
from points_de_service import (
    PointDeService, POINTS_DE_SERVICE, PRESENTATION_CATEGORIE,
    distance_km, TYPE_SERVICE
)


# ─────────────────────────────────────────────
#  RÈGLES D'ÉLIGIBILITÉ CTAS
#  Quels types de service sont autorisés
#  selon le niveau CTAS détecté
# ─────────────────────────────────────────────

ELIGIBILITE_CTAS = {
    1: ["urgence"],                                          # Réanimation → urgence uniquement
    2: ["urgence"],                                          # Emergent → urgence uniquement
    3: ["urgence", "clinique", "specialise"],                # Urgent → urgence ou clinique
    4: ["clinique", "gmf", "specialise", "tel"],             # Moins urgent → pas d'urgence
    5: ["pharmacie", "gmf", "clinique", "tel"],              # Non urgent → soins de base
}

# Seuil minimal de capacité pour qu'un service soit considéré
SEUIL_CAPACITE_MIN = 0.3

# Poids du scoring (documentés et assumés — voir note ci-dessus)
POIDS = {
    "attente":   0.40,
    "capacite":  0.35,
    "distance":  0.25,
}


# ─────────────────────────────────────────────
#  STRUCTURE DU RÉSULTAT
# ─────────────────────────────────────────────

@dataclass
class RecommandationItem:
    """Une option de service recommandée avec son score et sa justification."""
    rang: int
    point: PointDeService
    score_final: float
    score_attente: float
    score_capacite: float
    score_distance: float
    distance_km: float
    attente_min: int
    temps_sejour_estime: int         # attente + consultation estimée
    justification: str
    avertissement: Optional[str] = None


@dataclass
class ResultatRecommandation:
    """Résultat complet du moteur de recommandation."""
    ctas_niveau: int
    categorie_plainte: str
    localisation_user: tuple         # (lat, lon)
    recommandations: list            # liste de RecommandationItem
    nb_services_evalues: int
    nb_services_filtres_ctas: int
    nb_services_filtres_capacite: int
    message_special: Optional[str] = None


# ─────────────────────────────────────────────
#  MOTEUR DE RECOMMANDATION
# ─────────────────────────────────────────────

class RecommandationEngine:
    """
    Moteur de recommandation TriageAI MTL.
    Combine filtre CTAS + filtre capacité + scoring pondéré.
    """

    def recommander(
        self,
        ctas_niveau: int,
        presentation_id: Optional[str],
        lat_user: float,
        lon_user: float,
        top_n: int = 3,
    ) -> ResultatRecommandation:
        """
        Point d'entrée principal.

        Args:
            ctas_niveau     : niveau CTAS détecté (1-5)
            presentation_id : ID de la présentation (ex. 'P3_03')
            lat_user        : latitude de l'utilisateur
            lon_user        : longitude de l'utilisateur
            top_n           : nombre de recommandations à retourner

        Returns:
            ResultatRecommandation avec top_n options classées
        """

        # Cas CTAS 1 — arrêt immédiat, pas de recommandation normale
        if ctas_niveau == 1:
            return self._resultat_urgence_critique()

        categorie = PRESENTATION_CATEGORIE.get(presentation_id, "Administratif") \
                    if presentation_id else "Administratif"

        services = POINTS_DE_SERVICE.copy()
        nb_total = len(services)

        # ══════════════════════════════════════════
        # ÉTAPE 1 — FILTRE CTAS
        # ══════════════════════════════════════════
        types_eligibles = ELIGIBILITE_CTAS.get(ctas_niveau, ["clinique", "gmf"])
        services_post_ctas = [s for s in services if s.type in types_eligibles]
        nb_post_ctas = len(services_post_ctas)

        # ══════════════════════════════════════════
        # ÉTAPE 2 — FILTRE CAPACITÉ + OUVERTURE
        # ══════════════════════════════════════════
        services_eligibles = []
        for s in services_post_ctas:
            cap = s.capacite_pour(categorie)
            if cap >= SEUIL_CAPACITE_MIN and s.est_ouvert():
                services_eligibles.append(s)

        nb_post_capacite = len(services_eligibles)

        # Si moins de 3 services ouverts → compléter avec les fermés
        if len(services_eligibles) < 3:
            fermes = [
                s for s in services_post_ctas
                if s.capacite_pour(categorie) >= SEUIL_CAPACITE_MIN
                and not s.est_ouvert()
                and s not in services_eligibles
            ]
            services_eligibles += fermes

        # ══════════════════════════════════════════
        # ÉTAPE 3 — CALCUL DES SCORES
        # ══════════════════════════════════════════
        scores_bruts = []
        for s in services_eligibles:
            dist = distance_km(lat_user, lon_user, s.lat, s.lon)
            attente = s.attente_actuelle()
            cap = s.capacite_pour(categorie)
            scores_bruts.append({
                "service":  s,
                "distance": dist,
                "attente":  attente,
                "capacite": cap,
            })

        if not scores_bruts:
            return self._resultat_vide(ctas_niveau, categorie)

        # Normalisation 0-1 pour chaque critère
        max_dist    = max(x["distance"] for x in scores_bruts) or 1
        max_attente = max(x["attente"]  for x in scores_bruts) or 1

        recommandations = []
        for item in scores_bruts:
            # Plus la distance est petite → score élevé
            s_dist    = 1.0 - (item["distance"] / max_dist)
            # Plus l'attente est courte → score élevé
            s_attente = 1.0 - (item["attente"]  / max_attente)
            # Capacité déjà entre 0 et 1
            s_cap     = item["capacite"]

            score_final = (
                POIDS["attente"]  * s_attente +
                POIDS["capacite"] * s_cap     +
                POIDS["distance"] * s_dist
            )

            # Temps de séjour estimé = attente + durée consultation selon type
            durees_consultation = {
                "urgence": 120, "clinique": 45, "gmf": 30,
                "pharmacie": 10, "specialise": 60, "tel": 20,
            }
            duree_consult = durees_consultation.get(item["service"].type, 45)
            temps_sejour = item["attente"] + duree_consult

            justification = self._generer_justification(
                item["service"], item["attente"], item["distance"],
                s_cap, ctas_niveau, categorie
            )

            recommandations.append(RecommandationItem(
                rang=0,
                point=item["service"],
                score_final=round(score_final, 3),
                score_attente=round(s_attente, 3),
                score_capacite=round(s_cap, 3),
                score_distance=round(s_dist, 3),
                distance_km=round(item["distance"], 1),
                attente_min=item["attente"],
                temps_sejour_estime=temps_sejour,
                justification=justification,
            ))

        # Trier par score décroissant
        recommandations.sort(key=lambda x: x.score_final, reverse=True)

        # Assigner les rangs et avertissements
        for i, r in enumerate(recommandations[:top_n]):
            r.rang = i + 1
            r.avertissement = self._generer_avertissement(r.point, ctas_niveau)

        return ResultatRecommandation(
            ctas_niveau=ctas_niveau,
            categorie_plainte=categorie,
            localisation_user=(lat_user, lon_user),
            recommandations=recommandations[:top_n],
            nb_services_evalues=nb_total,
            nb_services_filtres_ctas=nb_total - nb_post_ctas,
            nb_services_filtres_capacite=nb_post_ctas - nb_post_capacite,
        )

    # ──────────────────────────────────────────
    #  GÉNÉRATEURS DE TEXTE
    # ──────────────────────────────────────────

    def _generer_justification(
        self, service: PointDeService, attente: int,
        distance: float, capacite: float,
        ctas: int, categorie: str
    ) -> str:
        """Génère une justification lisible pour l'utilisateur."""
        parties = []

        # Attente
        if attente <= 15:
            parties.append(f"attente très courte ({attente} min)")
        elif attente <= 45:
            parties.append(f"attente raisonnable ({attente} min)")
        else:
            parties.append(f"attente de {attente} min")

        # Distance
        if distance < 2:
            parties.append(f"très proche ({distance:.1f} km)")
        elif distance < 5:
            parties.append(f"proche ({distance:.1f} km)")
        else:
            parties.append(f"à {distance:.1f} km")

        # Capacité
        if capacite >= 0.8:
            parties.append(f"équipé pour traiter {categorie.lower()}")
        elif capacite >= 0.5:
            parties.append(f"peut prendre en charge votre situation")
        else:
            parties.append(f"peut vous orienter")

        return " · ".join(parties)

    def _generer_avertissement(
        self, service: PointDeService, ctas: int
    ) -> Optional[str]:
        """Génère un avertissement si nécessaire."""
        if not service.est_ouvert():
            return "⚠️ Vérifiez les heures d'ouverture avant de vous déplacer."
        if ctas == 3 and service.type == "clinique":
            return "⚠️ Si votre état se détériore en route, dirigez-vous vers l'urgence."
        if service.type == "tel":
            return "ℹ️ Le 811 peut vous orienter vers le bon service sans déplacement."
        return None

    def _resultat_urgence_critique(self) -> ResultatRecommandation:
        """Résultat spécial pour CTAS 1 — appel 911 uniquement."""
        urgences = [s for s in POINTS_DE_SERVICE if s.type == "urgence"]
        items = []
        for i, u in enumerate(urgences[:3]):
            items.append(RecommandationItem(
                rang=i+1,
                point=u,
                score_final=1.0,
                score_attente=1.0,
                score_capacite=1.0,
                score_distance=1.0,
                distance_km=0.0,
                attente_min=0,
                temps_sejour_estime=0,
                justification="Urgence vitale — accueil immédiat garanti",
                avertissement="🚨 Appelez le 911 — ne vous déplacez pas seul.",
            ))
        return ResultatRecommandation(
            ctas_niveau=1,
            categorie_plainte="Critique",
            localisation_user=(0, 0),
            recommandations=items,
            nb_services_evalues=len(POINTS_DE_SERVICE),
            nb_services_filtres_ctas=len(POINTS_DE_SERVICE) - len(urgences),
            nb_services_filtres_capacite=0,
            message_special="🚨 CTAS 1 — Situation critique. Appelez le 911 immédiatement.",
        )

    def _resultat_vide(self, ctas: int, categorie: str) -> ResultatRecommandation:
        """Résultat si aucun service éligible trouvé."""
        return ResultatRecommandation(
            ctas_niveau=ctas,
            categorie_plainte=categorie,
            localisation_user=(0, 0),
            recommandations=[],
            nb_services_evalues=len(POINTS_DE_SERVICE),
            nb_services_filtres_ctas=0,
            nb_services_filtres_capacite=0,
            message_special="Aucun service disponible trouvé. Contactez le 811.",
        )


# ─────────────────────────────────────────────
#  TESTS DE VALIDATION
# ─────────────────────────────────────────────

def run_tests():
    engine = RecommandationEngine()

    # Localisation simulée : centre de Montréal (métro Berri)
    LAT_USER = 45.5196
    LON_USER = -73.5674

    tests = [
        {
            "desc": "CTAS 2 — AVC suspect (Neurologique)",
            "ctas": 2, "pid": "P2_02",
            "types_attendus": ["urgence"],
        },
        {
            "desc": "CTAS 3 — Douleur abdominale (GI)",
            "ctas": 3, "pid": "P3_03",
            "types_attendus": ["urgence", "clinique", "specialise"],
        },
        {
            "desc": "CTAS 4 — Entorse (MSK)",
            "ctas": 4, "pid": "P4_03",
            "types_attendus": ["clinique", "gmf", "specialise", "tel"],
            "types_exclus": ["urgence"],
        },
        {
            "desc": "CTAS 5 — Rhume (ORL)",
            "ctas": 5, "pid": "P5_04",
            "types_attendus": ["pharmacie", "gmf", "clinique", "tel"],
            "types_exclus": ["urgence"],
        },
        {
            "desc": "CTAS 1 — Arrêt cardiaque",
            "ctas": 1, "pid": "P1_01",
            "types_attendus": ["urgence"],
        },
    ]

    print("=" * 65)
    print("  TESTS DU MOTEUR DE RECOMMANDATION")
    print("=" * 65)

    for t in tests:
        result = engine.recommander(t["ctas"], t["pid"], LAT_USER, LON_USER)
        print(f"\n[CTAS {t['ctas']}] {t['desc']}")
        print(f"  Catégorie : {result.categorie_plainte}")
        print(f"  Filtrés CTAS : {result.nb_services_filtres_ctas} | "
              f"Filtrés capacité : {result.nb_services_filtres_capacite}")

        if result.message_special:
            print(f"  ⚡ {result.message_special}")

        for r in result.recommandations:
            print(f"  #{r.rang} [{r.point.type:10}] {r.point.nom:<40} "
                  f"Score: {r.score_final:.3f} | "
                  f"Attente: {r.attente_min:3d}min | "
                  f"Dist: {r.distance_km:.1f}km")
            print(f"       → {r.justification}")
            if r.avertissement:
                print(f"       {r.avertissement}")

        # Vérification des types exclus
        if "types_exclus" in t:
            types_retournes = [r.point.type for r in result.recommandations]
            for type_exclu in t["types_exclus"]:
                if type_exclu in types_retournes:
                    print(f"  ❌ ERREUR : {type_exclu} ne devrait pas apparaître pour CTAS {t['ctas']}")
                else:
                    print(f"  ✅ {type_exclu} correctement exclu")

    print("\n" + "=" * 65)
    print("  Tests terminés")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
