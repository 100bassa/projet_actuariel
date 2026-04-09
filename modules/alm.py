# ══════════════════════════════════════════════════════════════
#  modules/alm.py — ASSET-LIABILITY MANAGEMENT (ALM)
#
#  RÔLE DE CE MODULE
#  ─────────────────
#  Ce module implémente les outils standard de gestion actif-passif
#  pour un assureur vie. L'ALM cherche à s'assurer que les actifs
#  financiers de l'assureur peuvent toujours faire face à ses
#  engagements envers les assurés, quelle que soit l'évolution
#  des taux d'intérêt.
#
#  POURQUOI L'ALM EST-IL IMPORTANT ?
#  ───────────────────────────────────
#  Les assureurs vie collectent des primes et s'engagent à payer des
#  capitaux décès ou des rentes dans 10, 20 ou 30 ans.
#  Pour faire face à ces engagements futurs (= passif), ils investissent
#  principalement dans des obligations (= actif).
#
#  Le problème : si les taux d'intérêt bougent...
#    → La valeur des obligations (actif) change
#    → La valeur actualisée des engagements (passif = BEL) change aussi
#    → Mais pas forcément dans les mêmes proportions !
#    → Le "surplus" (Actif - BEL) peut être érodé.
#
#  L'ALM mesure et gère ce risque de taux structurel via le Duration Gap.
#
#  CE FICHIER CONTIENT 4 CLASSES
#  ──────────────────────────────
#   1. Obligation        → valorise une obligation et calcule sa sensibilité
#   2. PortefeuilleActif → agrège obligations + actions + immobilier
#   3. ProjectionPassif  → projette les flux vie (décès, rachats, frais)
#   4. AnalyseALM        → combine actif et passif pour les indicateurs ALM
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
# typing = annotations de types (pour la documentation et les IDE)
# List = liste typée, Optional = peut être None, Dict = dictionnaire typé
from typing import List, Optional, Dict


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE 1 — OBLIGATION
# ─────────────────────────────────────────────────────────────────────────────
# Rôle : représente une obligation à taux fixe et calcule toutes ses
# caractéristiques financières (prix, duration, convexité, sensibilité).
#
# Pourquoi ? Les obligations sont la principale classe d'actifs des assureurs vie.
# Comprendre leur sensibilité aux taux est fondamental pour l'ALM.
# ─────────────────────────────────────────────────────────────────────────────

class Obligation:
    """
    Obligation à taux fixe cotée au rendement actuariel.

    Calcule le prix théorique, la duration de Macaulay, la duration modifiée
    et la convexité à partir des flux futurs (coupons + remboursement du nominal).

    Parameters
    ----------
    nominal : float
        Valeur nominale (en euros).
    coupon : float
        Taux de coupon annuel (ex. 0.03 pour 3 %).
    maturite : int
        Durée résiduelle en années entières.
    taux_rendement : float
        Taux de rendement actuariel de marché (ex. 0.04 pour 4 %).
    notation : str
        Notation crédit (AAA, AA, A, BBB, …).
    type_ : str
        Catégorie d'obligation (« souveraine », « entreprise », etc.).
    """

    def __init__(
        self,
        nominal: float,
        coupon: float,
        maturite: int,
        taux_rendement: float,
        notation: str = "BBB",
        type_: str = "entreprise",
    ):
        # On stocke tous les paramètres comme attributs de l'objet
        # pour les réutiliser dans les propriétés calculées (@property)
        self.nominal        = nominal
        self.coupon         = coupon
        self.maturite       = maturite
        self.taux_rendement = taux_rendement
        self.notation       = notation
        self.type_          = type_

    # ── Flux de trésorerie ────────────────────────────────────────────────────

    @property
    def flux(self) -> np.ndarray:
        """
        Tableau des flux annuels de l'obligation.

        Rôle : génère les flux coupons + remboursement final.
        Pourquoi un @property ? → s'utilise comme un attribut (obj.flux)
        sans parenthèses, mais le calcul est fait à la demande.

        Structure :
          années 1 à T-1 : flux = nominal × coupon   (coupon seul)
          année T         : flux = nominal × coupon + nominal  (coupon + remboursement)

        np.full(n, val) = crée un tableau de n éléments tous égaux à val
        cf[-1] += nominal = modifie le dernier élément pour ajouter le remboursement
        """
        cf = np.full(self.maturite, self.nominal * self.coupon)
        cf[-1] += self.nominal   # remboursement du principal à l'échéance
        return cf

    # ── Prix de marché ────────────────────────────────────────────────────────

    @property
    def prix(self) -> float:
        """
        Valeur actuelle (prix de marché) de l'obligation.

        Rôle : calcule le prix théorique en actualisant tous les flux
        au taux de rendement de marché.

        Formule : P = Σ(t=1 à T) flux_t / (1 + r)^t

        Pourquoi le prix peut-il différer du nominal ?
          - Si r > coupon → prix < nominal (décote : l'obligation vaut moins)
          - Si r < coupon → prix > nominal (surcote : l'obligation vaut plus)
          - Si r = coupon → prix = nominal (au pair)

        np.arange(1, T+1) = [1, 2, 3, ..., T] (les années)
        np.sum() = somme de tous les éléments du tableau
        """
        t = np.arange(1, self.maturite + 1)   # tableau des années
        return float(np.sum(self.flux / (1 + self.taux_rendement) ** t))

    # ── Duration de Macaulay ──────────────────────────────────────────────────

    @property
    def duration_macaulay(self) -> float:
        """
        Duration de Macaulay (en années).

        Rôle : mesure la « durée de vie moyenne » pondérée des flux,
        c'est-à-dire le centre de gravité temporel des paiements.

        Formule : D = Σ(t × VA(flux_t)) / P
                    = Σ(t × flux_t / (1+r)^t) / P

        Interprétation :
          D = 5 ans → en moyenne, les flux sont reçus dans 5 ans.
          Une obligation zéro-coupon a une duration = sa maturité.
          Plus le coupon est élevé, plus la duration est courte.

        np.dot(t, va_flux) = produit scalaire = Σ(t_i × va_flux_i)
        """
        t       = np.arange(1, self.maturite + 1)
        va_flux = self.flux / (1 + self.taux_rendement) ** t  # flux actualisés
        return float(np.dot(t, va_flux) / self.prix)

    @property
    def duration_modifiee(self) -> float:
        """
        Duration modifiée (sans unité).

        Rôle : mesure la sensibilité du prix aux variations de taux.
        C'est la duration de Macaulay ajustée d'un facteur (1 + r).

        Formule : MD = D_Macaulay / (1 + r)

        Interprétation :
          ΔP/P ≈ −MD × Δr
          Si MD = 5 et les taux montent de 1% → le prix baisse d'environ 5%.

        C'est l'outil fondamental de la couverture de taux (hedging).
        """
        return self.duration_macaulay / (1 + self.taux_rendement)

    @property
    def convexite(self) -> float:
        """
        Convexité de l'obligation.

        Rôle : correction du second ordre dans l'approximation du prix.
        La duration seule sous-estime le prix quand les taux baissent
        et le sur-estime quand les taux montent.
        La convexité corrige cette asymétrie.

        Formule : C = Σ(t×(t+1) × VA(flux_t)) / (P × (1+r)²)

        Interprétation :
          ΔP/P ≈ −MD × Δr + ½ × C × (Δr)²
          Une convexité élevée est favorable : le prix remonte plus
          qu'il ne baisse pour un même choc de taux.

        np.dot(t*(t+1), va_flux) = Σ(t_i × (t_i+1) × va_flux_i)
        """
        t       = np.arange(1, self.maturite + 1)
        va_flux = self.flux / (1 + self.taux_rendement) ** t
        return float(
            np.dot(t * (t + 1), va_flux)
            / (self.prix * (1 + self.taux_rendement) ** 2)
        )

    # ── Sensibilité aux chocs de taux ────────────────────────────────────────

    def variation_prix(self, delta_taux: float) -> float:
        """
        Variation approchée du prix pour un choc parallèle de taux.

        Rôle : estime l'impact d'une hausse ou baisse de taux sur la valeur
        de l'obligation (formule de Taylor au second ordre).

        Formule : ΔP ≈ −MD × P × Δy + ½ × C × P × (Δy)²

        Paramètre :
          delta_taux : choc de taux en fraction décimale
                       (ex: 0.01 = +100 bps = +1%)

        Exemple :
          Si MD=5, C=30, P=100, Δy=+0.01 (taux montent de 1%) :
          ΔP ≈ −5 × 100 × 0.01 + ½ × 30 × 100 × 0.0001
             ≈ −5 + 0.15 = −4.85 €
        """
        return (
            -self.duration_modifiee * self.prix * delta_taux
            + 0.5 * self.convexite * self.prix * delta_taux ** 2
        )

    def to_dict(self) -> dict:
        """
        Sérialise l'obligation en dictionnaire.

        Rôle : permet de construire facilement un DataFrame pandas
        avec une ligne par obligation (utilisé dans tableau_obligations()).
        """
        return {
            "Notation":            self.notation,
            "Type":                self.type_,
            "Nominal (€)":         self.nominal,
            "Coupon (%)":          round(self.coupon * 100, 2),
            "Maturité (ans)":      self.maturite,
            "Taux rend. (%)":      round(self.taux_rendement * 100, 2),
            "Prix (€)":            round(self.prix, 2),
            "Duration Mac. (ans)": round(self.duration_macaulay, 2),
            "Duration mod.":       round(self.duration_modifiee, 2),
            "Convexité":           round(self.convexite, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE 2 — PORTEFEUILLE D'ACTIFS
# ─────────────────────────────────────────────────────────────────────────────
# Rôle : agrège trois classes d'actifs (obligations, actions, immobilier)
# et calcule les indicateurs consolidés du portefeuille (valeur, duration,
# convexité, rendement, sensibilité aux taux).
#
# Pourquoi trois classes ?
#   - Obligations : principale classe, sensible aux taux, gestion ALM fine
#   - Actions     : rendement supérieur mais duration ≈ 0 en ALM classique
#   - Immobilier  : actif réel, duration proxy (baux long terme)
# ─────────────────────────────────────────────────────────────────────────────

class PortefeuilleActif:
    """
    Portefeuille d'actifs d'un assureur vie composé de trois classes :

    * Obligations (liste d'objets ``Obligation``)
    * Actions (valeur forfaitaire + rendement attendu)
    * Immobilier (valeur forfaitaire + rendement attendu + duration proxy)

    Parameters
    ----------
    obligations : list[Obligation], optional
        Portefeuille obligataire.
    valeur_actions : float
        Valeur de marché du portefeuille actions.
    rendement_actions : float
        Rendement espéré des actions (ex. 0.07 pour 7 %).
    duration_actions : float
        Duration proxy des actions (souvent 0 en ALM classique).
    valeur_immobilier : float
        Valeur de marché du portefeuille immobilier.
    rendement_immobilier : float
        Rendement espéré de l'immobilier.
    duration_immobilier : float
        Duration proxy de l'immobilier (typiquement 5–10 ans).
    """

    def __init__(
        self,
        obligations: Optional[List[Obligation]] = None,
        valeur_actions: float = 0.0,
        rendement_actions: float = 0.07,
        duration_actions: float = 0.0,       # actions peu sensibles aux taux
        valeur_immobilier: float = 0.0,
        rendement_immobilier: float = 0.04,
        duration_immobilier: float = 7.0,    # baux immobiliers ≈ 7 ans de duration
    ):
        # `or []` : si None est passé, on utilise une liste vide
        self.obligations          = obligations or []
        self.valeur_actions       = valeur_actions
        self.rendement_actions    = rendement_actions
        self.duration_actions     = duration_actions
        self.valeur_immobilier    = valeur_immobilier
        self.rendement_immobilier = rendement_immobilier
        self.duration_immobilier  = duration_immobilier

    # ── Valeurs ───────────────────────────────────────────────────────────────

    @property
    def valeur_obligations(self) -> float:
        """
        Valeur totale du portefeuille obligataire (prix de marché).
        sum(o.prix for o in ...) = générateur : somme des prix de chaque obligation.
        """
        return sum(o.prix for o in self.obligations)

    @property
    def valeur_totale(self) -> float:
        """Valeur de marché totale de tous les actifs."""
        return self.valeur_obligations + self.valeur_actions + self.valeur_immobilier

    # ── Allocation ────────────────────────────────────────────────────────────

    @property
    def allocation(self) -> Dict[str, float]:
        """
        Poids de chaque classe d'actifs dans le portefeuille (somme = 1).

        Rôle : donne la répartition stratégique actif (pour le graphique camembert).
        Pourquoi surveiller l'allocation ? Une forte part obligataire signifie
        une forte sensibilité aux taux → risk ALM élevé.
        """
        vt = self.valeur_totale
        if vt == 0:  # protection contre la division par zéro
            return {"Obligations": 0.0, "Actions": 0.0, "Immobilier": 0.0}
        return {
            "Obligations": self.valeur_obligations / vt,
            "Actions":     self.valeur_actions     / vt,
            "Immobilier":  self.valeur_immobilier  / vt,
        }

    # ── Duration & Convexité agrégées ─────────────────────────────────────────

    @property
    def duration(self) -> float:
        """
        Duration de Macaulay pondérée par la valeur de marché du portefeuille.

        Rôle : résume en un seul chiffre la sensibilité globale du portefeuille
        aux mouvements de taux. C'est l'indicateur clé de l'ALM côté actif.

        Formule : D_A = Σ (w_i × D_i) où w_i = poids de chaque actif

        Calcul par classe :
          d_obl = duration moyenne pondérée des obligations
          d_act = duration_actions × poids_actions  (souvent ≈ 0)
          d_imm = duration_immobilier × poids_immobilier
        """
        vt = self.valeur_totale
        if vt == 0:
            return 0.0
        # Duration obligations : moyenne pondérée par les prix de chaque obligation
        d_obl = (
            sum(o.duration_macaulay * o.prix for o in self.obligations) / vt
            if self.obligations
            else 0.0
        )
        # Duration actions et immobilier : duration proxy × poids
        d_act = self.duration_actions   * self.valeur_actions   / vt
        d_imm = self.duration_immobilier * self.valeur_immobilier / vt
        return d_obl + d_act + d_imm

    @property
    def convexite(self) -> float:
        """
        Convexité agrégée pondérée (obligations uniquement).

        Rôle : mesure l'asymétrie de la réponse du portefeuille aux chocs de taux.
        Utilisée dans la vérification de l'immunisation de Redington (condition 3).
        """
        vt = self.valeur_totale
        if vt == 0 or not self.obligations:
            return 0.0
        return sum(o.convexite * o.prix for o in self.obligations) / vt

    @property
    def rendement_moyen(self) -> float:
        """
        Taux de rendement agrégé pondéré par la valeur de marché.

        Rôle : donne le rendement global du portefeuille d'actifs.
        Utilisé dans la projection du bilan (croissance de l'actif par an).
        """
        vt = self.valeur_totale
        if vt == 0:
            return 0.0
        r_obl = (
            sum(o.taux_rendement * o.prix for o in self.obligations) / vt
            if self.obligations
            else 0.0
        )
        r_act = self.rendement_actions    * self.valeur_actions    / vt
        r_imm = self.rendement_immobilier * self.valeur_immobilier / vt
        return r_obl + r_act + r_imm

    # ── Sensibilité aux chocs de taux ─────────────────────────────────────────

    def impact_choc_taux(self, delta_taux: float) -> Dict[str, float]:
        """
        Impact d'un choc de taux parallèle sur la valeur du portefeuille.

        Rôle : calcule de combien la valeur de marché des actifs change
        si les taux d'intérêt bougent de delta_taux (ex: 0.01 = +100 bps).

        Méthode :
          - Obligations : utilise variation_prix() = formule duration + convexité
          - Actions     : ΔA ≈ −duration_actions × VA × Δr (approximation linéaire)
          - Immobilier  : ΔI ≈ −duration_imm × VI × Δr (approximation linéaire)
          - Total       : somme des trois variations

        Retourne un dictionnaire avec la variation par classe et au total.
        """
        # Variation de valeur pour chaque obligation (formule du 2e ordre)
        delta_obl = sum(o.variation_prix(delta_taux) for o in self.obligations)
        # Approximation linéaire pour les actifs non-obligataires
        delta_act = -self.duration_actions    * self.valeur_actions    * delta_taux
        delta_imm = -self.duration_immobilier * self.valeur_immobilier * delta_taux
        delta_tot = delta_obl + delta_act + delta_imm
        vt = self.valeur_totale
        return {
            "delta_obligations": delta_obl,
            "delta_actions":     delta_act,
            "delta_immobilier":  delta_imm,
            "delta_total":       delta_tot,
            # Variation en % du portefeuille total
            "delta_pct":         delta_tot / vt if vt else 0.0,
        }

    def tableau_obligations(self) -> pd.DataFrame:
        """
        DataFrame récapitulatif de chaque obligation.

        Rôle : produit le tableau affiché dans le dashboard ALM.
        Chaque ligne = une obligation avec toutes ses caractéristiques.
        [o.to_dict() for o in self.obligations] = compréhension de liste
        """
        if not self.obligations:
            return pd.DataFrame()
        return pd.DataFrame([o.to_dict() for o in self.obligations])


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE 3 — PROJECTION DU PASSIF
# ─────────────────────────────────────────────────────────────────────────────
# Rôle : modélise les engagements futurs d'un assureur vie en projetant
# les flux de trésorerie sortants (prestations) et entrants (primes)
# sur toute la durée résiduelle du portefeuille.
#
# Pourquoi modéliser le passif ?
# → Le BEL (Best Estimate Liability) = valeur actualisée des flux nets futurs
# → La duration du passif mesure la sensibilité du BEL aux taux
# → Ces deux grandeurs sont au cœur de l'analyse ALM
# ─────────────────────────────────────────────────────────────────────────────

class ProjectionPassif:
    """
    Projection des flux de trésorerie du passif vie sur toute la durée résiduelle.

    Modélise quatre décréments (causes de sortie du portefeuille) :
    * Mortalité  → paiement du capital décès
    * Rachat     → paiement de la valeur de rachat (90 % du capital)
    * Échéance   → paiement du capital à terme (dernière année)
    * Frais      → charges de gestion annuelles

    Le Best Estimate Liability (BEL) est la valeur actualisée des flux nets sortants.

    Parameters
    ----------
    age_moyen : int
        Âge moyen du portefeuille (ans).
    capital_moyen : float
        Capital moyen garanti par contrat (€).
    nb_contrats : int
        Effectif initial du portefeuille.
    duree : int
        Durée résiduelle maximale (ans).
    prime_annuelle : float
        Prime annuelle moyenne par contrat (€).
    taux_actualisation : float
        Taux d'actualisation (courbe des taux sans risque EIOPA).
    taux_mortalite_base : float
        Paramètre de calibration du taux de mortalité de base.
    taux_rachat : float
        Taux de rachat annuel (proportion de survivants qui résilie).
    taux_frais : float
        Taux de frais de gestion (% des engagements en cours).
    coefficient_mortalite : float
        Multiplicateur d'expérience (1.0 = table standard, >1 = surmortalité).
    """

    def __init__(
        self,
        age_moyen: int = 45,
        capital_moyen: float = 100_000,
        nb_contrats: int = 1_000,
        duree: int = 20,
        prime_annuelle: float = 1_500,
        taux_actualisation: float = 0.035,
        taux_mortalite_base: float = 0.005,
        taux_rachat: float = 0.03,
        taux_frais: float = 0.008,
        coefficient_mortalite: float = 1.0,
    ):
        self.age_moyen             = age_moyen
        self.capital_moyen         = capital_moyen
        self.nb_contrats           = nb_contrats
        self.duree                 = duree
        self.prime_annuelle        = prime_annuelle
        self.taux_actualisation    = taux_actualisation
        self.taux_mortalite_base   = taux_mortalite_base
        self.taux_rachat           = taux_rachat
        self.taux_frais            = taux_frais
        self.coefficient_mortalite = coefficient_mortalite
        # Cache : evite de recalculer la projection à chaque accès
        # None = pas encore calculé
        self._cache: Optional[pd.DataFrame] = None

    # ── Loi de mortalité ─────────────────────────────────────────────────────

    def _qx(self, age: int) -> float:
        """
        Probabilité de décès à l'âge x (loi de Makeham simplifiée).

        Rôle : calcule le taux de mortalité pour chaque année de la projection.
        Pourquoi Makeham ? C'est le modèle paramétrique standard en démographie
        actuarielle (force de mortalité = a + b×c^x).

        Paramètres calibrés a, b, c : donnent une mortalité croissante avec l'âge.
        coefficient_mortalite : permet d'ajuster l'expérience réelle par rapport
        à la table (ex: 1.2 = 20% de surmortalité observée).
        min(..., 0.999) : la probabilité de décès ne peut pas dépasser 1.
        """
        a, b, c = 0.001, 0.0001, 1.09   # paramètres Makeham calibrés
        return min(self.coefficient_mortalite * (a + b * c ** age), 0.999)

    # ── Projection des flux ───────────────────────────────────────────────────

    def projeter_flux(self) -> pd.DataFrame:
        """
        Projette les flux annuels sur toute la durée résiduelle.

        Rôle : moteur de calcul central du passif. Génère année par année
        tous les flux entrants (primes) et sortants (décès, rachats, frais,
        paiements à l'échéance) en tenant compte de l'évolution de la population.

        Algorithme :
          Pour chaque année t :
            1. Calculer les décès : nb_deces = nb_vivants × qx
            2. Calculer les rachats : nb_rachats = (nb_vivants - décès) × taux_rachat
            3. Calculer tous les flux monétaires
            4. Actualiser le flux net : flux_net × (1/(1+r))^t
            5. Mettre à jour nb_vivants pour l'année suivante

        Pattern cache (lazy evaluation) :
          Si la projection a déjà été calculée (self._cache is not None),
          on retourne directement le résultat sans recalculer.
          → Gain de performance quand bel et duration sont appelés successivement.

        Colonnes retournées :
          annee, nb_vivants, nb_deces, nb_rachats,
          primes, capitaux_deces, rachats_montant, maturite, frais,
          flux_net, facteur_actualisation, flux_net_actualise
        """
        if self._cache is not None:  # cache hit : on retourne le résultat déjà calculé
            return self._cache

        records    = []
        nb_vivants = float(self.nb_contrats)  # float pour les calculs décimaux (décès partiels)

        for t in range(1, self.duree + 1):
            # Taux de mortalité de l'année (dépend de l'âge atteint)
            qx = self._qx(self.age_moyen + t - 1)

            # ── Décréments (sorties du portefeuille) ──────────────────────────
            # Les décès sont appliqués en premier, puis les rachats sur le solde
            nb_deces   = nb_vivants * qx
            nb_apres_d = nb_vivants - nb_deces          # survivants après décès
            nb_rachats = nb_apres_d * self.taux_rachat  # rachats = % des survivants
            nb_fin_t   = nb_apres_d - nb_rachats        # vivants en fin d'année t

            # ── Flux de trésorerie ────────────────────────────────────────────
            # Flux entrant : primes payées par les assurés encore en vie
            primes = nb_vivants * self.prime_annuelle

            # Flux sortants : prestations versées aux assurés ou bénéficiaires
            cap_deces = nb_deces  * self.capital_moyen          # capital décès = capital plein
            rach_mont = nb_rachats * self.capital_moyen * 0.90  # rachat = 90% du capital (pénalité 10%)
            # Paiement à l'échéance uniquement la dernière année
            maturite  = nb_fin_t * self.capital_moyen if t == self.duree else 0.0
            # Frais = % des engagements (capital × nb vivants)
            frais     = nb_vivants * self.capital_moyen * self.taux_frais

            # Flux net = entrées - sorties (positif = l'assureur reçoit plus qu'il ne paie)
            flux_net = primes - cap_deces - rach_mont - maturite - frais

            # Facteur d'actualisation : ramène le flux futur en valeur d'aujourd'hui
            # v_t = 1 / (1 + r)^t    (ex: r=3.5%, t=5 → v_5 = 0.842)
            v_t = 1 / (1 + self.taux_actualisation) ** t

            records.append({
                "annee":                 t,
                "nb_vivants":            nb_vivants,
                "nb_deces":              nb_deces,
                "nb_rachats":            nb_rachats,
                "primes":                primes,
                "capitaux_deces":        cap_deces,
                "rachats_montant":       rach_mont,
                "maturite":              maturite,
                "frais":                 frais,
                "flux_net":              flux_net,
                "facteur_actualisation": v_t,
                "flux_net_actualise":    flux_net * v_t,
            })

            # Mise à jour : la population de l'année t+1 = survivants de l'année t
            nb_vivants = nb_fin_t

        # Conversion liste de dicts → DataFrame pandas et mise en cache
        self._cache = pd.DataFrame(records)
        return self._cache

    # ── Indicateurs agrégés ───────────────────────────────────────────────────

    @property
    def bel(self) -> float:
        """
        Best Estimate Liability (BEL) — Engagement net actualisé.

        Rôle : calcule la valeur actuelle des flux nets futurs sortants.
        C'est la "vraie valeur" des engagements de l'assureur.

        Formule : BEL = −PV(flux nets)
        Le signe négatif est appliqué car :
          - Si sorties > entrées (cas habituel en phase de prestations)
            → flux_net < 0 → −(somme négative) > 0 → BEL positif (passif)

        .sum() = somme de la colonne flux_net_actualise du DataFrame
        """
        return -self.projeter_flux()["flux_net_actualise"].sum()

    @property
    def duration(self) -> float:
        """
        Duration du passif (en années).

        Rôle : mesure la sensibilité du BEL aux mouvements de taux.
        C'est l'équivalent de la duration obligataire mais pour les engagements.

        Formule : D_L = Σ(t × VA(sortie_t)) / Σ(VA(sortie_t))
        = centre de gravité temporel des flux sortants actualisés.

        Interprétation :
          D_L = 10 ans → en moyenne, les prestations sont versées dans 10 ans.
          Si D_A < D_L → actif "trop court" → risque si taux baissent.
          Si D_A > D_L → actif "trop long" → risque si taux montent.
        """
        df      = self.projeter_flux()
        # Total des sorties annuelles (décès + rachats + paiement final + frais)
        sorties = (
            df["capitaux_deces"]
            + df["rachats_montant"]
            + df["maturite"]
            + df["frais"]
        )
        # Valeur actualisée des sorties
        va_sorties = sorties * df["facteur_actualisation"]
        total = va_sorties.sum()
        if total == 0:
            return 0.0
        # Duration = moyenne pondérée des années par les sorties actualisées
        return float((df["annee"] * va_sorties).sum() / total)

    @property
    def duration_modifiee(self) -> float:
        """
        Duration modifiée du passif.

        Rôle : utilisée dans le calcul de la sensibilité du BEL aux taux.
        ΔBEL ≈ −D_L_mod × BEL × Δr
        """
        return self.duration / (1 + self.taux_actualisation)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE 4 — ANALYSE ALM
# ─────────────────────────────────────────────────────────────────────────────
# Rôle : moteur central de l'analyse ALM. Combine un PortefeuilleActif et
# un ProjectionPassif pour produire tous les indicateurs ALM :
#   - Surplus économique (EVE = Actif - BEL)
#   - Duration Gap (mesure du déséquilibre actif/passif)
#   - DV01 (sensibilité du surplus à 1 point de base)
#   - Scénarios de chocs de taux (stress-test)
#   - Immunisation de Redington (conditions formelles)
#   - Projection du bilan sur un horizon donné
#
# Pourquoi cette classe séparée ?
# → Séparer la logique de calcul (actif / passif) de la logique d'analyse.
# → Respecter le principe de responsabilité unique (chaque classe fait une chose).
# ─────────────────────────────────────────────────────────────────────────────

class AnalyseALM:
    """
    Moteur central de l'analyse ALM.

    Combine un ``PortefeuilleActif`` et un ``ProjectionPassif`` pour calculer :
    * le surplus économique (EVE = Actif − BEL) ;
    * le duration gap (DGap = D_A − BEL/A × D_L) ;
    * le DV01 du surplus ;
    * l'impact de chocs de taux parallèles ;
    * la conformité aux conditions d'immunisation de Redington ;
    * la projection du bilan sur un horizon donné.

    Parameters
    ----------
    actif : PortefeuilleActif
    passif : ProjectionPassif
    """

    def __init__(self, actif: PortefeuilleActif, passif: ProjectionPassif):
        self.actif  = actif
        self.passif = passif

    # ── Bilan économique instantané ───────────────────────────────────────────

    @property
    def valeur_actif(self) -> float:
        """Valeur de marché totale du portefeuille d'actifs."""
        return self.actif.valeur_totale

    @property
    def bel(self) -> float:
        """Best Estimate Liability (valeur actualisée des engagements nets)."""
        return self.passif.bel

    @property
    def surplus(self) -> float:
        """
        Surplus économique : A − BEL.

        Rôle : mesure la richesse économique nette de l'assureur.
        Aussi appelé EVE (Economic Value of Equity) ou fonds propres économiques.

        Interprétation :
          Surplus > 0 → l'assureur peut faire face à ses engagements (solvable)
          Surplus < 0 → déficit économique (insolvable en valeur de marché)
        """
        return self.valeur_actif - self.bel

    @property
    def ratio_couverture(self) -> float:
        """
        Ratio A / BEL.

        Rôle : mesure le degré de couverture des engagements.
        Ratio = 1.20 → l'assureur a 20% de capital excédentaire.
        Ratio < 1    → déficit (passif > actif).
        """
        return self.valeur_actif / self.bel if self.bel else float("inf")

    # ── Duration Gap ─────────────────────────────────────────────────────────

    @property
    def duration_actif(self) -> float:
        """Duration de Macaulay pondérée du portefeuille actif."""
        return self.actif.duration

    @property
    def duration_passif(self) -> float:
        """Duration du passif (des flux sortants actualisés)."""
        return self.passif.duration

    @property
    def duration_gap(self) -> float:
        """
        Duration Gap (en années).

        Rôle : mesure central de l'ALM. Quantifie le déséquilibre de sensibilité
        entre l'actif et le passif face aux mouvements de taux.

        Formule : DGap = D_A − (BEL / A) × D_L

        Interprétation :
          DGap > 0 → actifs plus sensibles que les passifs :
                     si les taux montent → Actif baisse plus que BEL → Surplus baisse
          DGap < 0 → passifs plus sensibles :
                     si les taux baissent → BEL monte plus qu'Actif → Surplus baisse
          DGap = 0 → immunisation parfaite (objectif ALM idéal)

        La formule DGap = D_A - (BEL/A) × D_L tient compte du levier :
        le passif est pondéré par le ratio BEL/A pour être comparable à l'actif.
        """
        if not self.valeur_actif:
            return 0.0
        return self.duration_actif - (self.bel / self.valeur_actif) * self.duration_passif

    @property
    def dv01(self) -> float:
        """
        DV01 (Dollar Value of 1 basis point) du surplus.

        Rôle : mesure concrète de la sensibilité au taux, en euros par point de base.
        Répond à la question : "Si les taux montent de 1 bp, de combien change le surplus ?"

        Formule : DV01 ≈ −DGap × A × 0.0001
          (0.0001 = 1 bp = 0.01% en fraction décimale)

        Exemple : DV01 = −50 000 € → chaque bp de hausse de taux coûte 50 000€ de surplus.
        """
        return -self.duration_gap * self.valeur_actif * 0.0001

    # ── Scénarios de taux ────────────────────────────────────────────────────

    def impact_choc_taux(self, delta_taux: float) -> Dict[str, float]:
        """
        Impact d'un choc de taux parallèle sur le bilan économique.

        Rôle : stress-test qui simule ce qui se passe si toute la courbe des taux
        monte ou descend de delta_taux (choc parallèle = même choc sur toutes les maturités).

        Méthode :
          ΔA   ≈ −D_A_mod × A × Δy + ½ × C_A × A × (Δy)²   (2e ordre pour l'actif)
          ΔBEL ≈ −D_L_mod × BEL × Δy                          (1er ordre pour le passif)
          ΔS   = ΔA − ΔBEL                                     (variation du surplus)

        Paramètre :
          delta_taux : choc en fraction décimale
                       (ex: 0.01 = +100 bps, −0.005 = −50 bps)

        Retourne un dictionnaire complet avec tous les indicateurs stressés.
        """
        # Variation de l'actif (utilise la sensibilité propre de chaque obligation)
        imp_actif = self.actif.impact_choc_taux(delta_taux)
        delta_a   = imp_actif["delta_total"]

        # Variation du BEL (approximation au 1er ordre avec la duration modifiée du passif)
        delta_bel = -self.passif.duration_modifiee * self.bel * delta_taux

        # Variation du surplus = variation actif - variation passif
        delta_s = delta_a - delta_bel

        return {
            "Choc (bps)":            round(delta_taux * 10_000),    # en points de base
            "Var. Actif (EUR)":      round(delta_a),
            "Var. BEL (EUR)":        round(delta_bel),
            "Var. Surplus (EUR)":    round(delta_s),
            "Actif stresse (EUR)":   round(self.valeur_actif + delta_a),
            "BEL stresse (EUR)":     round(self.bel + delta_bel),
            "Surplus stresse (EUR)": round(self.surplus + delta_s),
            # Variation en % du surplus actuel (None si surplus = 0 pour éviter /0)
            "Variation surplus (%)": round(100 * delta_s / self.surplus, 2)
                                     if self.surplus else None,
        }

    def tableau_scenarios(self) -> pd.DataFrame:
        """
        Tableau de scénarios de taux standardisés.

        Rôle : génère les 7 scénarios standard utilisés en stress-testing ALM.
        Ces scénarios couvrent des hausses et des baisses de taux typiques.

        Chocs : −200, −100, −50, 0, +50, +100, +200 bps.
        (exprimés en fraction décimale : −0.02, −0.01, −0.005, 0, 0.005, 0.01, 0.02)
        """
        chocs = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]
        return pd.DataFrame([self.impact_choc_taux(d) for d in chocs])

    # ── Immunisation de Redington ─────────────────────────────────────────────

    def rapport_immunisation(self) -> Dict:
        """
        Vérifie les trois conditions d'immunisation de Redington (1952).

        Rôle : évalue si le portefeuille est "immunisé" contre les petits chocs
        de taux parallèles, c'est-à-dire si le surplus ne peut pas baisser.

        Les 3 conditions de Redington :
          1. VA(actifs) = VA(passifs)        → ratio de couverture ≈ 1
          2. D_A = D_L                        → duration gap ≈ 0
          3. Convexité(actifs) > Convexité(passifs) → protection asymétrique

        Note : l'immunisation de Redington protège contre les petits chocs
        parallèles. Les chocs de forte amplitude ou non-parallèles peuvent
        toujours affecter le surplus.

        Retourne un dictionnaire complet utilisé par le dashboard ALM.
        """
        ecart_val = abs(self.valeur_actif - self.bel) / self.bel if self.bel else None
        ecart_dur = abs(self.duration_actif - self.duration_passif)

        return {
            "Valeur actif (EUR)":      round(self.valeur_actif),
            "BEL (EUR)":               round(self.bel),
            "Surplus (EUR)":           round(self.surplus),
            "Ratio couverture":        round(self.ratio_couverture, 4),
            "Duration actif (ans)":    round(self.duration_actif, 2),
            "Duration passif (ans)":   round(self.duration_passif, 2),
            "Duration Gap (ans)":      round(self.duration_gap, 2),
            "DV01 (EUR/bp)":           round(self.dv01),
            "Convexite actif":         round(self.actif.convexite, 2),
            "Cond1 ecart valeur (%)":  round(ecart_val * 100, 2) if ecart_val else None,
            "Cond2 ecart duration":    round(ecart_dur, 2),
            "Cond3 convexite OK":      self.actif.convexite > 0,
            # Immunisation = duration gap < 0.5 ans ET écart de valeur < 5%
            "Immunisation OK":         ecart_dur < 0.5 and (ecart_val or 1) < 0.05,
        }

    # ── Projection du bilan ───────────────────────────────────────────────────

    def projection_bilan(self, horizon: int = 10) -> pd.DataFrame:
        """
        Projection du bilan économique sur un horizon donné.

        Rôle : simule l'évolution du couple Actif/BEL dans le temps,
        en tenant compte des flux annuels (prestations, primes, rendement).

        Hypothèses de projection :
          - L'actif croît au rendement moyen actuel (rendement_moyen)
          - Les flux nets du passif diminuent l'actif (prestations > primes)
          - Le BEL décroît au fur et à mesure que les engagements sont honorés

        Algorithme :
          Pour chaque année t :
            a_t = a_{t-1} × (1 + r) - prestations_nettes_t
            bel_t = bel_{t-1} - (sorties_t - primes_t)

        Paramètre :
          horizon : nombre d'années à projeter (3 à 15)

        Retourne un DataFrame avec : annee, valeur_actif, bel, surplus,
        ratio_couverture, flux_net.
        """
        df_p  = self.passif.projeter_flux()
        r     = self.actif.rendement_moyen   # rendement annuel de l'actif (constant)
        a_t   = self.valeur_actif            # valeur initiale de l'actif
        bel_t = self.bel                     # BEL initial

        records = []
        for _, row in df_p.iterrows():
            t = int(row["annee"])
            if t > horizon:
                break

            # Actif : produits financiers − prestations nettes de l'année
            # flux_net < 0 si les sorties > entrées → prestations = −flux_net > 0
            prestations = -(row["flux_net"])
            a_t = a_t * (1 + r) - prestations

            # BEL : diminue des engagements honorés dans l'année
            # max(0, ...) : le BEL ne peut pas devenir négatif
            bel_t = max(
                0.0,
                bel_t
                - row["capitaux_deces"]   # capital versé aux décès
                - row["rachats_montant"]  # rachat versé aux résiliants
                - row["maturite"]         # capital versé à l'échéance
                - row["frais"]            # frais prélevés
                + row["primes"],          # primes reçues (réduisent le BEL)
            )

            records.append({
                "Année":            t,
                "Valeur actif (€)": round(a_t),
                "BEL (€)":          round(bel_t),
                "Surplus (€)":      round(a_t - bel_t),
                "Ratio couv.":      round(a_t / bel_t, 3) if bel_t else None,
                "Flux net (€)":     round(row["flux_net"]),
            })

        return pd.DataFrame(records)
