# ══════════════════════════════════════════════════════════════
#  modules/solvabilite2.py — SOLVABILITÉ II
#
#  Ce fichier contient 3 classes :
#   1. SCR_NonVie   → calcule le SCR de souscription Non-Vie
#   2. SCR_Vie      → calcule le SCR de souscription Vie
#   3. SolvabiliteII → agrège les SCR et calcule le ratio SII
#
#  Rappel Solvabilité II :
#  Le SCR (Solvency Capital Requirement) = capital minimum
#  que l'assureur doit détenir pour absorber des pertes
#  avec une probabilité de 99.5% sur 1 an.
#
#  Ratio SII = Fonds Propres / SCR × 100
#  Si ratio ≥ 100% → l'assureur est solvable ✅
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd

# ── RAPPEL NUMPY MATRICIEL ────────────────────────────────────
# np.array([[1,2],[3,4]])  → crée une matrice 2×2
# v @ M @ v               → produit matriciel (formule agrégation SCR)
# np.sqrt()               → racine carrée
# La formule SCR agrégé : SCR = √(v' × Corr × v)
# où v = vecteur des SCR par module/ligne de métier
# ─────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────
class SCR_NonVie:
    """
    Calcule le SCR de souscription Non-Vie (formule standard SII).

    La formule standard décompose le risque en :
      - Risque de primes   : risque sur les nouvelles affaires
      - Risque de réserves : risque sur les sinistres en cours

    SCR_lm = √(SCR_P² + SCR_R² + 0.5 × SCR_P × SCR_R)
    """

    # ── Constantes de classe (partagées par tous les objets) ──
    # Les facteurs de volatilité sont fixés par la réglementation
    # (Delegated Acts, Annexe II)
    # Plus σ est élevé → risque plus grand → SCR plus élevé
    SIGMA_PRIMES = {
        "Auto RC":       0.10,   # 10% de volatilité sur les primes
        "Auto Dommages": 0.08,
        "Incendie":      0.14,
        "RC Generale":   0.17,
        "Transport":     0.19,
        "Credit":        0.21,   # le crédit est le plus risqué
        "Santé":         0.05,   # santé : volatilité plus faible (SII non-SLT)
    }

    SIGMA_RESERVES = {
        "Auto RC":       0.09,
        "Auto Dommages": 0.08,
        "Incendie":      0.11,
        "RC Generale":   0.14,
        "Transport":     0.18,
        "Credit":        0.19,
        "Santé":         0.05,
    }

    # list(dict.keys()) = récupère la liste des clés du dictionnaire
    # Ex: ["Auto RC", "Auto Dommages", "Incendie", ...]
    LIGNES = list(SIGMA_PRIMES.keys())

    # Matrice de corrélation entre lignes de métier (réglementaire)
    # 1.00 = corrélation parfaite (même ligne)
    # 0.50 = forte corrélation (Auto RC et Auto Dom)
    # 0.25 = corrélation modérée (autres combinaisons)
    CORR = np.array([
        #  AutoRC AutoDo Incen  RCGen  Trans  Cred   Santé
        [1.00, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25],  # Auto RC
        [0.50, 1.00, 0.25, 0.25, 0.25, 0.25, 0.25],  # Auto Dommages
        [0.25, 0.25, 1.00, 0.25, 0.25, 0.25, 0.25],  # Incendie
        [0.25, 0.25, 0.25, 1.00, 0.25, 0.25, 0.25],  # RC Generale
        [0.25, 0.25, 0.25, 0.25, 1.00, 0.25, 0.25],  # Transport
        [0.25, 0.25, 0.25, 0.25, 0.25, 1.00, 0.25],  # Credit
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 1.00],  # Santé
    ])

    def __init__(self, primes: dict, provisions: dict):
        """
        primes     : dict {ligne_metier: montant_primes}
        provisions : dict {ligne_metier: montant_provisions}
        Exemple : {"Auto RC": 5_000_000, "Incendie": 2_000_000}
        """
        self.primes     = primes
        self.provisions = provisions

    def scr_par_ligne(self):
        """
        Calcule le SCR pour chaque ligne de métier séparément.

        Volume de primes    : Vp = primes × 3 (facteur réglementaire)
        Volume de réserves  : Vr = provisions
        SCR Primes          : SCR_P = 3 × σ_p × Vp
        SCR Réserves        : SCR_R = 3 × σ_r × Vr
        SCR ligne de métier : √(SCR_P² + SCR_R² + 0.5 × SCR_P × SCR_R)
        """
        details = {}   # dictionnaire pour stocker les résultats

        for lm in self.primes:   # lm = ligne de métier
            # continue = passe à la ligne suivante si non reconnue
            if lm not in self.SIGMA_PRIMES:
                continue

            # Volume primes (×3 = facteur de chargement réglementaire)
            Vp = self.primes[lm] * 3
            # .get(lm, 0) = retourne 0 si la clé n'existe pas
            Vr = self.provisions.get(lm, 0)

            # SCR par composante
            scr_p  = 3 * self.SIGMA_PRIMES[lm]   * Vp
            scr_r  = 3 * self.SIGMA_RESERVES[lm] * Vr

            # Formule combinée SII (corrélation interne = 0.5)
            # np.sqrt() = racine carrée
            scr_lm = np.sqrt(scr_p**2 + scr_r**2 + 0.5 * scr_p * scr_r)

            details[lm] = {
                "Primes":       self.primes[lm],
                "Provisions":   self.provisions.get(lm, 0),
                "SCR_Primes":   round(scr_p),
                "SCR_Reserves": round(scr_r),
                "SCR_Total":    round(scr_lm),
            }

        # pd.DataFrame(dict).T = transpose le dict en tableau
        # .T = transposée (lignes ↔ colonnes)
        return pd.DataFrame(details).T

    def scr_total(self):
        """
        Agrège les SCR par ligne de métier avec la matrice de corrélation.

        Formule matricielle : SCR = √(v' × Corr × v)
        où v = vecteur colonne des SCR par ligne de métier

        En Python : np.sqrt(v @ CORR @ v)
        @ = opérateur de multiplication matricielle (Python 3.5+)
        """
        df = self.scr_par_ligne()

        # Construit le vecteur des SCR dans l'ordre défini par LIGNES
        # Si une ligne n'est pas dans le df, on met 0
        scr_vec = np.array([
            df.loc[lm, "SCR_Total"] if lm in df.index else 0.0
            for lm in self.LIGNES
        ])

        # Produit quadratique : v @ M @ v = Σᵢ Σⱼ Corr(i,j) × SCRᵢ × SCRⱼ
        SCR = np.sqrt(scr_vec @ self.CORR @ scr_vec)
        return round(SCR)


# ──────────────────────────────────────────────────────────────
class SCR_Vie:
    """
    Calcule le SCR de souscription Vie (formule standard SII).

    Principe : on applique des "chocs" réglementaires sur
    les hypothèses (mortalité, longévité, rachats...) et on
    mesure l'impact sur les fonds propres.

    SCR_risque = Impact(choc) sur le Best Estimate
    """

    # Chocs réglementaires (fixés par les Delegated Acts SII)
    CHOCS = {
        "mortalite":   0.15,    # +15% taux de mortalité → impact décès
        "longevite":   0.20,    # -20% mortalité → les rentiers vivent plus longtemps
        "invalidite":  0.35,    # +35% incidence invalidité
        "rachat_up":   0.50,    # +50% taux de rachat
        "rachat_dn":   0.50,    # -50% taux de rachat
        "frais":       0.10,    # +10% frais de gestion
        "catastrophe": 0.0015,  # 1.5‰ du capital sous risque
    }

    # Matrice de corrélation SII Vie (certains risques sont négativement
    # corrélés : ex. mortalité et longévité vont en sens inverse)
    CORR = np.array([
        #  mort   long   inval  rach   frais  catas
        [ 1.00, -0.25,  0.25,  0.00,  0.25,  0.25],  # mortalité
        [-0.25,  1.00, -0.25,  0.25, -0.25,  0.00],  # longévité
        [ 0.25, -0.25,  1.00,  0.00,  0.50,  0.25],  # invalidité
        [ 0.00,  0.25,  0.00,  1.00,  0.00,  0.00],  # rachat
        [ 0.25, -0.25,  0.50,  0.00,  1.00,  0.25],  # frais
        [ 0.25,  0.00,  0.25,  0.00,  0.25,  1.00],  # catastrophe
    ])

    def __init__(self, be_deces, be_rentes, be_frais,
                 be_rachat, capital_risque):
        """
        be_*          : Best Estimate par type de risque (en €)
        capital_risque : capital total sous risque de décès
        """
        # On organise les BE dans un dictionnaire
        self.be = {
            "mortalite":   be_deces,
            "longevite":   be_rentes,
            "invalidite":  be_frais * 0.5,   # hypothèse simplifiée
            "rachat":      be_rachat,
            "frais":       be_frais,
            "catastrophe": capital_risque,
        }

    def calculer(self):
        """
        Calcule le SCR Vie en appliquant les chocs et en agrégeant.

        Pour chaque risque :
          SCR_risque = BE_risque × choc_réglementaire

        Puis agrégation avec la matrice de corrélation.
        Retourne (SCR_total, dict_chocs_détaillés)
        """
        # Application des chocs sur les Best Estimates
        chocs = {
            "mortalite":   self.be["mortalite"]   * self.CHOCS["mortalite"],
            "longevite":   self.be["longevite"]   * self.CHOCS["longevite"],
            "invalidite":  self.be["invalidite"]  * self.CHOCS["invalidite"],
            "rachat":      self.be["rachat"]      * self.CHOCS["rachat_up"],
            "frais":       self.be["frais"]       * self.CHOCS["frais"],
            "catastrophe": self.be["catastrophe"] * self.CHOCS["catastrophe"],
        }

        # np.array(list(...)) = convertit les valeurs du dict en tableau
        v   = np.array(list(chocs.values()))
        # Agrégation matricielle
        SCR = np.sqrt(v @ self.CORR @ v)

        print("\n── SCR Vie détaillé ──")
        # .items() = itère sur les paires (clé, valeur) du dict
        for k, val in chocs.items():
            print(f"  SCR {k:<15}: {val:>12,.0f} €")
        print(f"  {'SCR Vie TOTAL':<15}: {SCR:>12,.0f} €")

        # On retourne le SCR ET le détail des chocs (pour le dashboard)
        return round(SCR), chocs


# ──────────────────────────────────────────────────────────────
class SolvabiliteII:
    """
    Calcule le BSCR (Basic SCR) et le ratio de solvabilité final.

    BSCR = agrégation des modules : Non-Vie + Vie + Marché
    SCR  = BSCR × (1 - ajustement fiscalité)
    Ratio = Fonds Propres Éligibles / SCR × 100

    Seuil réglementaire : Ratio ≥ 100%
    """

    # Matrice de corrélation entre modules du BSCR
    # Non-Vie et Vie ne sont pas corrélés (0.00)
    # Tous deux sont modérément corrélés au Marché (0.25)
    CORR_BSCR = np.array([
        #  NL    Vie   Mch
        [1.00, 0.00, 0.25],   # Non-Vie
        [0.00, 1.00, 0.25],   # Vie
        [0.25, 0.25, 1.00],   # Marché
    ])

    def __init__(self, scr_nl, scr_vie, scr_marche,
                 fonds_propres, taux_abs=0.15):
        """
        scr_nl, scr_vie, scr_marche : SCR de chaque module (en €)
        fonds_propres               : fonds propres éligibles (en €)
        taux_abs                    : taux d'absorption fiscale (15%)
                                      réduit le SCR (impôts différés)
        """
        self.scr_nl        = scr_nl
        self.scr_vie       = scr_vie
        self.scr_marche    = scr_marche
        self.fonds_propres = fonds_propres
        self.taux_abs      = taux_abs

    def bscr(self):
        """
        Calcule le Basic SCR par agrégation matricielle des 3 modules.
        BSCR = √(v' × Corr_BSCR × v)
        """
        v = np.array([self.scr_nl, self.scr_vie, self.scr_marche])
        return round(np.sqrt(v @ self.CORR_BSCR @ v))

    def scr(self):
        """
        SCR final = BSCR réduit de l'ajustement fiscalité.
        Les impôts différés peuvent absorber une partie des pertes.
        SCR = BSCR × (1 - taux_abs)
        """
        return round(self.bscr() * (1 - self.taux_abs))

    def ratio_solvabilite(self):
        """
        Ratio de solvabilité = FP / SCR × 100
        Ex: ratio = 150% → l'assureur a 1.5x le capital requis
        """
        return round(self.fonds_propres / self.scr() * 100, 1)

    def rapport(self):
        """Affiche le rapport de solvabilité complet et retourne les KPIs."""
        bscr  = self.bscr()
        scr_f = self.scr()
        ratio = self.ratio_solvabilite()

        # Séparateurs pour la mise en forme du rapport
        sep  = "=" * 50
        dash = "-" * 40

        print(f"\n{sep}")
        print(f"{'RAPPORT SOLVABILITÉ II':^50}")
        print(f"{sep}")
        print(f"  SCR Non-Vie        : {self.scr_nl:>12,.0f} €")
        print(f"  SCR Vie            : {self.scr_vie:>12,.0f} €")
        print(f"  SCR Marché         : {self.scr_marche:>12,.0f} €")
        print(f"  {dash}")
        print(f"  BSCR               : {bscr:>12,.0f} €")
        # Taux d'absorption en % avec formatage
        print(f"  Ajustement fiscal  :         -{self.taux_abs*100:.0f}%")
        print(f"  SCR FINAL          : {scr_f:>12,.0f} €")
        print(f"  Fonds Propres      : {self.fonds_propres:>12,.0f} €")
        print(f"  {dash}")
        # Expression ternaire : valeur_si_vrai if condition else valeur_si_faux
        emoji = "OK" if ratio >= 100 else "KO"
        print(f"  RATIO SII          : {ratio:>11.1f}% [{emoji}]")
        print(f"{sep}")

        # Retourne un dict pour utilisation dans le dashboard
        return {"BSCR": bscr, "SCR": scr_f, "Ratio": ratio}
