# ══════════════════════════════════════════════════════════════
#  modules/non_vie.py — ASSURANCE NON-VIE
#
#  Ce fichier contient 3 classes :
#   1. PortefeuilleAuto → simule un portefeuille automobile
#   2. Tarification     → GLM pour calculer les primes
#   3. ChainLadder      → provisionnement IBNR
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
from scipy import stats

# statsmodels = librairie pour les modèles statistiques (GLM, etc.)
import statsmodels.api as sm
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import warnings
# On masque les alertes non-critiques pour garder la console propre
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────
class PortefeuilleAuto:
    """
    Simule un portefeuille d'assurance automobile.
    Génère des assurés fictifs avec leurs caractéristiques
    et leurs sinistres selon des lois statistiques réalistes.
    """

    def __init__(self, n_contrats=10000, seed=42):
        """
        n_contrats : nombre de contrats à simuler
        seed       : graine aléatoire (42 = résultats reproductibles)
                     Avec la même seed, on obtient toujours les mêmes données.
        """
        self.n = n_contrats
        # np.random.seed() fixe le générateur aléatoire
        # → même résultats à chaque exécution
        np.random.seed(seed)
        # On appelle la méthode _simuler() pour remplir le DataFrame
        self.df = self._simuler()

    def _simuler(self):
        """
        Génère les données du portefeuille.
        Utilise des lois de probabilité pour simuler
        les caractéristiques et les sinistres.
        """
        n = self.n  # raccourci local

        # ── Caractéristiques des assurés ──────────────────────
        df = pd.DataFrame({
            "id":          range(n),   # identifiant unique

            # np.random.randint(a, b, n) = n entiers entre a et b-1
            "age":         np.random.randint(18, 75, n),
            "anciennete":  np.random.randint(0, 50, n),

            # np.random.choice(liste, n, p=proba) = tirage aléatoire
            # p = probabilités de chaque choix (doit sommer à 1)
            "zone": np.random.choice(
                ["urbain", "rural", "mixte"], n,
                p=[0.5, 0.3, 0.2]   # 50% urbain, 30% rural, 20% mixte
            ),
            "categorie_vh": np.random.choice(
                ["citadine", "berline", "SUV", "sport"], n,
                p=[0.4, 0.3, 0.2, 0.1]
            ),

            # np.random.uniform(a, b, n) = n valeurs entre a et b
            "bonus_malus": np.random.uniform(0.5, 3.5, n),

            # exposition = fraction d'année assurée (0.1 à 1.0)
            "exposition":  np.random.uniform(0.1, 1.0, n),
        })

        # ── Variables binaires (0 ou 1) ───────────────────────
        # (condition).astype(int) convertit True/False en 1/0
        df["jeune"]  = (df["age"] < 25).astype(int)   # 1 si < 25 ans
        df["senior"] = (df["age"] > 65).astype(int)   # 1 si > 65 ans
        df["novice"] = (df["anciennete"] < 2).astype(int)  # 1 si < 2 ans de permis

        # ── Génération des sinistres ──────────────────────────
        # Fréquence : loi de Poisson (nb sinistres par an)
        # λ (lambda) = fréquence attendue, varie selon le profil
        lambda_base = 0.07   # fréquence de base = 7 sinistres pour 100 assurés/an

        # Chaque assuré a sa propre fréquence selon son profil
        lambda_ind = (
            lambda_base
            * df["bonus_malus"]          # malus augmente la fréquence
            * (1 + 0.4 * df["jeune"])    # +40% pour les jeunes
            * (1 + 0.2 * df["novice"])   # +20% pour les novices
            * df["exposition"]           # prorata temporis
        )

        # np.random.poisson(lambda) = tire un nb entier (0, 1, 2, ...)
        # selon la loi de Poisson de paramètre lambda
        df["nb_sinistres"] = np.random.poisson(lambda_ind)

        # Coût : loi log-normale (toujours positive, asymétrique à droite)
        # Réaliste car les gros sinistres existent mais sont rares
        # np.where(condition, valeur_si_vrai, valeur_si_faux)
        df["cout_total"] = np.where(
            df["nb_sinistres"] > 0,
            # lognormal(mean, sigma) : mean = ln(coût moyen) ≈ 2440€
            np.random.lognormal(mean=7.8, sigma=0.9, size=n) * df["nb_sinistres"],
            0.0   # pas de sinistre → coût = 0
        )

        return df

    def statistiques(self):
        """
        Affiche les statistiques descriptives du portefeuille.
        groupby() = regroupe les lignes par valeur d'une colonne.
        """
        print("── Résumé portefeuille ──")
        print(f"Nb contrats    : {self.n:>8,}")
        print(f"Fréquence moy. : {self.df['nb_sinistres'].mean():>8.4f}")

        # Filtre uniquement les contrats avec sinistres pour le coût moyen
        sin_pos = self.df[self.df["cout_total"] > 0]["cout_total"]
        print(f"Coût moyen     : {sin_pos.mean():>8,.0f} €")

        # .groupby("zone") = groupe par zone géographique
        # .agg() = applique des fonctions d'agrégation à chaque groupe
        stats_zone = self.df.groupby("zone").agg(
            nb_contrats = ("id",           "count"),   # nb de lignes
            freq_moy    = ("nb_sinistres", "mean"),    # moyenne
        ).round(4)

        print("\n── Stats par zone ──")
        print(stats_zone)
        return stats_zone


# ──────────────────────────────────────────────────────────────
class Tarification:
    """
    Tarification par GLM (Generalized Linear Model).

    Le GLM est le modèle standard en tarification non-vie.
    On sépare la prime pure en deux composantes :
      Prime pure = Fréquence × Coût moyen

    - Fréquence : combien de sinistres en moyenne ? → Loi de Poisson
    - Coût      : quel est le coût moyen ?           → Loi Gamma
    """

    def __init__(self, df: pd.DataFrame):
        """df : DataFrame du portefeuille (issu de PortefeuilleAuto)."""
        # .copy() = on travaille sur une copie pour ne pas modifier l'original
        self.df   = df.copy()
        self.freq = None   # modèle GLM fréquence (pas encore calculé)
        self.cout = None   # modèle GLM coût     (pas encore calculé)

    def ajuster_frequence(self):
        """
        Ajuste un GLM Poisson pour modéliser la fréquence des sinistres.

        Pourquoi Poisson ?
          - Données de comptage : 0, 1, 2, 3 sinistres...
          - Toujours ≥ 0
          - Lien logarithmique : log(λ) = Xβ → λ = exp(Xβ) > 0

        La formule '~' signifie "expliqué par" (comme en maths : Y = f(X))
        C(zone) = variable catégorielle → crée des variables indicatrices
        offset  = correction pour l'exposition (durée d'assurance)
        """
        print("Ajustement GLM Fréquence (Poisson)...")

        self.freq = smf.glm(
            # Formule : variable_cible ~ variables_explicatives
            formula = ("nb_sinistres ~ age + bonus_malus "
                       "+ C(zone) + C(categorie_vh) + jeune"),
            data    = self.df,
            # Famille Poisson avec lien logarithmique
            family  = sm.families.Poisson(
                link = sm.families.links.Log()
            ),
            # Offset = log(exposition) pour tenir compte de la durée
            offset  = np.log(self.df["exposition"])
        ).fit()   # .fit() = entraîne le modèle sur les données

        # AIC (Akaike Information Criterion) = mesure la qualité du modèle
        # Plus l'AIC est faible, meilleur est le modèle
        print(f"AIC Fréquence : {self.freq.aic:.2f}")
        return self.freq

    def ajuster_cout(self):
        """
        Ajuste un GLM Gamma pour modéliser le coût moyen par sinistre.

        Pourquoi Gamma ?
          - Valeurs toujours positives (coût ≥ 0)
          - Distribution asymétrique à droite (quelques gros sinistres)
          - Lien logarithmique : log(μ) = Xβ → μ = exp(Xβ) > 0
        """
        print("Ajustement GLM Coût (Gamma)...")

        # On ne garde que les contrats avec au moins 1 sinistre
        # car on modélise le COÛT sachant qu'il y a un sinistre
        df_sin = self.df[self.df["nb_sinistres"] > 0].copy()

        # Coût moyen par sinistre = coût total / nb sinistres
        df_sin["cout_moyen"] = (df_sin["cout_total"]
                                / df_sin["nb_sinistres"])

        self.cout = smf.glm(
            formula = ("cout_moyen ~ age + bonus_malus "
                       "+ C(zone) + C(categorie_vh)"),
            data    = df_sin,
            family  = sm.families.Gamma(
                link = sm.families.links.Log()
            )
        ).fit()

        print(f"AIC Coût : {self.cout.aic:.2f}")
        return self.cout

    def calculer_primes(self):
        """
        Calcule la prime pure pour chaque contrat.

        Prime pure = Fréquence prédite × Coût moyen prédit
                   = E[N] × E[S|N>0]

        .predict() applique le modèle ajusté aux données
        et retourne les valeurs prédites.
        """
        # On ajuste les modèles s'ils ne l'ont pas encore été
        if self.freq is None:
            self.ajuster_frequence()
        if self.cout is None:
            self.ajuster_cout()

        # .predict() = applique le modèle sur le DataFrame
        self.df["freq_pred"]  = self.freq.predict(self.df)
        self.df["cout_pred"]  = self.cout.predict(self.df)

        # Prime pure = produit des deux prédictions
        self.df["prime_pure"] = (self.df["freq_pred"]
                                 * self.df["cout_pred"])

        # On retourne seulement les colonnes utiles
        return self.df[["age", "zone", "bonus_malus",
                        "freq_pred", "cout_pred", "prime_pure"]]


# ──────────────────────────────────────────────────────────────
class ChainLadder:
    """
    Méthode Chain-Ladder pour estimer les provisions IBNR.

    IBNR = Incurred But Not Reported
         = sinistres survenus mais pas encore déclarés/réglés

    On utilise un triangle de développement :
    ┌─────┬──────┬──────┬──────┬──────┐
    │ Ann │ Dev1 │ Dev2 │ Dev3 │ Dev4 │
    ├─────┼──────┼──────┼──────┼──────┤
    │2020 │ 1000 │ 1500 │ 1700 │ 1750 │  ← données complètes
    │2021 │ 1100 │ 1650 │ 1870 │  ???  │  ← à estimer
    │2022 │ 1200 │ 1800 │  ???  │  ???  │  ← à estimer
    │2023 │ 1300 │  ???  │  ???  │  ???  │  ← à estimer
    └─────┴──────┴──────┴──────┴──────┘
    np.nan = valeur manquante (case à estimer)
    """

    def __init__(self, triangle: np.ndarray):
        """
        triangle : matrice numpy (array 2D) avec np.nan pour les cases vides
        np.ndarray = type de tableau numpy
        """
        # .astype(float) = convertit en nombres décimaux
        # nécessaire pour que np.nan fonctionne (nan est un float)
        self.triangle         = triangle.astype(float)
        self.n                = triangle.shape[0]  # nb lignes = nb années
        self.facteurs         = None
        self.triangle_complet = None

    def calculer_facteurs(self):
        """
        Calcule les facteurs de développement (Link Ratios).

        f_j = Σ colonne(j+1) / Σ colonne(j)
             = ratio moyen de progression entre deux colonnes

        Si f2 = 1.5, ça signifie que les sinistres augmentent
        en moyenne de 50% entre la 1ère et 2ème année de développement.
        """
        n        = self.n
        facteurs = []   # liste vide qu'on va remplir

        for j in range(n - 1):   # pour chaque colonne (sauf la dernière)
            # ~np.isnan() = "n'est pas NaN" = la case a une valeur
            # & = ET logique : les deux cases (j et j+1) doivent exister
            lignes_ok = (
                ~np.isnan(self.triangle[:n-j-1, j]) &
                ~np.isnan(self.triangle[:n-j-1, j+1])
            )

            # Somme des valeurs disponibles en colonne j+1 et j
            numerateur   = self.triangle[:n-j-1, j+1][lignes_ok].sum()
            denominateur = self.triangle[:n-j-1, j  ][lignes_ok].sum()

            # Facteur = ratio entre les deux colonnes
            facteurs.append(numerateur / denominateur)

        self.facteurs = facteurs
        print("Facteurs de développement :")
        for i, f in enumerate(facteurs):
            print(f"  f{i+1} = {f:.4f}")
        return facteurs

    def completer_triangle(self):
        """
        Complète le triangle en projetant les cases manquantes.
        Chaque case vide = case précédente × facteur de développement
        """
        if self.facteurs is None:
            self.calculer_facteurs()

        # .copy() = on travaille sur une copie pour ne pas écraser l'original
        tri = self.triangle.copy()

        # i = ligne (année de survenance), j = colonne (développement)
        for i in range(1, self.n):
            for j in range(self.n - i, self.n):
                # np.isnan() = True si la case est vide
                if np.isnan(tri[i, j]):
                    # Projection : case vide = case précédente × facteur
                    tri[i, j] = tri[i, j-1] * self.facteurs[j-1]

        self.triangle_complet = tri
        return tri

    def calculer_ibnr(self):
        """
        Calcule l'IBNR pour chaque année de survenance.

        IBNR(année i) = Ultime projeté - Dernier montant connu
                      = triangle_complet[i, -1] - triangle[i, dernier_connu]

        Retourne un DataFrame avec le détail par année.
        """
        if self.triangle_complet is None:
            self.completer_triangle()

        resultats = []

        for i in range(self.n):
            # Récupère la ligne i (une année de survenance)
            ligne  = self.triangle[i, :]

            # ~np.isnan() = filtre les valeurs non-manquantes
            connus = ligne[~np.isnan(ligne)]

            # [-1] = dernier élément du tableau (le plus récent connu)
            dernier_connu = connus[-1] if len(connus) > 0 else 0

            # Ultime = valeur finale projetée (dernière colonne)
            ultime = self.triangle_complet[i, -1]

            # IBNR = ce qu'il reste à payer
            ibnr   = ultime - dernier_connu

            # On ajoute un dict (ligne) à la liste de résultats
            resultats.append({
                "annee_surv":    2020 + i,
                "dernier_connu": round(dernier_connu),
                "ultime":        round(ultime),
                "IBNR":          round(ibnr),
                # Pourcentage de développement atteint
                "pct_dev": f"{dernier_connu / ultime * 100:.1f}%",
            })

        # Crée le DataFrame final depuis la liste de dicts
        df_ibnr = pd.DataFrame(resultats)

        print("\n── Triangle IBNR (Chain-Ladder) ──")
        print(df_ibnr.to_string(index=False))
        print(f"\nProvision IBNR TOTALE : {df_ibnr['IBNR'].sum():>12,.0f} €")

        return df_ibnr
