# ══════════════════════════════════════════════════════════════
#  modules/vie.py — ASSURANCE VIE
#
#  Ce fichier contient 2 classes :
#   1. TableMortalite → gère les probabilités de décès par âge
#   2. AssuranceVie   → calcule primes, provisions, annuités
# ══════════════════════════════════════════════════════════════

# On importe les librairies dont on a besoin
# numpy  = calculs mathématiques sur des tableaux de nombres
# pandas = manipulation de tableaux de données (comme Excel)
# scipy  = distributions statistiques (loi normale, etc.)
# matplotlib = création de graphiques
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────
# CLASS = un modèle pour créer des objets
# Pense à une classe comme un "moule" :
#   - TableMortalite est le moule
#   - table = TableMortalite() crée un objet concret depuis ce moule
# ──────────────────────────────────────────────────────────────
class TableMortalite:
    """
    Représente une table de mortalité actuarielle.
    Contient pour chaque âge x :
      - qx = probabilité de décéder dans l'année
      - px = probabilité de survivre dans l'année (px = 1 - qx)
    """

    def __init__(self):
        # __init__ = constructeur : s'exécute automatiquement
        # quand on écrit : table = TableMortalite()

        # np.arange(0, 121) crée un tableau [0, 1, 2, ..., 120]
        # = tous les âges de 0 à 120 ans
        ages = np.arange(0, 121)

        # Loi de Makeham : modèle mathématique de la mortalité
        # log(qx) augmente avec l'âge → risque croissant
        # a, b, c sont des paramètres calibrés sur des tables réelles
        a, b, c = -7.5, -10, 1.10

        # np.exp() = exponentielle (e^x)
        # np.minimum(1.0, ...) = on s'assure que qx ne dépasse pas 1
        qx = np.minimum(1.0, np.exp(a) + np.exp(b) * (c ** ages))

        # pd.DataFrame() crée un tableau avec colonnes nommées
        # self.table = on stocke le tableau dans l'objet
        self.table = pd.DataFrame({
            "age": ages,
            "qx":  qx,        # probabilité de décès
            "px":  1 - qx     # probabilité de survie
        })

    def get_qx(self, age):
        """
        Retourne qx (proba de décès) pour un âge donné.
        Exemple : get_qx(45) → 0.00312
        """
        # .loc[] = accès aux lignes qui vérifient une condition
        # self.table["age"] == age → filtre la ligne de l'âge voulu
        ligne = self.table.loc[self.table["age"] == age, "qx"]

        # .values[0] = prend la valeur (pas l'index pandas)
        # Si l'âge n'existe pas, retourne 0
        return ligne.values[0] if len(ligne) > 0 else 0.0

    def probabilite_survie(self, age_depart, t):
        """
        Calcule tpx = probabilité de survivre t années depuis age_depart.

        Formule : tpx = px * p(x+1) * p(x+2) * ... * p(x+t-1)
        On multiplie les probabilités de survie année par année.

        Exemple : probabilite_survie(40, 5)
        = P(vivre de 40 à 45 ans)
        = p40 × p41 × p42 × p43 × p44
        """
        # Sélectionne les px pour les âges de age_depart à age_depart+t-1
        # & = ET logique entre deux conditions
        px_values = self.table.loc[
            (self.table["age"] >= age_depart) &
            (self.table["age"] <  age_depart + t),
            "px"
        ].values  # .values = convertit en tableau numpy

        # np.prod() = multiplie tous les éléments ensemble
        # [0.99, 0.98, 0.97] → 0.99 × 0.98 × 0.97 = 0.941
        return np.prod(px_values)

    def esperance_vie(self, age):
        """
        Calcule l'espérance de vie résiduelle ex.
        ex = nombre moyen d'années restant à vivre depuis l'âge x.

        Formule : ex = Σ(t=1 à ω) tpx
        = somme des probabilités de vivre encore t années
        """
        # sum() avec un générateur (boucle condensée)
        # range(1, 120-age) = t prend les valeurs 1, 2, ..., 120-age
        return sum(
            self.probabilite_survie(age, t)
            for t in range(1, 120 - age)
        )

    def afficher_table(self, age_min=30, age_max=80):
        """
        Retourne un extrait de la table pour les âges min à max.
        Utile pour visualiser une tranche d'âges.
        """
        # Masque booléen : True pour les lignes dans l'intervalle
        masque = (self.table["age"] >= age_min) & \
                 (self.table["age"] <= age_max)
        # .round(6) = arrondi à 6 décimales
        return self.table[masque].round(6)


# ──────────────────────────────────────────────────────────────
class AssuranceVie:
    """
    Calcule les grandeurs actuarielles en assurance vie :
      - Annuités viagères
      - Assurances décès temporaires
      - Primes nettes annuelles
      - Provisions mathématiques

    Utilise une table de mortalité et un taux technique.
    """

    def __init__(self, table: TableMortalite, taux_tech: float = 0.02):
        """
        table     : objet TableMortalite (les qx)
        taux_tech : taux d'actualisation (ex: 0.02 = 2%)

        Le taux technique sert à actualiser les flux futurs :
        1€ reçu dans t ans vaut aujourd'hui v^t = (1/(1+i))^t
        """
        self.table     = table
        self.taux_tech = taux_tech
        # v = facteur d'actualisation annuel
        # Ex: taux=2% → v = 1/1.02 ≈ 0.9804
        self.v         = 1 / (1 + taux_tech)

    def _tpx(self, x, t):
        """Raccourci : probabilité de survie t ans depuis x."""
        # Le _ au début = méthode privée (usage interne)
        return self.table.probabilite_survie(x, t)

    def _qx(self, x):
        """Raccourci : probabilité de décès à l'âge x."""
        return self.table.get_qx(x)

    def annuite_viagere(self, age, duree_max=100):
        """
        Calcule äx = valeur actuelle d'une rente viagère de 1€/an.
        C'est le prix aujourd'hui d'une rente versée tant que
        l'assuré est vivant.

        Formule : äx = Σ(t=0 à ω-x) [ tpx × v^t ]

        Exemple : äx = 15 signifie qu'une rente de 1€/an à vie
        vaut 15€ aujourd'hui pour un assuré d'âge x.
        """
        v     = self.v
        total = 0.0

        # Boucle sur chaque année future t = 0, 1, 2, ...
        for t in range(duree_max - age):
            tpx    = self._tpx(age, t)   # prob. d'être vivant à t
            total += tpx * (v ** t)       # on actualise : × v^t
        return total

    def assurance_deces_temporaire(self, age, duree):
        """
        Calcule Ax:n = valeur actuelle des prestations décès.
        = prix aujourd'hui de 1€ payé au décès si décès avant n ans.

        Formule : Ax:n = Σ(t=0 à n-1) [ tpx × qx+t × v^(t+1) ]

        Décomposition :
          tpx     = prob. de survivre jusqu'à t
          qx+t    = prob. de mourir dans l'année t
          v^(t+1) = actualisation au moment du paiement
        """
        v     = self.v
        total = 0.0

        for t in range(duree):
            tpx   = self._tpx(age, t)        # survie jusqu'à t
            qx_t  = self._qx(age + t)        # décès dans l'année t
            # On paie le capital en fin d'année t+1 → v^(t+1)
            total += tpx * qx_t * (v ** (t + 1))
        return total

    def prime_nette(self, age, duree, capital):
        """
        Calcule la prime annuelle nette P.

        Principe d'équivalence actuarielle :
        PV(primes) = PV(prestations)
        P × äx:n  = Capital × Ax:n
        P          = Capital × Ax:n / äx:n

        La prime nette = ce que l'assuré doit payer chaque année
        pour que l'assureur soit à l'équilibre (sans marge).
        """
        # Valeur actuelle des prestations (numérateur)
        Ax_n = self.assurance_deces_temporaire(age, duree)

        # Valeur actuelle des primes (dénominateur)
        # = annuité sur la durée du contrat (pas viagère)
        ax_n = sum(
            self._tpx(age, t) * (self.v ** t)
            for t in range(duree)
        )

        # Si ax_n = 0 (impossible de payer des primes), retourne 0
        if ax_n == 0:
            return 0

        return (capital * Ax_n) / ax_n

    def provision_mathematique(self, age, duree, capital):
        """
        Calcule la provision mathématique à chaque date t.

        Méthode prospective :
        Vm(t) = PV(prestations futures) - PV(primes futures)
              = Capital × Ax+t:n-t - P × äx+t:n-t

        La provision = ce que l'assureur doit mettre de côté
        pour faire face aux engagements futurs.

        Retourne un DataFrame avec une ligne par année.
        """
        # On calcule d'abord la prime constante sur tout le contrat
        prime = self.prime_nette(age, duree, capital)

        # Liste vide qu'on va remplir dans la boucle
        provisions = []

        # t = nombre d'années écoulées (0 = début du contrat)
        for t in range(duree + 1):
            age_t  = age + t       # âge actuel de l'assuré
            duree_r = duree - t    # durée restante

            if duree_r <= 0:
                # En fin de contrat, la provision tombe à 0
                vm = 0.0
            else:
                # PV prestations restantes
                Ax = self.assurance_deces_temporaire(age_t, duree_r)
                # PV primes restantes
                ax = sum(
                    self._tpx(age_t, k) * (self.v ** k)
                    for k in range(duree_r)
                )
                # Provision = PV sorties - PV entrées
                vm = capital * Ax - prime * ax

            # On ajoute un dict (ligne) à la liste
            provisions.append({
                "annee":     t,
                "age":       age_t,
                "provision": round(vm, 2),
                "prime":     round(prime, 2),
            })

        # pd.DataFrame() transforme la liste de dicts en tableau
        return pd.DataFrame(provisions)
