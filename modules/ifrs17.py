# ══════════════════════════════════════════════════════════════
#  modules/ifrs17.py — NORME COMPTABLE IFRS 17
#
#  RÔLE DE CE MODULE
#  ─────────────────
#  Ce module implémente la mesure du passif d'assurance selon la norme
#  comptable internationale IFRS 17, entrée en vigueur le 1er janvier 2023.
#  Il utilise l'approche GMM (General Measurement Model), qui est l'approche
#  standard pour les contrats d'assurance vie avec participation discrétionnaire.
#
#  POURQUOI IFRS 17 ?
#  ──────────────────
#  Avant IFRS 17, les assureurs utilisaient des méthodes comptables très
#  hétérogènes selon les pays. IFRS 17 uniformise la comptabilisation pour
#  tous les assureurs cotés en Europe, rendant les états financiers comparables.
#
#  CHANGEMENT FONDAMENTAL PAR RAPPORT À L'ANCIENNE NORME (IFRS 4) :
#    - Avant : les primes reçues = revenus de l'exercice
#    - Après : les primes NE SONT PAS des revenus → elles sont "décomposées"
#      Le résultat provient uniquement du SERVICE RENDU aux assurés
#      (= release annuelle de la CSM + release du RA)
#
#  STRUCTURE DU PASSIF IFRS 17 (approche GMM)
#  ────────────────────────────────────────────
#
#  Passif IFRS 17 = FCF + CSM
#  où FCF (Fulfilment Cash Flows) = BEL + RA
#
#  ┌─────────────────────────────────────────────────────────────┐
#  │  BEL  = Best Estimate Liabilities                          │
#  │       = Valeur actualisée des flux futurs attendus          │
#  │       = PV(prestations + frais + frais acq) − PV(primes)   │
#  │                                                             │
#  │  RA   = Risk Adjustment (Ajustement pour risque)            │
#  │       = Compensation pour l'incertitude non-financière      │
#  │       = Marge pour le risque que les flux réels dévient     │
#  │                                                             │
#  │  CSM  = Contractual Service Margin                          │
#  │       = Profit futur non encore reconnu (service non rendu)  │
#  │       = Calculé à l'émission : max(Primes − FCF, 0)         │
#  │       → Amorti progressivement en résultat au fil du temps  │
#  └─────────────────────────────────────────────────────────────┘
#
#  CE FICHIER CONTIENT 1 CLASSE
#  ─────────────────────────────
#   GroupeContrats → implémente le GMM pour un portefeuille de contrats vie
# ══════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# CLASSE — GROUPE DE CONTRATS IFRS 17
# ──────────────────────────────────────────────────────────────
# Rôle : représente un groupe homogène de contrats d'assurance
# soumis au même traitement comptable IFRS 17 (même millésime,
# même portefeuille, même niveau de rentabilité).
#
# Pourquoi un "groupe" et pas un contrat individuel ?
# → IFRS 17 impose de regrouper les contrats par :
#    1. Portefeuille (même risque + gestion commune)
#    2. Année de souscription (même cohorte annuelle)
#    3. Profitabilité (onéreux / neutre / rentable)
# → On mesure et on amortit la CSM au niveau du groupe, pas du contrat.
# ──────────────────────────────────────────────────────────────

class GroupeContrats:
    """
    Représente un groupe de contrats IFRS 17.

    Selon IFRS 17, les contrats sont regroupés par :
      - Portefeuille (même risque + gestion commune)
      - Année de souscription (cohorte annuelle)
      - Profitabilité (onéreux / rentable / neutre)

    Cette classe implémente le GMM (General Measurement Model).
    """

    def __init__(self, params: dict):
        """
        Rôle : initialise le groupe de contrats avec toutes ses hypothèses.

        Pourquoi passer un dict plutôt que 10 paramètres séparés ?
        → Plus lisible à l'appel : on voit le nom de chaque paramètre
        → Facilement extensible : ajouter un paramètre ne casse pas les appels existants
        → Validation centralisée dans _valider_params()

        params : dictionnaire contenant toutes les hypothèses actuarielles.
        Clés obligatoires : nb_contrats, duree, prime_annuelle, capital_deces,
                            taux_actualisation, taux_mortalite, frais_gestion,
                            frais_acquisition, cov_risque.
        """
        self.p = params   # raccourci : self.p["cle"] au lieu de params["cle"]
        self._valider_params()

        # Facteur d'actualisation annuel : v = 1 / (1 + taux)
        # Exemple : taux = 2.5% → v = 1/1.025 ≈ 0.9756
        # Utilisé pour ramener les flux futurs en valeur d'aujourd'hui
        self.v = 1 / (1 + params["taux_actualisation"])
        self.n = params["nb_contrats"]  # nb contrats dans le groupe
        self.T = params["duree"]        # durée du contrat (ans)

        # Lazy evaluation (évaluation à la demande) :
        # Les calculs coûteux ne sont faits qu'une seule fois, la première fois
        # qu'on en a besoin. Les attributs sont None tant que le calcul n'a pas eu lieu.
        self._flux = None   # flux de trésorerie projetés
        self._bel  = None   # Best Estimate Liability
        self._ra   = None   # Risk Adjustment
        self._csm  = None   # Contractual Service Margin initiale

    def _valider_params(self):
        """
        Vérifie que tous les paramètres obligatoires sont présents.

        Rôle : "fail fast" — on détecte l'erreur dès la construction de l'objet
        plutôt qu'en plein milieu d'un calcul (ce qui serait plus difficile à débugger).

        raise ValueError : déclenche une exception Python et arrête l'exécution
        avec un message d'erreur explicite indiquant quel paramètre est absent.
        """
        requis = [
            "nb_contrats", "duree", "prime_annuelle",
            "capital_deces", "taux_actualisation",
            "taux_mortalite", "frais_gestion",
            "frais_acquisition", "cov_risque",
        ]
        for param in requis:
            if param not in self.p:
                raise ValueError(f"Paramètre manquant : {param}")

    def flux_tresorerie(self):
        """
        Projette les flux de trésorerie futurs du groupe de contrats.

        Rôle : génère, année par année, tous les flux entrants (primes)
        et sortants (prestations décès + frais) pendant la durée du contrat.
        C'est la base du calcul du BEL.

        Flux entrants  (+) : primes versées par les assurés vivants
        Flux sortants  (-) : capital décès versé aux bénéficiaires
                             frais de gestion annuels
                             frais d'acquisition (payés dès la souscription)

        Algorithme (boucle for sur T années) :
          t = 1, 2, ..., T
          deces_t     = nb_survivants_{t-1} × taux_mortalite
          primes_t    = nb_survivants_{t-1} × prime_annuelle
          prestations_t = deces_t × capital_deces
          frais_t     = nb_survivants_{t-1} × frais_gestion
          flux_net_t  = primes_t - prestations_t - frais_t
          flux_act_t  = flux_net_t × v^t   (actualisation)

        Pattern cache :
          if self._flux is not None: return self._flux
          → si déjà calculé, retourne directement le résultat
          → évite de recalculer les flux chaque fois que best_estimate()
            ou risk_adjustment() sont appelés
        """
        if self._flux is not None:  # cache hit : résultat déjà calculé
            return self._flux

        p          = self.p            # raccourci local
        survivants = float(self.n)     # nombre de survivants (commence à nb_contrats)
        records    = []                # liste des flux annuels (liste de dicts)

        for t in range(1, self.T + 1):
            # Nombre de décès dans l'année t = survivants × taux mortalité
            # Note : modèle simple à taux constant (pas de table par âge ici)
            deces = survivants * p["taux_mortalite"]

            flux = {
                "annee":       t,
                "survivants":  survivants,
                "deces":       deces,
                # Primes : versées par les assurés encore en vie (= survivants)
                "primes":      survivants * p["prime_annuelle"],
                # Prestations : capital décès versé aux bénéficiaires des décédés
                "prestations": deces * p["capital_deces"],
                # Frais : coûts de gestion annuels pour les assurés vivants
                "frais":       survivants * p["frais_gestion"],
            }

            # Flux net = entrées - sorties (positif si l'assureur encaisse plus)
            flux["flux_net"] = (flux["primes"]
                                - flux["prestations"]
                                - flux["frais"])

            # Flux actualisé = flux net ramené en valeur d'aujourd'hui
            # v^t = 1 / (1 + taux)^t = facteur d'actualisation à la date t
            flux["flux_actualise"] = flux["flux_net"] * (self.v ** t)

            records.append(flux)

            # Les décédés de l'année t quittent le groupe → nb survivants diminue
            survivants -= deces

        # Conversion liste de dicts → DataFrame pandas et mise en cache
        self._flux = pd.DataFrame(records)
        return self._flux

    def best_estimate(self):
        """
        Calcule le BEL (Best Estimate Liabilities).

        Rôle : mesure la valeur actuelle "objective" des engagements de l'assureur.
        C'est le premier composant du passif IFRS 17.

        Formule :
          BEL = PV(prestations) + PV(frais) + frais_acq − PV(primes)
              = sorties actualisées − entrées actualisées

        Interprétation :
          BEL > 0 → passif net : l'assureur doit plus qu'il ne reçoit (cas habituel)
          BEL < 0 → actif net  : rare, contrats très profitables (généreront une CSM élevée)

        Calcul vectorisé avec numpy :
          df["colonne"].values → tableau numpy (plus rapide qu'une boucle Python)
          v_arr = [v^1, v^2, ..., v^T] → tableau des facteurs d'actualisation
          (colonne × v_arr).sum() → produit scalaire = valeur actualisée totale

        Pattern cache : retourne self._bel si déjà calculé.
        """
        if self._bel is not None:   # cache hit
            return self._bel

        df = self.flux_tresorerie()

        # t_arr = [1, 2, 3, ..., T] (tableau des années)
        t_arr = df["annee"].values
        # v_arr = [v^1, v^2, ..., v^T] (tableau des facteurs d'actualisation)
        v_arr = self.v ** t_arr

        # Valeur actuelle de chaque composante (multiplication élément par élément)
        pv_presta = (df["prestations"].values * v_arr).sum()
        pv_frais  = (df["frais"].values        * v_arr).sum()
        pv_primes = (df["primes"].values        * v_arr).sum()

        # Frais d'acquisition = coûts fixes payés dès la souscription (non actualisés)
        # Ces coûts sont payés immédiatement (t=0) donc pas d'actualisation
        frais_acq = self.n * self.p["frais_acquisition"]

        # BEL = toutes les sorties actualisées − toutes les entrées actualisées
        self._bel = pv_presta + pv_frais + frais_acq - pv_primes
        return self._bel

    def risk_adjustment(self):
        """
        Calcule le RA (Risk Adjustment / Ajustement pour risque).

        Rôle : deuxième composant du passif IFRS 17.
        Le RA répond à la question :
        "De combien faut-il majorer le BEL pour tenir compte du fait
         que les flux réels pourraient dévier des flux attendus ?"

        Contexte :
          Le BEL = meilleure estimation (esperance mathématique des flux).
          Mais les flux réels peuvent être supérieurs (plus de décès, plus de rachats...).
          Le RA est la prime de risque pour cette incertitude.

        Méthode CoV (Coefficient of Variation) :
          RA = |BEL| × CoV × facteur_percentile
          Le facteur 1.15 correspond approximativement au percentile 75%
          (hypothèse simplifiée, en pratique calculé par simulation ou formule analytique)

        Exemple : BEL = 100 000€, CoV = 5% → RA = 100 000 × 0.05 × 1.15 = 5 750€

        abs() = valeur absolue : on ignore le signe du BEL pour le calcul du RA.
        """
        if self._ra is not None:  # cache hit
            return self._ra

        self._ra = abs(self.best_estimate()) * self.p["cov_risque"] * 1.15
        return self._ra

    def csm_initial(self):
        """
        Calcule la CSM (Contractual Service Margin) à la date d'émission.

        Rôle : troisième composant du passif IFRS 17.
        La CSM = profit futur contractuel non encore gagné par l'assureur.
        C'est le "bénéfice en attente de reconnaissance" stocké au passif.

        Formule :
          CSM = Primes_initiales − FCF
              = Primes − (BEL + RA)
              = Ce qui reste après avoir couvert les engagements futurs

        Règle IFRS 17 fondamentale :
          Si CSM > 0 → contrats profitables → on stocke la marge en CSM
                       elle sera relâchée progressivement en résultat
          Si CSM < 0 → contrats ONÉREUX (loss-making contracts)
                       → la perte est reconnue immédiatement en résultat
                       → CSM est fixée à 0 (elle ne peut pas être négative)

        max(valeur, 0) = empêche la CSM d'être négative (règle IFRS 17 art. 38).

        FCF = Fulfilment Cash Flows = BEL + RA = coût total estimé des engagements.
        """
        if self._csm is not None:  # cache hit
            return self._csm

        # Primes initiales = total des primes encaissées au départ
        # (dans ce modèle simplifié = nb_contrats × prime_annuelle)
        primes_init = self.n * self.p["prime_annuelle"]

        # FCF = Fulfilment Cash Flows = coût actuariel complet des engagements
        fcf = self.best_estimate() + self.risk_adjustment()

        # CSM = excédent de primes sur les FCF (= profit futur à étaler)
        # Si primes < FCF → contrat onéreux → CSM = 0 et perte reconnue immédiatement
        self._csm = max(primes_init - fcf, 0)
        return self._csm

    def amortissement_csm(self):
        """
        Calcule la release (amortissement) annuelle de la CSM.

        Rôle : détermine combien de profit est reconnu en résultat chaque année.
        Sous IFRS 17, la CSM est relâchée au rythme du service rendu.

        Principe IFRS 17 :
          La CSM est amortie proportionnellement aux "unités de couverture",
          qui représentent le service rendu aux assurés chaque année.
          Unité de couverture t = nb_survivants_t × v^t

        Algorithme :
          1. Calculer les unités de couverture pour chaque année
          2. Calculer leur total sur toute la durée
          3. Taux d'amortissement_t = unité_t / total
          4. Release_t = CSM_initial × taux_t

        Interprétation du résultat :
          Si la CSM initiale est 500 000€ sur 20 ans :
          - Année 1 : release élevée (beaucoup d'assurés vivants)
          - Dernières années : release faible (peu de survivants)

        np.cumsum([a, b, c]) → [a, a+b, a+b+c] (somme cumulée)
        np.round() = arrondi sur un tableau entier numpy
        """
        df  = self.flux_tresorerie()
        csm = self.csm_initial()

        # Unités de couverture = nb survivants × facteur d'actualisation
        # Cela tient compte du fait que les survivants futurs comptent moins
        # (à cause de l'actualisation temporelle)
        t_arr  = df["annee"].values
        unites = df["survivants"].values * (self.v ** t_arr)
        total  = unites.sum()   # somme totale = dénominateur du taux d'amortissement

        # Taux d'amortissement annuel : part de chaque année dans le total
        # taux[0] = % de la CSM reconnu en résultat à l'année 1
        taux = unites / total

        # Montants de CSM relâchés chaque année
        # Multiplication scalaire × tableau numpy = multiplication élément par élément
        releases = csm * taux

        # CSM restante en fin d'année = CSM initial − somme des releases passées
        csm_cum = csm - np.cumsum(releases)

        return pd.DataFrame({
            "annee":        t_arr,
            "unites":       np.round(unites, 1),
            "taux_release": np.round(taux * 100, 2),   # en % pour la lisibilité
            "release_CSM":  np.round(releases, 2),
            "CSM_restant":  np.round(csm_cum, 2),
        })

    def compte_resultat(self):
        """
        Construit le compte de résultat IFRS 17 sur les 10 premières années.

        Rôle : modélise la reconnaissance des profits en résultat selon IFRS 17.

        Structure du P&L IFRS 17 (différent d'IFRS 4 / ancien référentiel) :
        ──────────────────────────────────────────────────────────────────────
        + Revenus d'assurance
            = charges attendues (services rendus)
            + Release CSM (profit de la période)
            + Release RA (incertitude levée)

        − Charges d'assurance
            = prestations réelles
            + frais réels

        ──────────────────────────────────────────────────────────────────────
        = Résultat technique (= Release CSM + Release RA)
            (= marges qui se révèlent année par année)

        + Produits financiers sur actifs (rendement du portefeuille)
        ──────────────────────────────────────────────────────────────────────
        = Résultat net

        Point fondamental IFRS 17 :
          Les primes reçues des assurés n'apparaissent PAS comme revenus.
          Seul le service rendu (marges libérées) est un revenu.
          Cela rend le résultat plus stable et prévisible que l'ancien modèle.

        min(10, self.T) : on affiche au maximum 10 ans (ou la durée du contrat
        si elle est inférieure à 10 ans).
        """
        df_flux  = self.flux_tresorerie()
        df_amort = self.amortissement_csm()
        bel      = self.best_estimate()
        ra       = self.risk_adjustment()

        records = []

        for i in range(min(10, self.T)):
            t = i + 1   # année 1 à 10

            # Release CSM de l'année i (depuis le tableau d'amortissement)
            # .loc[i, "colonne"] = accès à la ligne i (index 0-based)
            rel_csm = df_amort.loc[i, "release_CSM"]

            # Release RA : simplification linéaire = RA total / durée
            # (en pratique, le RA est recalculé chaque année selon l'incertitude restante)
            rel_ra  = ra / self.T

            # Charges réelles de l'année = prestations + frais
            charge = (df_flux.loc[i, "prestations"] + df_flux.loc[i, "frais"])

            # Revenus assurance = charges attendues + marges relâchées
            # (les "charges attendues" représentent la partie du service liée aux sinistres)
            rev_ass = charge + rel_csm + rel_ra

            # Résultat technique = marges de service reconnues cette année
            # (= Release CSM + Release RA = nouveau profit comptabilisé)
            res_tech = rel_csm + rel_ra

            # Produits financiers = rendement des actifs investis
            # Simplification : on applique le taux d'actualisation au BEL
            # (comme si le BEL était l'actif investi au taux sans risque)
            prod_fi = abs(bel) * self.p["taux_actualisation"]

            records.append({
                "Année":               t,
                "Revenus assurance":   round(rev_ass),
                "Charges assurance":   round(charge),
                "Release CSM":         round(rel_csm),
                "Release RA":          round(rel_ra),
                "Résultat technique":  round(res_tech),
                "Produits financiers": round(prod_fi),
                "Résultat net":        round(res_tech + prod_fi),
            })

        return pd.DataFrame(records)

    def bilan_ifrs17(self):
        """
        Affiche le passif IFRS 17 dans le terminal (bilan simplifié).

        Rôle : point d'entrée pour la démo console (appelée depuis main.py).
        Affiche la décomposition BEL + RA + CSM = Passif Total.

        Retourne un dict pour permettre l'utilisation dans d'autres contextes.

        Formatage Python utilisé :
          :>12,.0f = aligné à droite (>), largeur 12, séparateur milliers (,), 0 décimales
          ^50      = centré sur 50 caractères
        """
        bel = self.best_estimate()
        ra  = self.risk_adjustment()
        csm = self.csm_initial()

        sep  = "=" * 50
        dash = "-" * 38

        print(f"\n{sep}")
        print(f"{'PASSIF IFRS 17 — GMM':^50}")
        print(f"{sep}")
        print(f"  Best Estimate (BEL)  : {bel:>12,.0f} €")
        print(f"  Risk Adjustment (RA) : {ra:>12,.0f} €")
        print(f"  CSM                  : {csm:>12,.0f} €")
        print(f"  {dash}")
        print(f"  TOTAL PASSIF         : {bel+ra+csm:>12,.0f} €")
        print(f"{sep}")

        # Retourne un dict avec les montants pour utilisation dans dashboard.py
        return {
            "BEL":   bel,
            "RA":    ra,
            "CSM":   csm,
            "Total": bel + ra + csm
        }
