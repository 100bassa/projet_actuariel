"""
═══════════════════════════════════════════════════════════════════════════════
main.py — Point d'entrée en ligne de commande du projet actuariel
═══════════════════════════════════════════════════════════════════════════════

RÔLE DE CE FICHIER
──────────────────
Ce fichier est le script principal du projet. Il exécute une démonstration
complète de tous les modules actuariels dans le terminal, sans interface
graphique. C'est le moyen le plus rapide de vérifier que tout fonctionne.

POURQUOI CE FICHIER EXISTE-T-IL ?
───────────────────────────────────
Le dashboard Streamlit (dashboard.py) est l'interface principale pour les
utilisateurs. Mais main.py sert à :
  1. Tester rapidement tous les modules en une seule commande
  2. Obtenir des résultats numériques précis dans le terminal
  3. Valider les calculs sans avoir besoin d'une interface graphique
  4. Servir de documentation exécutable : chaque section montre comment
     utiliser chaque module avec des paramètres réalistes

STRUCTURE DU PROJET
───────────────────
  main.py           → ce fichier : démo console de tous les modules
  dashboard.py      → interface web interactive (streamlit run dashboard.py)
  modules/
    vie.py          → assurance vie : mortalité, primes, provisions
    non_vie.py      → non-vie : portefeuille auto, tarification, Chain-Ladder
    solvabilite2.py → réglementation SII : SCR, BSCR, ratio de solvabilité
    ifrs17.py       → norme comptable IFRS 17 : BEL, RA, CSM, P&L
    alm.py          → gestion actif-passif : duration gap, immunisation, stress-test

UTILISATION
───────────
  python main.py
  # Puis pour le dashboard :
  streamlit run dashboard.py
"""

from modules.vie import TableMortalite, AssuranceVie
from modules.non_vie import PortefeuilleAuto, ChainLadder
from modules.solvabilite2 import SCR_NonVie, SCR_Vie, SolvabiliteII
from modules.ifrs17 import GroupeContrats
from modules.alm import Obligation, PortefeuilleActif, ProjectionPassif, AnalyseALM
import numpy as np


def main():
    """
    Exécute une démonstration complète des 5 modules actuariels.

    Rôle : point d'entrée unique qui instancie chaque module avec des
    paramètres réalistes, effectue les calculs et affiche les résultats.

    Pourquoi cette fonction plutôt que du code à la racine du fichier ?
    → Le bloc `if __name__ == "__main__":` en bas permet d'importer ce
      fichier sans exécuter les calculs (utile pour les tests unitaires).
    """
    print("=" * 60)
    print("   PROJET ACTUARIEL PYTHON — EXECUTION COMPLETE")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # MODULE VIE — modules/vie.py
    # Rôle : calcule les grandeurs actuarielles fondamentales de
    # l'assurance vie (prime nette, provision mathématique, annuité).
    # Pourquoi : permet de tarifer un contrat décès temporaire en
    # appliquant le principe d'équivalence actuarielle.
    # ══════════════════════════════════════════════════════════════
    print("\n[VIE]")

    # TableMortalite() : génère la table de mortalité avec la loi de Makeham
    # → probabilité de décès qx pour chaque âge de 0 à 120 ans
    table = TableMortalite()

    # AssuranceVie() : encapsule les formules de calcul actuariel
    # taux_tech=0.025 → taux technique de 2,5% pour actualiser les flux futurs
    av = AssuranceVie(table, taux_tech=0.025)

    # prime_nette() : applique le principe d'équivalence
    # PV(primes) = PV(capital décès)
    # → pour un assuré de 45 ans, contrat de 20 ans, capital 100 000€
    prime = av.prime_nette(age=45, duree=20, capital=100_000)
    print(f"Prime nette (45 ans, 20 ans, 100k€) : {prime:,.2f} €")

    # annuite_viagere() : valeur actualisée de 1€/an versé tant que l'assuré vit
    # → donne aussi l'espérance de vie résiduelle en années
    print(f"Espérance de vie résiduelle à 45 ans : {av.annuite_viagere(45):.1f} ans")

    # ══════════════════════════════════════════════════════════════
    # MODULE NON-VIE — modules/non_vie.py
    # Rôle : simule un portefeuille automobile et calcule les
    # provisions IBNR (sinistres survenus non déclarés) via
    # la méthode Chain-Ladder.
    # Pourquoi : les assureurs non-vie doivent provisionner les
    # sinistres en cours de développement pour respecter leurs
    # obligations comptables et réglementaires.
    # ══════════════════════════════════════════════════════════════
    print("\n[NON-VIE]")

    # PortefeuilleAuto() : génère 5 000 contrats fictifs avec loi de Poisson
    # (fréquence de sinistres) et loi log-normale (coût des sinistres)
    ptf = PortefeuilleAuto(n_contrats=5_000)
    ptf.statistiques()  # affiche nb contrats, fréquence moyenne, coût moyen

    # Triangle de développement des sinistres cumulés
    # Lignes = années de survenance (2020 à 2023)
    # Colonnes = années de développement (1 à 4)
    # np.nan = cases à estimer par la méthode Chain-Ladder
    triangle = np.array([
        [4_500_000, 6_750_000, 7_650_000, 7_875_000],   # 2020 : complet
        [4_950_000, 7_425_000, 8_415_000,       np.nan], # 2021 : col 4 manquante
        [5_400_000, 8_100_000,      np.nan,      np.nan], # 2022 : col 3-4 manquantes
        [5_850_000,      np.nan,    np.nan,      np.nan], # 2023 : col 2-4 manquantes
    ])

    # ChainLadder() : méthode standard de provisionnement IBNR
    # calculer_ibnr() : calcule les facteurs de développement et projette le triangle
    cl = ChainLadder(triangle)
    cl.calculer_ibnr()

    # ══════════════════════════════════════════════════════════════
    # MODULE SOLVABILITÉ II — modules/solvabilite2.py
    # Rôle : calcule le SCR (capital minimum réglementaire) et le
    # ratio de solvabilité selon la formule standard SII.
    # Pourquoi : la directive Solvabilité II oblige les assureurs
    # européens à détenir des fonds propres suffisants pour résister
    # à un choc avec 99,5% de probabilité sur 1 an. Sans ce capital,
    # l'autorité de contrôle (ACPR) peut intervenir.
    # ══════════════════════════════════════════════════════════════
    print("\n[SOLVABILITE II]")

    # SCR_NonVie() : calcule le SCR de souscription pour les branches
    # non-vie (Auto RC, Auto Dommages, Incendie) avec les facteurs
    # de volatilité réglementaires et la matrice de corrélation SII
    scr_nl = SCR_NonVie(
        {"Auto RC": 5e6, "Auto Dommages": 3e6, "Incendie": 2e6},     # primes
        {"Auto RC": 8e6, "Auto Dommages": 4e6, "Incendie": 3.5e6},   # provisions
    ).scr_total()

    # SCR_Vie() : calcule le SCR de souscription vie via des chocs
    # réglementaires sur mortalité (+15%), longévité (-20%), rachats, frais
    # calculer() retourne (SCR_total, détail_par_risque)
    scr_v, _ = SCR_Vie(15e6, 20e6, 5e6, 20e6, 100e6).calculer()

    # SolvabiliteII() : agrège les SCR modulaires (NL + Vie + Marché)
    # en BSCR puis en SCR final (avec ajustement fiscal)
    # rapport() : affiche le tableau de bord réglementaire complet
    sii = SolvabiliteII(scr_nl, scr_v, 3.5e6, 50e6)
    sii.rapport()

    # ══════════════════════════════════════════════════════════════
    # MODULE IFRS 17 — modules/ifrs17.py
    # Rôle : modélise le passif d'un groupe de contrats d'assurance
    # vie selon la norme comptable IFRS 17 (GMM).
    # Pourquoi : depuis le 1er janvier 2023, tous les assureurs
    # cotés en Europe appliquent IFRS 17. Cette norme change
    # radicalement la comptabilisation : les primes ne sont plus
    # des revenus directs — le profit est reconnu progressivement
    # via l'amortissement de la CSM.
    # ══════════════════════════════════════════════════════════════
    print("\n[IFRS 17]")

    # GroupeContrats() : représente un millésime de contrats soumis
    # au même traitement comptable IFRS 17
    gc = GroupeContrats({
        "nb_contrats":        1_000,
        "duree":              20,
        "prime_annuelle":     1_380,    # prime annuelle par contrat (€)
        "capital_deces":      100_000,  # capital garanti en cas de décès (€)
        "taux_actualisation": 0.025,    # courbe des taux sans risque EIOPA
        "taux_mortalite":     0.005,    # taux de mortalité annuel moyen
        "frais_gestion":      80,       # frais de gestion annuels par contrat (€)
        "frais_acquisition":  150,      # frais d'acquisition (commissionnement) (€)
        "cov_risque":         0.05,     # CoV = 5% → RA = 5% × |BEL| × 1.15
    })

    # bilan_ifrs17() : affiche le passif décomposé BEL + RA + CSM
    gc.bilan_ifrs17()

    # ══════════════════════════════════════════════════════════════
    # MODULE ALM — modules/alm.py
    # Rôle : analyse l'adossement actif-passif d'un assureur vie.
    # Mesure la sensibilité du surplus économique aux mouvements
    # de taux d'intérêt et vérifie les conditions d'immunisation.
    # Pourquoi : les assureurs vie investissent les primes dans des
    # obligations à long terme pour couvrir les engagements futurs.
    # Si la duration des actifs ≠ duration des passifs, une hausse
    # (ou baisse) des taux peut éroder le surplus économique. L'ALM
    # gère ce risque de taux structurel.
    # ══════════════════════════════════════════════════════════════
    print("\n[ALM — Asset-Liability Management]")

    # ── Portefeuille obligataire (3 lignes) ───────────────────────────────────
    # Trois obligations avec des maturités échelonnées (5, 10, 15 ans)
    # pour assurer une meilleure couverture des flux passifs dans le temps
    obligations = [
        Obligation(nominal=20_000_000, coupon=0.03, maturite=5,  taux_rendement=0.035, notation="AA",  type_="souveraine"),
        Obligation(nominal=30_000_000, coupon=0.04, maturite=10, taux_rendement=0.040, notation="BBB", type_="entreprise"),
        Obligation(nominal=20_000_000, coupon=0.025, maturite=15, taux_rendement=0.038, notation="A",  type_="souveraine"),
    ]

    # ── Portefeuille d'actifs complet ─────────────────────────────────────────
    # Obligations (taux fixe, sensibles aux taux) +
    # Actions (rendement élevé, duration ≈ 0) +
    # Immobilier (duration proxy ≈ 7 ans via les baux)
    actif = PortefeuilleActif(
        obligations          = obligations,
        valeur_actions       = 10_000_000,
        rendement_actions    = 0.07,
        valeur_immobilier    =  5_000_000,
        rendement_immobilier = 0.04,
        duration_immobilier  = 7.0,
    )

    # ── Passif vie (projection des flux sortants) ─────────────────────────────
    # Modélise les engagements de l'assureur : décès, rachats, échéances, frais
    # Le BEL = valeur actualisée des flux nets sortants
    passif = ProjectionPassif(
        age_moyen          = 45,
        capital_moyen      = 100_000,
        nb_contrats        = 1_000,
        duree              = 20,
        prime_annuelle     = 1_500,
        taux_actualisation = 0.035,   # taux sans risque EIOPA
        taux_rachat        = 0.03,    # 3% des assurés rachètent par an
        taux_frais         = 0.008,   # 0,8% des engagements en frais annuels
    )

    # ── Analyse ALM ───────────────────────────────────────────────────────────
    # AnalyseALM() combine actif et passif pour calculer :
    #   - Surplus économique = Actif - BEL
    #   - Duration Gap = D_A - (BEL/A) × D_L
    #   - DV01 = sensibilité du surplus à +1 bp de taux
    #   - Immunisation de Redington (3 conditions)
    alm = AnalyseALM(actif, passif)
    rapport = alm.rapport_immunisation()

    # Affichage des indicateurs principaux du bilan économique
    print(f"  Valeur actif  : {rapport['Valeur actif (EUR)']:>15,.0f} €")
    print(f"  BEL           : {rapport['BEL (EUR)']:>15,.0f} €")
    print(f"  Surplus       : {rapport['Surplus (EUR)']:>15,.0f} €")
    print(f"  Ratio couvert.: {rapport['Ratio couverture']:>15.4f}")
    print(f"  Duration actif: {rapport['Duration actif (ans)']:>15.2f} ans")
    print(f"  Duration pass.: {rapport['Duration passif (ans)']:>15.2f} ans")
    print(f"  Duration Gap  : {rapport['Duration Gap (ans)']:>15.2f} ans")
    print(f"  DV01          : {rapport['DV01 (EUR/bp)']:>15,.0f} €/bp")
    print(f"  Immunisation  : {'✓ OK' if rapport['Immunisation OK'] else '✗ à corriger'}")

    # ── Analyse de sensibilité (stress-test de taux) ──────────────────────────
    # tableau_scenarios() simule 7 chocs parallèles de taux : -200 à +200 bps
    # Permet de répondre à la question : "Que se passe-t-il si les taux montent de 1% ?"
    print("\n  Scénarios de taux (impact sur le surplus) :")
    df_sc = alm.tableau_scenarios()
    for _, row in df_sc.iterrows():
        bps = int(row["Choc (bps)"])
        ds  = row["Var. Surplus (EUR)"]
        pct = row["Variation surplus (%)"]
        print(f"    {bps:+5d} bps → ΔSurplus = {ds:>+12,.0f} € ({pct:+.1f}%)")

    print("\nCalculs terminés !")
    print("Lance le dashboard avec : streamlit run dashboard.py")


# ─────────────────────────────────────────────────────────────────────────────
# Point d'entrée Python standard
# ─────────────────────────────────────────────────────────────────────────────
# `if __name__ == "__main__":` vérifie que ce fichier est exécuté directement
# (python main.py) et non importé par un autre module.
# Sans cette protection, les calculs s'exécuteraient même lors d'un simple import.
if __name__ == "__main__":
    main()
