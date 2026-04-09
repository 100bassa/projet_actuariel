"""
═══════════════════════════════════════════════════════════════════════════════
dashboard.py — Interface graphique interactive du projet actuariel
═══════════════════════════════════════════════════════════════════════════════

RÔLE DE CE FICHIER
──────────────────
Ce fichier est le point d'entrée de l'interface utilisateur du projet.
Il construit un dashboard web interactif avec Streamlit, permettant d'explorer
tous les résultats actuariels via des sliders, des métriques et des graphiques
sans avoir à toucher au code Python.

POURQUOI UN DASHBOARD ?
────────────────────────
En actuariat, les résultats dépendent fortement des hypothèses choisies
(taux technique, taux de mortalité, primes, etc.).
Le dashboard permet de :
  1. Modifier les hypothèses en temps réel et voir l'impact immédiat
  2. Présenter les résultats aux décideurs (direction, régulateurs)
  3. Réaliser des analyses de sensibilité interactives

STRUCTURE DU DASHBOARD
───────────────────────
Le dashboard est organisé en 5 onglets (tabs), un par module :
  Tab 1 — Vie         : table de mortalité, primes, provisions mathématiques
  Tab 2 — Non-Vie     : portefeuille auto, sinistres, Chain-Ladder IBNR
  Tab 3 — Solvabilité II : SCR, ratio de solvabilité, waterfall BSCR
  Tab 4 — IFRS 17     : passif GMM (BEL + RA + CSM), compte de résultat
  Tab 5 — ALM         : duration gap, bilan économique, scénarios de taux

LANCEMENT
─────────
  streamlit run dashboard.py

DÉPENDANCES
───────────
  streamlit : framework web pour apps data en Python (créer l'UI)
  plotly    : librairie de graphiques interactifs
  pandas    : manipulation de tableaux de données
  numpy     : calculs numériques
  modules/  : les 5 modules actuariels du projet
"""

import streamlit as st       # Framework UI web : crée les widgets (sliders, métriques, etc.)
import numpy as np           # Calculs numériques : tableaux, NaN, opérations mathématiques
import pandas as pd          # Tableaux de données : construction et manipulation des DataFrames
import plotly.express as px  # Graphiques simples (histogrammes, barres, lignes, aires)
import plotly.graph_objects as go  # Graphiques avancés (gauge, waterfall, scatter personnalisé)
from plotly.subplots import make_subplots  # Permet de créer des figures avec plusieurs sous-graphiques

# ── Import des modules actuariels du projet ───────────────────────────────────
# Chaque module encapsule un domaine actuariel distinct.
# On n'importe que les classes nécessaires à l'affichage du dashboard.
from modules.vie import TableMortalite, AssuranceVie
from modules.non_vie import PortefeuilleAuto, ChainLadder, Tarification
from modules.solvabilite2 import SCR_NonVie, SCR_Vie, SolvabiliteII
from modules.ifrs17 import GroupeContrats
from modules.alm import Obligation, PortefeuilleActif, ProjectionPassif, AnalyseALM

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE DE LA PAGE
# ─────────────────────────────────────────────────────────────────────────────
# st.set_page_config() doit être le PREMIER appel Streamlit du fichier.
# layout="wide" = utilise toute la largeur de l'écran (meilleur pour les graphiques)
st.set_page_config(
    page_title="Projet Actuariel",
    page_icon="📊",
    layout="wide",
)

# Titre principal affiché en haut de toutes les pages
st.title("📊 Projet Actuariel — Dashboard Interactif")

# ── Création des 5 onglets ────────────────────────────────────────────────────
# st.tabs() retourne une liste d'objets "contexte" (tab_vie, tab_nv, ...)
# Chaque section "with tab_xxx:" correspond à un onglet affiché
tab_vie, tab_nv, tab_sii, tab_ifrs, tab_alm = st.tabs(
    ["🔵 Vie", "🟠 Non-Vie", "🟢 Solvabilité II", "🟣 IFRS 17", "🟡 ALM"]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ASSURANCE VIE
# ═══════════════════════════════════════════════════════════════════════════════
# Rôle : permet de calculer interactivement la prime nette et la provision
# mathématique d'un contrat décès temporaire.
#
# Pourquoi ? L'utilisateur peut faire varier l'âge, la durée, le capital
# et le taux technique pour comprendre leur impact sur le coût de l'assurance.
# ──────────────────────────────────────────────────────────────────────────────
with tab_vie:
    st.header("Assurance Vie — Table de mortalité & provisionnement")

    # st.columns([1, 2]) = divise l'espace en 2 colonnes de largeurs proportionnelles
    # col1 (largeur 1) = paramètres | col2 (largeur 2) = graphique (plus large)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Paramètres")

        # st.slider(label, min, max, valeur_défaut) = curseur interactif
        # Quand l'utilisateur bouge le curseur, Streamlit recalcule tout → résultat en temps réel
        age       = st.slider("Âge assuré",        20, 70, 45)
        duree     = st.slider("Durée du contrat",   5, 40, 20)

        # st.number_input(label, min, max, valeur_défaut, pas) = champ numérique
        capital   = st.number_input("Capital décès (€)", 10_000, 1_000_000, 100_000, 10_000)

        # Taux en % côté UI → divisé par 100 pour les calculs (ex: 2.5% → 0.025)
        taux_tech = st.slider("Taux technique (%)", 0.5, 5.0, 2.5, 0.25) / 100

    # ── Instanciation des objets actuariels ───────────────────────────────────
    # TableMortalite() : génère la table qx (Makeham)
    # AssuranceVie()   : encapsule les formules actuarielles (primes, provisions)
    table = TableMortalite()
    av    = AssuranceVie(table, taux_tech=taux_tech)

    # Calcul des deux indicateurs principaux
    prime = av.prime_nette(age=age, duree=duree, capital=capital)
    ev    = av.annuite_viagere(age)  # espérance de vie résiduelle = valeur de l'annuité

    with col1:
        # st.metric(label, value) = grande carte KPI avec valeur mise en évidence
        st.metric("Prime nette annuelle", f"{prime:,.0f} €")
        st.metric("Espérance de vie résiduelle", f"{ev:.1f} ans")

    with col2:
        # ── Graphique 1 : provision mathématique dans le temps ────────────────
        # provision_mathematique() retourne un DataFrame : annee / provision / prime
        # On visualise l'évolution de la réserve que l'assureur doit constituer
        df_prov = av.provision_mathematique(age, duree, capital)
        fig = px.area(
            df_prov, x="annee", y="provision",
            title=f"Provision mathématique — Assuré {age} ans, contrat {duree} ans",
            labels={"annee": "Année", "provision": "Provision (€)"},
            color_discrete_sequence=["#1f77b4"],
        )
        # use_container_width=True = le graphique prend toute la largeur de la colonne
        st.plotly_chart(fig, use_container_width=True)

    # ── Section table de mortalité (extrait interactif) ───────────────────────
    # Permet de visualiser les taux de décès qx sur une plage d'âges choisie
    st.subheader("Table de mortalité (extrait)")

    # key= est obligatoire quand on a plusieurs sliders du même type sur la même page
    # pour que Streamlit les différencie
    age_min_t = st.slider("Âge min table", 0, 80, 30, key="tmin")
    age_max_t = st.slider("Âge max table", age_min_t + 5, 120, 80, key="tmax")
    df_table  = table.afficher_table(age_min_t, age_max_t)

    # ── Graphique 2 : courbe de mortalité qx ─────────────────────────────────
    # Montre comment la probabilité de décès augmente avec l'âge
    # Cette courbe en forme exponentielle est caractéristique de la loi de Makeham
    fig2 = px.line(
        df_table, x="age", y="qx",
        title="Taux de mortalité qx",
        labels={"age": "Âge", "qx": "qx"},
        color_discrete_sequence=["#d62728"],
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ASSURANCE NON-VIE
# ═══════════════════════════════════════════════════════════════════════════════
# Rôle : simulation d'un portefeuille auto et calcul des provisions IBNR
# par la méthode Chain-Ladder.
#
# Pourquoi ? Le département non-vie doit constituer des provisions pour les
# sinistres survenus mais non encore déclarés ou réglés (IBNR).
# Le triangle Chain-Ladder est la méthode standard pour estimer ces IBNR.
# ──────────────────────────────────────────────────────────────────────────────
with tab_nv:
    st.header("Non-Vie — Portefeuille Auto & Chain-Ladder")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Portefeuille")
        # Paramètres de simulation du portefeuille
        n_contrats = st.number_input("Nb contrats", 1_000, 50_000, 10_000, 1_000)
        seed       = st.number_input("Seed aléatoire", 0, 999, 42)
        # Note : changer la seed génère un portefeuille différent (autre tirage aléatoire)

    # ── Cache Streamlit ────────────────────────────────────────────────────────
    # @st.cache_data = mise en cache : si n et s n'ont pas changé,
    # Streamlit ne re-simule pas le portefeuille → gain de performance significatif
    # Sans cache, la simulation de 10 000 contrats serait relancée à chaque interaction
    @st.cache_data
    def get_portefeuille(n, s):
        """Crée et met en cache le portefeuille auto simulé."""
        return PortefeuilleAuto(n_contrats=n, seed=s)

    ptf = get_portefeuille(int(n_contrats), int(seed))
    df  = ptf.df   # DataFrame du portefeuille (une ligne par contrat)

    with col2:
        # ── KPIs du portefeuille ───────────────────────────────────────────────
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Nb contrats",   f"{len(df):,}")
        col_b.metric("Fréq. moyenne", f"{df['nb_sinistres'].mean():.4f}")

        # Coût moyen uniquement sur les contrats avec sinistres (cout_total > 0)
        # On exclut les zéros pour ne pas biaiser la moyenne vers le bas
        sin_pos = df[df["cout_total"] > 0]["cout_total"]
        col_c.metric("Coût moyen",    f"{sin_pos.mean():,.0f} €")

    col3, col4 = st.columns(2)

    with col3:
        # ── Graphique 3 : distribution du nombre de sinistres ─────────────────
        # On s'attend à une distribution de Poisson (nb entier ≥ 0, asymétrique)
        # La majorité des assurés n'a aucun sinistre (0 domaine)
        fig3 = px.histogram(
            df, x="nb_sinistres", nbins=10,
            title="Distribution du nombre de sinistres",
            labels={"nb_sinistres": "Nb sinistres", "count": "Effectif"},
            color_discrete_sequence=["#ff7f0e"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # ── Graphique 4 : fréquence par zone géographique ─────────────────────
        # groupby("zone") + agg() = calcule les statistiques par zone
        # Permet de voir si la zone urbaine a une sinistralité plus élevée
        stats_zone = df.groupby("zone").agg(
            nb_contrats=("id", "count"),
            freq_moy=("nb_sinistres", "mean"),
        ).reset_index()
        fig4 = px.bar(
            stats_zone, x="zone", y="freq_moy",
            title="Fréquence moyenne par zone",
            color="zone",
            labels={"freq_moy": "Fréquence moy.", "zone": "Zone"},
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Section Chain-Ladder ──────────────────────────────────────────────────
    # Le triangle de développement représente les paiements cumulés par année
    # de survenance (lignes) et année de développement (colonnes).
    # np.nan = cases vides = montants encore inconnus (à estimer)
    st.subheader("Triangle Chain-Ladder")
    triangle_def = np.array([
        [4_500_000, 6_750_000, 7_650_000, 7_875_000],  # 2020 : données complètes
        [4_950_000, 7_425_000, 8_415_000, np.nan],      # 2021 : manque col 4
        [5_400_000, 8_100_000, np.nan,    np.nan],       # 2022 : manque col 3-4
        [5_850_000, np.nan,    np.nan,    np.nan],        # 2023 : seulement col 1 connue
    ])
    cl      = ChainLadder(triangle_def)
    df_ibnr = cl.calculer_ibnr()  # calcule les facteurs, complète le triangle et retourne les IBNR

    # ── Graphique 5 : IBNR par année de survenance ────────────────────────────
    # Montre quelle année a la plus grande provision à constituer
    # text_auto=True = affiche les valeurs directement sur les barres
    fig5 = px.bar(
        df_ibnr, x="annee_surv", y="IBNR",
        title="Provision IBNR par année de survenance",
        labels={"annee_surv": "Année survenance", "IBNR": "IBNR (€)"},
        color_discrete_sequence=["#2ca02c"],
        text_auto=True,
    )
    st.plotly_chart(fig5, use_container_width=True)

    # Tableau détaillé sous le graphique : dernier connu, ultime projeté, IBNR
    st.dataframe(df_ibnr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — SOLVABILITÉ II
# ═══════════════════════════════════════════════════════════════════════════════
# Rôle : calcule le SCR (capital réglementaire) et le ratio de solvabilité SII.
#
# Pourquoi ? La réglementation Solvabilité II impose aux assureurs de détenir
# un capital minimum (SCR) pour résister à un choc de marché avec 99,5% de
# probabilité sur 1 an. Le dashboard permet de tester la solidité financière
# pour différentes tailles de portefeuille.
# ──────────────────────────────────────────────────────────────────────────────
with tab_sii:
    st.header("Solvabilité II — SCR & Ratio de couverture")

    st.subheader("SCR Non-Vie")

    # ── Saisie des volumes par ligne de métier ─────────────────────────────────
    # 3 colonnes : col1 = primes NV, col2 = provisions NV, col3 = BE Vie + FP
    col1, col2, col3 = st.columns(3)

    with col1:
        # Volumes de primes annuelles par ligne de métier (en M€)
        # Ces montants entrent dans le calcul du SCR Primes
        p_auto_rc  = st.number_input("Primes Auto RC (M€)",  0.1, 50.0, 5.0, 0.5) * 1e6
        p_auto_do  = st.number_input("Primes Auto Dom (M€)", 0.1, 50.0, 3.0, 0.5) * 1e6
        p_incendie = st.number_input("Primes Incendie (M€)", 0.1, 50.0, 2.0, 0.5) * 1e6

    with col2:
        # Provisions de réserves par ligne de métier (en M€)
        # Ces montants entrent dans le calcul du SCR Réserves
        r_auto_rc  = st.number_input("Prov. Auto RC (M€)",  0.1, 100.0, 8.0, 0.5) * 1e6
        r_auto_do  = st.number_input("Prov. Auto Dom (M€)", 0.1, 100.0, 4.0, 0.5) * 1e6
        r_incendie = st.number_input("Prov. Incendie (M€)", 0.1, 100.0, 3.5, 0.5) * 1e6

    with col3:
        # Données Vie et fonds propres pour le BSCR global
        be_deces     = st.number_input("BE Décès (M€)",   1.0, 200.0, 15.0, 1.0) * 1e6
        be_rentes    = st.number_input("BE Rentes (M€)",  1.0, 200.0, 20.0, 1.0) * 1e6
        fonds_propres= st.number_input("Fonds propres (M€)", 1.0, 500.0, 50.0, 1.0) * 1e6

    # ── Calcul du SCR Non-Vie ────────────────────────────────────────────────
    scr_nl_obj = SCR_NonVie(
        {"Auto RC": p_auto_rc, "Auto Dommages": p_auto_do, "Incendie": p_incendie},
        {"Auto RC": r_auto_rc, "Auto Dommages": r_auto_do, "Incendie": r_incendie},
    )
    # scr_par_ligne() retourne un DataFrame avec SCR_Primes et SCR_Reserves par ligne
    df_scr_ligne = scr_nl_obj.scr_par_ligne().reset_index()
    df_scr_ligne.columns = ["Ligne"] + list(df_scr_ligne.columns[1:])

    # ── Graphique 6 : SCR par ligne de métier (barres groupées) ──────────────
    # Permet de voir quelle ligne de métier contribue le plus au SCR
    # et quelle composante (primes vs réserves) est prépondérante
    fig6 = px.bar(
        df_scr_ligne, x="Ligne", y=["SCR_Primes", "SCR_Reserves"],
        barmode="group",
        title="SCR par ligne et composante",
        labels={"value": "SCR (€)", "variable": "Composante"},
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ── Calcul BSCR + Ratio SII ───────────────────────────────────────────────
    scr_nl = scr_nl_obj.scr_total()
    scr_v, chocs = SCR_Vie(be_deces, be_rentes, 5e6, 20e6, 100e6).calculer()
    scr_marche   = 3.5e6   # SCR Marché simplifié (fixe dans cette démo)
    sii = SolvabiliteII(scr_nl, scr_v, scr_marche, fonds_propres)

    # ── KPIs principaux ───────────────────────────────────────────────────────
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("SCR Non-Vie",  f"{scr_nl/1e6:.2f} M€")
    col_m2.metric("SCR Vie",      f"{scr_v/1e6:.2f} M€")
    col_m3.metric("SCR Final",    f"{sii.scr()/1e6:.2f} M€")
    ratio = sii.ratio_solvabilite()
    # delta = écart au seuil réglementaire de 100%
    # Vert si > 0 (excédent de capital), rouge si < 0 (déficit)
    col_m4.metric("Ratio SII",    f"{ratio:.1f}%", delta=f"{ratio-100:.1f}%")

    # ── Graphique 7 : Jauge de solvabilité ────────────────────────────────────
    # go.Indicator avec mode="gauge+number+delta" = jauge circulaire type compteur
    # Zones colorées : rouge (<100%), orange (100-200%), vert (>200%)
    # Le seuil réglementaire est matérialisé par la ligne rouge à 100%
    fig7 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        delta={"reference": 100},  # delta affiché = ratio - 100% (seuil légal)
        gauge={
            "axis": {"range": [0, 300]},
            "bar":  {"color": "darkblue"},
            "steps": [
                {"range": [0,   100], "color": "#ff4b4b"},   # rouge : non solvable
                {"range": [100, 200], "color": "#ffd700"},   # jaune : solvable mais fragile
                {"range": [200, 300], "color": "#00cc44"},   # vert : bonne solvabilité
            ],
            # Ligne rouge épaisse à 100% = seuil réglementaire minimum
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 100},
        },
        title={"text": "Ratio de Solvabilité (%)"},
    ))
    st.plotly_chart(fig7, use_container_width=True)

    # ── Graphique 8 : Waterfall BSCR ─────────────────────────────────────────
    # Un graphique en cascade (waterfall) montre la contribution de chaque module
    # au BSCR total. Chaque barre représente un module SCR.
    # measure="relative" = chaque barre s'ajoute à la précédente
    # measure="total"    = barre finale = somme de toutes les relatives
    fig8 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["SCR Non-Vie", "SCR Vie", "SCR Marché", "BSCR"],
        y=[scr_nl / 1e6, scr_v / 1e6, scr_marche / 1e6, sii.bscr() / 1e6],
        connector={"line": {"color": "rgb(63,63,63)"}},
    ))
    fig8.update_layout(title="Décomposition BSCR (M€)", yaxis_title="M€")
    st.plotly_chart(fig8, use_container_width=True)

    # ── Graphique 9 : Camembert SCR Vie par risque ────────────────────────────
    # Montre la répartition entre les différents risques vie :
    # mortalité, longévité, invalidité, rachat, frais, catastrophe
    df_chocs = pd.DataFrame(
        [{"Risque": k, "SCR (€)": round(v)} for k, v in chocs.items()]
    )
    fig9 = px.pie(
        df_chocs, values="SCR (€)", names="Risque",
        title="Répartition SCR Vie par risque",
    )
    st.plotly_chart(fig9, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — IFRS 17
# ═══════════════════════════════════════════════════════════════════════════════
# Rôle : modélise le passif IFRS 17 d'un groupe de contrats d'assurance vie
# selon l'approche GMM (General Measurement Model = approche standard IFRS 17).
#
# Pourquoi ? IFRS 17 (en vigueur depuis 2023) a fondamentalement changé la façon
# dont les assureurs comptabilisent leurs contrats. Contrairement à l'ancienne
# norme, les primes ne sont plus des revenus directs — le résultat provient du
# service rendu progressivement (amortissement de la CSM).
#
# Le dashboard permet de comprendre la sensibilité du passif IFRS 17
# aux hypothèses actuarielles (taux, mortalité, frais).
# ──────────────────────────────────────────────────────────────────────────────
with tab_ifrs:
    st.header("IFRS 17 — Modèle GMM (General Measurement Model)")

    col1, col2 = st.columns(2)

    with col1:
        # ── Paramètres du portefeuille de contrats ─────────────────────────────
        nb_contrats = st.number_input("Nb contrats",   100, 10_000, 1_000, 100)
        duree_ifrs  = st.slider("Durée (ans)",          5,      40,    20)
        prime_ann   = st.number_input("Prime annuelle (€)", 100, 10_000, 1_380, 100)
        capital_d   = st.number_input("Capital décès (€)",  10_000, 500_000, 100_000, 10_000)

    with col2:
        # ── Hypothèses actuarielles ────────────────────────────────────────────
        taux_act    = st.slider("Taux actualisation (%)", 0.5, 8.0, 2.5, 0.1) / 100
        taux_mort   = st.slider("Taux mortalité (%)",     0.1, 5.0, 0.5, 0.1) / 100
        frais_gest  = st.number_input("Frais de gestion (€/an)", 0, 1_000, 80, 10)
        frais_acq   = st.number_input("Frais acquisition (€)",   0, 1_000, 150, 10)
        # CoV (Coefficient of Variation) = mesure de l'incertitude pour le RA
        cov_risque  = st.slider("CoV risque (%)", 1, 20, 5) / 100

    # ── Construction du groupe de contrats IFRS 17 ────────────────────────────
    # GroupeContrats() reçoit toutes les hypothèses sous forme de dict
    gc = GroupeContrats({
        "nb_contrats":        int(nb_contrats),
        "duree":              int(duree_ifrs),
        "prime_annuelle":     float(prime_ann),
        "capital_deces":      float(capital_d),
        "taux_actualisation": float(taux_act),
        "taux_mortalite":     float(taux_mort),
        "frais_gestion":      float(frais_gest),
        "frais_acquisition":  float(frais_acq),
        "cov_risque":         float(cov_risque),
    })

    # ── Calcul des trois composantes du passif IFRS 17 ────────────────────────
    # BEL = valeur actuelle des flux futurs attendus (Best Estimate)
    # RA  = majoration pour incertitude (Risk Adjustment)
    # CSM = profit futur non encore reconnu (Contractual Service Margin)
    bel = gc.best_estimate()
    ra  = gc.risk_adjustment()
    csm = gc.csm_initial()

    # ── KPIs passif IFRS 17 ───────────────────────────────────────────────────
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Best Estimate (BEL)", f"{bel:,.0f} €")
    col_b.metric("Risk Adjustment",     f"{ra:,.0f} €")
    col_c.metric("CSM initial",         f"{csm:,.0f} €")
    col_d.metric("Total Passif IFRS17", f"{bel+ra+csm:,.0f} €")

    # ── Graphique 10 : Waterfall du passif IFRS 17 ───────────────────────────
    # Visualise la décomposition BEL + RA + CSM = Passif Total
    # Cette décomposition est au cœur d'IFRS 17 (article 32 de la norme)
    fig10 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["BEL", "RA", "CSM", "Passif Total"],
        y=[bel, ra, csm, bel + ra + csm],
        connector={"line": {"color": "rgb(63,63,63)"}},
    ))
    fig10.update_layout(title="Passif IFRS 17 — GMM (€)", yaxis_title="€")
    st.plotly_chart(fig10, use_container_width=True)

    # ── Graphique 11 : Amortissement de la CSM ───────────────────────────────
    # La CSM est un profit différé : elle est relâchée progressivement en résultat
    # au fil du service rendu aux assurés (proportionnel aux survivants).
    # Ce graphique montre le rythme de reconnaissance du profit.
    df_csm = gc.amortissement_csm()
    fig11 = px.bar(
        df_csm.head(min(10, len(df_csm))),  # affiche max 10 ans
        x="annee", y="release_CSM",
        title="Amortissement du CSM (10 premières années)",
        labels={"annee": "Année", "release_CSM": "Release CSM (€)"},
        color_discrete_sequence=["#9467bd"],
    )
    st.plotly_chart(fig11, use_container_width=True)

    # ── Compte de résultat IFRS 17 ────────────────────────────────────────────
    # Sous IFRS 17, le résultat = Release CSM + Release RA (et non plus les primes reçues)
    st.subheader("Compte de résultat IFRS 17")
    df_cr = gc.compte_resultat()

    # ── Graphique 12 : Revenus vs Charges ────────────────────────────────────
    # barmode="overlay" = barres superposées (pour comparer revenus et charges)
    # On s'attend à ce que les revenus > charges → résultat technique positif
    fig12 = px.bar(
        df_cr, x="Année",
        y=["Revenus assurance", "Charges assurance"],
        barmode="overlay",
        title="Revenus vs Charges d'assurance",
        labels={"value": "€", "variable": "Poste"},
    )
    st.plotly_chart(fig12, use_container_width=True)

    # ── Graphique 13 : Résultat net ───────────────────────────────────────────
    # markers=True = points visibles sur la ligne (pour voir les valeurs annuelles)
    # On visualise l'évolution du bénéfice sur les 10 premières années
    fig13 = px.line(
        df_cr, x="Année", y="Résultat net",
        title="Résultat net IFRS 17",
        markers=True,
        color_discrete_sequence=["#17becf"],
    )
    st.plotly_chart(fig13, use_container_width=True)

    # Tableau détaillé du P&L : toutes les colonnes année par année
    st.subheader("Tableau — Compte de résultat détaillé")
    st.dataframe(df_cr.set_index("Année"), use_container_width=True)

    # ── Flux de trésorerie actualisés ─────────────────────────────────────────
    # Visualise les entrées (primes) et sorties (prestations) par année,
    # ainsi que le flux net actualisé (base du BEL)
    st.subheader("Flux de trésorerie actualisés")
    df_flux = gc.flux_tresorerie()

    # make_subplots = 2 graphiques côte à côte dans la même figure
    # cols=2 avec subplot_titles = titres individuels de chaque sous-graphique
    fig14 = make_subplots(rows=1, cols=2, subplot_titles=["Flux bruts", "Flux nets actualisés"])

    # Sous-graphique gauche : flux bruts (primes entrantes / prestations sortantes)
    fig14.add_trace(go.Bar(name="Primes",      x=df_flux["annee"], y=df_flux["primes"],      marker_color="#2ca02c"), row=1, col=1)
    fig14.add_trace(go.Bar(name="Prestations", x=df_flux["annee"], y=df_flux["prestations"], marker_color="#d62728"), row=1, col=1)

    # Sous-graphique droit : flux net actualisé = base du calcul du BEL
    fig14.add_trace(go.Scatter(name="Flux net actualisé", x=df_flux["annee"], y=df_flux["flux_actualise"], mode="lines+markers", marker_color="#1f77b4"), row=1, col=2)

    fig14.update_layout(title_text="Flux de trésorerie IFRS 17", barmode="group")
    st.plotly_chart(fig14, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ALM (ASSET-LIABILITY MANAGEMENT)
# ═══════════════════════════════════════════════════════════════════════════════
# Rôle : analyse de l'adossement actif-passif d'un assureur vie.
# Calcule le duration gap, le bilan économique (surplus = Actif - BEL)
# et l'impact de chocs de taux sur le surplus.
#
# Pourquoi ? Les assureurs vie sont très exposés au risque de taux :
#   - Si les taux montent → la valeur des obligations baisse
#   - Si les taux baissent → le BEL augmente (les flux futurs sont moins actualisés)
# Le duration gap mesure le déséquilibre entre les sensibilités actif et passif.
# L'objectif de l'ALM est d'immuniser le surplus contre les mouvements de taux.
# ──────────────────────────────────────────────────────────────────────────────
with tab_alm:
    st.header("ALM — Asset-Liability Management (Assurance Vie)")

    # ── SECTION PARAMÈTRES ────────────────────────────────────────────────────
    st.subheader("Paramètres du portefeuille")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        # ── Portefeuille obligataire : 3 obligations ───────────────────────────
        # Chaque obligation est définie par : nominal, coupon, maturité, taux
        # La maturité échelonnée (5, 10, 15 ans) permet une meilleure immunisation
        st.markdown("**Portefeuille obligataire**")
        nom_obl1  = st.number_input("Nominal Oblig. 1 (M€)", 1.0, 200.0, 20.0, 1.0) * 1e6
        coup_obl1 = st.slider("Coupon 1 (%)", 0.5, 8.0, 3.0, 0.1) / 100
        mat_obl1  = st.slider("Maturité 1 (ans)", 1, 30, 5)
        trd_obl1  = st.slider("Taux rend. 1 (%)", 0.5, 8.0, 3.5, 0.1) / 100

        nom_obl2  = st.number_input("Nominal Oblig. 2 (M€)", 1.0, 200.0, 30.0, 1.0) * 1e6
        coup_obl2 = st.slider("Coupon 2 (%)", 0.5, 8.0, 4.0, 0.1) / 100
        mat_obl2  = st.slider("Maturité 2 (ans)", 1, 30, 10)
        trd_obl2  = st.slider("Taux rend. 2 (%)", 0.5, 8.0, 4.0, 0.1) / 100

        nom_obl3  = st.number_input("Nominal Oblig. 3 (M€)", 1.0, 200.0, 20.0, 1.0) * 1e6
        coup_obl3 = st.slider("Coupon 3 (%)", 0.5, 8.0, 2.5, 0.1) / 100
        mat_obl3  = st.slider("Maturité 3 (ans)", 1, 30, 15)
        trd_obl3  = st.slider("Taux rend. 3 (%)", 0.5, 8.0, 3.8, 0.1) / 100

    with col_b:
        # ── Actions et immobilier ──────────────────────────────────────────────
        # Ces classes d'actifs ont une duration proxy (non obligataire)
        # Actions : duration ≈ 0 en ALM classique (insensibles aux taux)
        # Immobilier : duration proxy ≈ 7 ans (baux long terme)
        st.markdown("**Actions & Immobilier**")
        val_act   = st.number_input("Valeur actions (M€)",     0.0, 100.0, 10.0, 1.0) * 1e6
        rend_act  = st.slider("Rendement actions (%)",          1.0, 15.0,  7.0, 0.5) / 100
        val_imm   = st.number_input("Valeur immobilier (M€)",  0.0, 100.0,  5.0, 1.0) * 1e6
        rend_imm  = st.slider("Rendement immobilier (%)",       1.0, 10.0,  4.0, 0.5) / 100
        dur_imm   = st.slider("Duration immobilier (ans)",      1.0, 15.0,  7.0, 0.5)

        # ── Passif vie ────────────────────────────────────────────────────────
        # Ces paramètres définissent le portefeuille d'assurés (côté passif)
        st.markdown("**Passif — portefeuille vie**")
        age_moy_alm = st.slider("Âge moyen assuré",             25, 70, 45)
        nb_ct_alm   = st.number_input("Nb contrats",           100, 50_000, 1_000, 100)
        cap_moy_alm = st.number_input("Capital moyen (€)",  10_000, 500_000, 100_000, 10_000)
        dur_alm     = st.slider("Durée résiduelle (ans)",        5, 30, 20)

    with col_c:
        # ── Hypothèses actuarielles du passif ─────────────────────────────────
        # taux_actualisation : taux sans risque (EIOPA) pour actualiser les flux
        # taux_rachat        : proportion d'assurés qui résilie par an
        # taux_frais         : charges de gestion en % des engagements
        # coefficient_mortalite : ajustement d'expérience (1.0 = table standard)
        st.markdown("**Paramètres actuariels passif**")
        prime_alm      = st.number_input("Prime annuelle (€)",    100, 10_000, 1_500, 100)
        taux_act_alm   = st.slider("Taux actualisation (%)",     0.5,  8.0,   3.5, 0.1) / 100
        taux_rac_alm   = st.slider("Taux de rachat (%)",         0.5, 10.0,   3.0, 0.5) / 100
        taux_frais_alm = st.slider("Taux de frais (%)",          0.1,  3.0,   0.8, 0.1) / 100
        coeff_mort_alm = st.slider("Coefficient mortalité",      0.5,  2.0,   1.0, 0.1)

    # ── Construction des objets ALM ───────────────────────────────────────────
    # 1. Trois obligations avec les paramètres saisis ci-dessus
    obligations_alm = [
        Obligation(nom_obl1, coup_obl1, mat_obl1, trd_obl1, "AA",  "souveraine"),
        Obligation(nom_obl2, coup_obl2, mat_obl2, trd_obl2, "BBB", "entreprise"),
        Obligation(nom_obl3, coup_obl3, mat_obl3, trd_obl3, "A",   "souveraine"),
    ]
    # 2. Portefeuille d'actifs : obligations + actions + immobilier
    actif_alm = PortefeuilleActif(
        obligations          = obligations_alm,
        valeur_actions       = val_act,
        rendement_actions    = rend_act,
        valeur_immobilier    = val_imm,
        rendement_immobilier = rend_imm,
        duration_immobilier  = dur_imm,
    )
    # 3. Projection du passif vie (flux de trésorerie futurs)
    passif_alm = ProjectionPassif(
        age_moyen             = age_moy_alm,
        capital_moyen         = cap_moy_alm,
        nb_contrats           = int(nb_ct_alm),
        duree                 = dur_alm,
        prime_annuelle        = prime_alm,
        taux_actualisation    = taux_act_alm,
        taux_rachat           = taux_rac_alm,
        taux_frais            = taux_frais_alm,
        coefficient_mortalite = coeff_mort_alm,
    )
    # 4. Analyse ALM : combine actif et passif pour les indicateurs
    alm         = AnalyseALM(actif_alm, passif_alm)
    rapport_alm = alm.rapport_immunisation()

    # ── SECTION KPIs : BILAN ÉCONOMIQUE ──────────────────────────────────────
    # Le bilan économique instantané résume la situation actif/passif en 5 chiffres
    st.subheader("Bilan économique instantané")
    k1, k2, k3, k4, k5 = st.columns(5)

    # Valeur Actif = somme des actifs au prix de marché
    k1.metric("Valeur Actif",    f"{rapport_alm['Valeur actif (EUR)'] / 1e6:.2f} M€")
    # BEL = valeur actualisée des flux passifs nets sortants
    k2.metric("BEL",             f"{rapport_alm['BEL (EUR)'] / 1e6:.2f} M€")
    # Surplus = Actif - BEL = fonds propres économiques (EVE = Economic Value of Equity)
    k3.metric("Surplus (EVE)",   f"{rapport_alm['Surplus (EUR)'] / 1e6:.2f} M€")
    # Ratio de couverture = Actif / BEL (≥ 1 = assureur solvable économiquement)
    k4.metric("Ratio couverture",f"{rapport_alm['Ratio couverture']:.3f}")
    # DV01 = variation du surplus pour +1 point de base (+0,01%) de taux
    k5.metric("DV01",            f"{rapport_alm['DV01 (EUR/bp)']:,.0f} €/bp")

    # ── SECTION DURATION GAP ─────────────────────────────────────────────────
    # Duration Gap = Duration Actif - (BEL/Actif) × Duration Passif
    # Gap > 0 : actif plus sensible → risque si taux montent
    # Gap < 0 : passif plus sensible → risque si taux baissent
    # Gap ≈ 0 : immunisation de Redington (objectif ALM)
    st.subheader("Duration Gap")
    dg_col1, dg_col2 = st.columns(2)

    with dg_col1:
        d_gap = rapport_alm["Duration Gap (ans)"]
        # Couleur de la jauge selon l'urgence du duration gap
        couleur_gap = "#00cc44" if abs(d_gap) < 0.5 else ("#ffd700" if abs(d_gap) < 2 else "#ff4b4b")

        # ── Graphique 15 : Jauge Duration Actif vs. Passif ────────────────────
        # La valeur affichée = duration actif
        # La ligne rouge = duration passif (cible à atteindre)
        # Delta = duration gap (écart entre les deux)
        fig_gap = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=rapport_alm["Duration actif (ans)"],
            # delta = durée actif - durée passif = duration gap
            delta={"reference": rapport_alm["Duration passif (ans)"], "valueformat": ".2f"},
            gauge={
                "axis":  {"range": [0, 20]},
                "bar":   {"color": "steelblue"},
                "steps": [
                    {"range": [0,  5], "color": "#eef"},
                    {"range": [5, 10], "color": "#ddf"},
                    {"range": [10,20], "color": "#ccf"},
                ],
                # Ligne rouge = duration passif (cible d'immunisation)
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": rapport_alm["Duration passif (ans)"],
                },
            },
            title={"text": "Duration Actif vs. Passif (ans)<br><sup>Ligne rouge = Duration Passif</sup>"},
        ))
        st.plotly_chart(fig_gap, use_container_width=True)

    with dg_col2:
        # ── Graphique 16 : Allocation du portefeuille d'actifs ────────────────
        # Camembert montrant la répartition obligations / actions / immobilier
        # L'ALM s'intéresse principalement à la part obligataire (sensible aux taux)
        alloc = actif_alm.allocation
        fig_alloc = px.pie(
            names=list(alloc.keys()),
            values=[v * 100 for v in alloc.values()],
            title="Allocation du portefeuille d'actifs (%)",
            color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c"],
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

    # ── Bandeau de synthèse immunisation ─────────────────────────────────────
    # Résume en une ligne les trois conditions d'immunisation de Redington :
    # 1. VA(actif) ≈ VA(passif) | 2. Duration actif ≈ Duration passif | 3. Convexité actif > passif
    imm_ok = rapport_alm["Immunisation OK"]
    st.info(
        f"**Duration Gap = {d_gap:+.2f} ans** | "
        f"DV01 = {rapport_alm['DV01 (EUR/bp)']:,.0f} EUR/bp | "
        f"Immunisation de Redington : {'✅ satisfaite' if imm_ok else '⚠️ non satisfaite'}"
    )

    # ── Tableau du portefeuille obligataire ──────────────────────────────────
    # Affiche les caractéristiques de chaque obligation : prix, duration, convexité
    st.subheader("Détail du portefeuille obligataire")
    df_obl = actif_alm.tableau_obligations()
    st.dataframe(df_obl, use_container_width=True)

    # ── SECTION ANALYSE DE SENSIBILITÉ AUX TAUX ──────────────────────────────
    # Simule l'impact de chocs de taux parallèles (±200, ±100, ±50, 0 bps)
    # sur le surplus économique = stress testing réglementaire
    st.subheader("Analyse de sensibilité — Scénarios de taux")
    df_sc = alm.tableau_scenarios()

    # ── Graphique 17 : Impact des chocs de taux (2 panneaux) ─────────────────
    # Panneau gauche : variation de surplus en € (rouge = perte, vert = gain)
    # Panneau droit  : valeurs stressées de l'actif et du BEL en M€
    fig_sc = make_subplots(rows=1, cols=2,
                           subplot_titles=["Impact sur le Surplus (€)", "Actif & BEL stressés (M€)"])

    # Couleurs des barres : rouge si variation négative, vert si positive
    colors_sc = ["#d62728" if v < 0 else "#2ca02c" for v in df_sc["Var. Surplus (EUR)"]]

    # Barres des variations de surplus (col 1)
    fig_sc.add_trace(
        go.Bar(x=df_sc["Choc (bps)"].astype(str) + " bps",
               y=df_sc["Var. Surplus (EUR)"],
               marker_color=colors_sc, name="Var. Surplus"),
        row=1, col=1,
    )
    # Courbes des valeurs stressées (col 2) : actif en bleu, BEL en rouge
    fig_sc.add_trace(
        go.Scatter(x=df_sc["Choc (bps)"].astype(str) + " bps",
                   y=df_sc["Actif stresse (EUR)"] / 1e6,
                   mode="lines+markers", name="Actif stresse", marker_color="steelblue"),
        row=1, col=2,
    )
    fig_sc.add_trace(
        go.Scatter(x=df_sc["Choc (bps)"].astype(str) + " bps",
                   y=df_sc["BEL stresse (EUR)"] / 1e6,
                   mode="lines+markers", name="BEL stresse", marker_color="tomato"),
        row=1, col=2,
    )
    fig_sc.update_layout(title_text="Sensibilité aux chocs de taux parallèles")
    st.plotly_chart(fig_sc, use_container_width=True)

    # Tableau détaillé des scénarios (choc, ΔActif, ΔBEL, ΔSurplus, ...)
    st.dataframe(df_sc.set_index("Choc (bps)"), use_container_width=True)

    # ── SECTION FLUX DU PASSIF PROJETÉ ───────────────────────────────────────
    # Visualise les flux annuels du portefeuille vie sur toute la durée :
    # primes (entrées), capitaux décès, rachats, frais (sorties)
    # et la population de survivants restante
    st.subheader("Flux de trésorerie du passif projeté")
    df_flux_alm = passif_alm.projeter_flux()

    # ── Graphique 18 : Flux du passif (2 panneaux) ───────────────────────────
    # barmode="relative" = barres empilées permettant de voir le flux net visuel
    # Panneau gauche : flux bruts par composante (primes, décès, rachats, frais)
    # Panneau droit  : évolution du nombre de survivants (décroissant)
    fig_flux = make_subplots(rows=1, cols=2,
                             subplot_titles=["Flux bruts par composante (€)", "Population en cours"])
    # Primes = flux entrant (vert, positif)
    fig_flux.add_trace(go.Bar(name="Primes",         x=df_flux_alm["annee"], y=df_flux_alm["primes"],           marker_color="#2ca02c"), row=1, col=1)
    # Les sorties sont mises en négatif (−) pour l'empilement "relative"
    fig_flux.add_trace(go.Bar(name="Capitaux décès", x=df_flux_alm["annee"], y=-df_flux_alm["capitaux_deces"],  marker_color="#d62728"), row=1, col=1)
    fig_flux.add_trace(go.Bar(name="Rachats",        x=df_flux_alm["annee"], y=-df_flux_alm["rachats_montant"], marker_color="#ff7f0e"), row=1, col=1)
    fig_flux.add_trace(go.Bar(name="Frais",          x=df_flux_alm["annee"], y=-df_flux_alm["frais"],           marker_color="#9467bd"), row=1, col=1)
    # Courbe du nombre de vivants (décroissante : décès + rachats)
    fig_flux.add_trace(go.Scatter(name="Nb vivants", x=df_flux_alm["annee"], y=df_flux_alm["nb_vivants"],
                                  mode="lines+markers", marker_color="#1f77b4"), row=1, col=2)
    fig_flux.update_layout(barmode="relative", title_text="Projection des flux passif vie")
    st.plotly_chart(fig_flux, use_container_width=True)

    # ── SECTION PROJECTION DU BILAN ÉCONOMIQUE ───────────────────────────────
    # Projette l'évolution de l'actif et du BEL sur un horizon choisi
    # Permet de visualiser si le surplus reste positif dans le temps
    st.subheader("Projection du bilan économique")
    # Le slider est limité à min(durée du contrat, 15 ans) pour la lisibilité
    horizon_alm = st.slider("Horizon de projection (ans)", 3, min(dur_alm, 15), 10, key="horizon_alm")
    df_bilan = alm.projection_bilan(horizon_alm)

    # ── Graphique 19 : Évolution du bilan (actif, BEL, surplus) ──────────────
    # Trois courbes superposées :
    #   Actif (bleu)   : doit rester au-dessus du BEL
    #   BEL (rouge)    : décroît au fur et à mesure des prestations
    #   Surplus (or)   : zone remplie = coussin de sécurité économique
    fig_bilan = go.Figure()
    fig_bilan.add_trace(go.Scatter(
        x=df_bilan["Année"], y=df_bilan["Valeur actif (€)"] / 1e6,
        mode="lines+markers", name="Actif", line={"color": "steelblue", "width": 2}
    ))
    fig_bilan.add_trace(go.Scatter(
        x=df_bilan["Année"], y=df_bilan["BEL (€)"] / 1e6,
        mode="lines+markers", name="BEL", line={"color": "tomato", "width": 2}
    ))
    # fill="tozeroy" = zone colorée entre la courbe et l'axe zéro (surplus visible)
    fig_bilan.add_trace(go.Scatter(
        x=df_bilan["Année"], y=df_bilan["Surplus (€)"] / 1e6,
        fill="tozeroy", name="Surplus", line={"color": "gold", "width": 2},
        fillcolor="rgba(255,215,0,0.2)"  # couleur or semi-transparente
    ))
    fig_bilan.update_layout(
        title="Évolution du bilan économique (M€)",
        xaxis_title="Année",
        yaxis_title="M€",
        hovermode="x unified",  # affiche toutes les valeurs au survol d'un point x
    )
    st.plotly_chart(fig_bilan, use_container_width=True)

    # Tableau numérique de la projection du bilan (année par année)
    st.dataframe(df_bilan.set_index("Année"), use_container_width=True)
