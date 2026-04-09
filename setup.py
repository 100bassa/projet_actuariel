"""
setup.py — Génère automatiquement toute la structure du projet actuariel.
Usage : python setup.py
"""
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  [OK]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# File contents
# ─────────────────────────────────────────────────────────────────────────────

MODULES_INIT = ""

# ── modules/vie.py ────────────────────────────────────────────────────────────
MODULES_VIE = '''\
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


class TableMortalite:
    def __init__(self):
        ages = np.arange(0, 121)
        a, b, c = -7.5, -10, 1.10
        qx = np.minimum(1.0, np.exp(a) + np.exp(b) * (c ** ages))
        self.table = pd.DataFrame({"age": ages, "qx": qx, "px": 1 - qx})

    def get_qx(self, age):
        ligne = self.table.loc[self.table["age"] == age, "qx"]
        return ligne.values[0] if len(ligne) > 0 else 0.0

    def probabilite_survie(self, age_depart, t):
        px_values = self.table.loc[
            (self.table["age"] >= age_depart) & (self.table["age"] < age_depart + t), "px"
        ].values
        return np.prod(px_values)

    def esperance_vie(self, age):
        return sum(self.probabilite_survie(age, t) for t in range(1, 120 - age))

    def afficher_table(self, age_min=30, age_max=80):
        masque = (self.table["age"] >= age_min) & (self.table["age"] <= age_max)
        return self.table[masque].round(6)


class AssuranceVie:
    def __init__(self, table: TableMortalite, taux_tech: float = 0.02):
        self.table = table
        self.taux_tech = taux_tech
        self.v = 1 / (1 + taux_tech)

    def _tpx(self, x, t):
        return self.table.probabilite_survie(x, t)

    def _qx(self, x):
        return self.table.get_qx(x)

    def annuite_viagere(self, age, duree_max=100):
        v, total = self.v, 0.0
        for t in range(duree_max - age):
            total += self._tpx(age, t) * (v ** t)
        return total

    def assurance_deces_temporaire(self, age, duree):
        v, total = self.v, 0.0
        for t in range(duree):
            tpx = self._tpx(age, t)
            qx_t = self._qx(age + t)
            total += tpx * qx_t * (v ** (t + 1))
        return total

    def prime_nette(self, age, duree, capital):
        Ax_n = self.assurance_deces_temporaire(age, duree)
        ax_n = sum(self._tpx(age, t) * (self.v ** t) for t in range(duree))
        if ax_n == 0:
            return 0
        return (capital * Ax_n) / ax_n

    def provision_mathematique(self, age, duree, capital):
        prime = self.prime_nette(age, duree, capital)
        provisions = []
        for t in range(duree + 1):
            age_t = age + t
            duree_r = duree - t
            if duree_r <= 0:
                vm = 0.0
            else:
                Ax = self.assurance_deces_temporaire(age_t, duree_r)
                ax = sum(self._tpx(age_t, k) * (self.v ** k) for k in range(duree_r))
                vm = capital * Ax - prime * ax
            provisions.append({
                "annee": t,
                "age": age_t,
                "provision": round(vm, 2),
                "prime": round(prime, 2),
            })
        return pd.DataFrame(provisions)
'''

# ── modules/non_vie.py ────────────────────────────────────────────────────────
MODULES_NON_VIE = '''\
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


class PortefeuilleAuto:
    def __init__(self, n_contrats=10000, seed=42):
        self.n = n_contrats
        np.random.seed(seed)
        self.df = self._simuler()

    def _simuler(self):
        n = self.n
        df = pd.DataFrame({
            "id": range(n),
            "age": np.random.randint(18, 75, n),
            "anciennete": np.random.randint(0, 50, n),
            "zone": np.random.choice(["urbain", "rural", "mixte"], n, p=[0.5, 0.3, 0.2]),
            "categorie_vh": np.random.choice(
                ["citadine", "berline", "SUV", "sport"], n, p=[0.4, 0.3, 0.2, 0.1]
            ),
            "bonus_malus": np.random.uniform(0.5, 3.5, n),
            "exposition": np.random.uniform(0.1, 1.0, n),
        })
        df["jeune"] = (df["age"] < 25).astype(int)
        df["senior"] = (df["age"] > 65).astype(int)
        df["novice"] = (df["anciennete"] < 2).astype(int)
        lambda_base = 0.07
        lambda_ind = (
            lambda_base
            * df["bonus_malus"]
            * (1 + 0.4 * df["jeune"])
            * (1 + 0.2 * df["novice"])
            * df["exposition"]
        )
        df["nb_sinistres"] = np.random.poisson(lambda_ind)
        df["cout_total"] = np.where(
            df["nb_sinistres"] > 0,
            np.random.lognormal(mean=7.8, sigma=0.9, size=n) * df["nb_sinistres"],
            0.0,
        )
        return df

    def statistiques(self):
        print("── Résumé portefeuille ──")
        print(f"Nb contrats    : {self.n:>8,}")
        print(f"Fréquence moy. : {self.df[\'nb_sinistres\'].mean():>8.4f}")
        sin_pos = self.df[self.df["cout_total"] > 0]["cout_total"]
        print(f"Coût moyen     : {sin_pos.mean():>8,.0f} €")
        stats_zone = self.df.groupby("zone").agg(
            nb_contrats=("id", "count"),
            freq_moy=("nb_sinistres", "mean"),
        ).round(4)
        print("\\n── Stats par zone ──")
        print(stats_zone)
        return stats_zone


class Tarification:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.freq = None
        self.cout = None

    def ajuster_frequence(self):
        print("Ajustement GLM Fréquence (Poisson)...")
        self.freq = smf.glm(
            formula="nb_sinistres ~ age + bonus_malus + C(zone) + C(categorie_vh) + jeune",
            data=self.df,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=np.log(self.df["exposition"]),
        ).fit()
        print(f"AIC Fréquence : {self.freq.aic:.2f}")
        return self.freq

    def ajuster_cout(self):
        print("Ajustement GLM Coût (Gamma)...")
        df_sin = self.df[self.df["nb_sinistres"] > 0].copy()
        df_sin["cout_moyen"] = df_sin["cout_total"] / df_sin["nb_sinistres"]
        self.cout = smf.glm(
            formula="cout_moyen ~ age + bonus_malus + C(zone) + C(categorie_vh)",
            data=df_sin,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        ).fit()
        print(f"AIC Coût : {self.cout.aic:.2f}")
        return self.cout

    def calculer_primes(self):
        if self.freq is None:
            self.ajuster_frequence()
        if self.cout is None:
            self.ajuster_cout()
        self.df["freq_pred"] = self.freq.predict(self.df)
        self.df["cout_pred"] = self.cout.predict(self.df)
        self.df["prime_pure"] = self.df["freq_pred"] * self.df["cout_pred"]
        return self.df[["age", "zone", "bonus_malus", "freq_pred", "cout_pred", "prime_pure"]]


class ChainLadder:
    def __init__(self, triangle: np.ndarray):
        self.triangle = triangle.astype(float)
        self.n = triangle.shape[0]
        self.facteurs = None
        self.triangle_complet = None

    def calculer_facteurs(self):
        n = self.n
        facteurs = []
        for j in range(n - 1):
            lignes_ok = (
                ~np.isnan(self.triangle[: n - j - 1, j])
                & ~np.isnan(self.triangle[: n - j - 1, j + 1])
            )
            numerateur = self.triangle[: n - j - 1, j + 1][lignes_ok].sum()
            denominateur = self.triangle[: n - j - 1, j][lignes_ok].sum()
            facteurs.append(numerateur / denominateur)
        self.facteurs = facteurs
        print("Facteurs de développement :")
        for i, f in enumerate(facteurs):
            print(f"  f{i+1} = {f:.4f}")
        return facteurs

    def completer_triangle(self):
        if self.facteurs is None:
            self.calculer_facteurs()
        tri = self.triangle.copy()
        for i in range(1, self.n):
            for j in range(self.n - i, self.n):
                if np.isnan(tri[i, j]):
                    tri[i, j] = tri[i, j - 1] * self.facteurs[j - 1]
        self.triangle_complet = tri
        return tri

    def calculer_ibnr(self):
        if self.triangle_complet is None:
            self.completer_triangle()
        resultats = []
        for i in range(self.n):
            ligne = self.triangle[i, :]
            connus = ligne[~np.isnan(ligne)]
            dernier_connu = connus[-1] if len(connus) > 0 else 0
            ultime = self.triangle_complet[i, -1]
            ibnr = ultime - dernier_connu
            resultats.append({
                "annee_surv": 2020 + i,
                "dernier_connu": round(dernier_connu),
                "ultime": round(ultime),
                "IBNR": round(ibnr),
                "pct_dev": f"{dernier_connu / ultime * 100:.1f}%",
            })
        df_ibnr = pd.DataFrame(resultats)
        print("\\n── Triangle IBNR (Chain-Ladder) ──")
        print(df_ibnr.to_string(index=False))
        print(f"\\nProvision IBNR TOTALE : {df_ibnr[\'IBNR\'].sum():>12,.0f} €")
        return df_ibnr
'''

# ── modules/solvabilite2.py ───────────────────────────────────────────────────
MODULES_SOLVABILITE2 = '''\
import numpy as np
import pandas as pd


class SCR_NonVie:
    SIGMA_PRIMES = {
        "Auto RC": 0.10,
        "Auto Dommages": 0.08,
        "Incendie": 0.14,
        "RC Generale": 0.17,
        "Transport": 0.19,
        "Credit": 0.21,
    }
    SIGMA_RESERVES = {
        "Auto RC": 0.09,
        "Auto Dommages": 0.08,
        "Incendie": 0.11,
        "RC Generale": 0.14,
        "Transport": 0.18,
        "Credit": 0.19,
    }
    LIGNES = list(SIGMA_PRIMES.keys())
    CORR = np.array([
        [1.00, 0.50, 0.25, 0.25, 0.25, 0.25],
        [0.50, 1.00, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0.25, 1.00, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0.25, 1.00, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0.25, 1.00, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0.25, 1.00],
    ])

    def __init__(self, primes: dict, provisions: dict):
        self.primes = primes
        self.provisions = provisions

    def scr_par_ligne(self):
        details = {}
        for lm in self.primes:
            if lm not in self.SIGMA_PRIMES:
                continue
            Vp = self.primes[lm] * 3
            Vr = self.provisions.get(lm, 0)
            scr_p = 3 * self.SIGMA_PRIMES[lm] * Vp
            scr_r = 3 * self.SIGMA_RESERVES[lm] * Vr
            scr_lm = np.sqrt(scr_p ** 2 + scr_r ** 2 + 0.5 * scr_p * scr_r)
            details[lm] = {
                "Primes": self.primes[lm],
                "Provisions": self.provisions.get(lm, 0),
                "SCR_Primes": round(scr_p),
                "SCR_Reserves": round(scr_r),
                "SCR_Total": round(scr_lm),
            }
        return pd.DataFrame(details).T

    def scr_total(self):
        df = self.scr_par_ligne()
        scr_vec = np.array(
            [df.loc[lm, "SCR_Total"] if lm in df.index else 0.0 for lm in self.LIGNES]
        )
        SCR = np.sqrt(scr_vec @ self.CORR @ scr_vec)
        return round(SCR)


class SCR_Vie:
    CHOCS = {
        "mortalite": 0.15,
        "longevite": 0.20,
        "invalidite": 0.35,
        "rachat_up": 0.50,
        "rachat_dn": 0.50,
        "frais": 0.10,
        "catastrophe": 0.0015,
    }
    CORR = np.array([
        [ 1.00, -0.25,  0.25,  0.00,  0.25,  0.25],
        [-0.25,  1.00, -0.25,  0.25, -0.25,  0.00],
        [ 0.25, -0.25,  1.00,  0.00,  0.50,  0.25],
        [ 0.00,  0.25,  0.00,  1.00,  0.00,  0.00],
        [ 0.25, -0.25,  0.50,  0.00,  1.00,  0.25],
        [ 0.25,  0.00,  0.25,  0.00,  0.25,  1.00],
    ])

    def __init__(self, be_deces, be_rentes, be_frais, be_rachat, capital_risque):
        self.be = {
            "mortalite": be_deces,
            "longevite": be_rentes,
            "invalidite": be_frais * 0.5,
            "rachat": be_rachat,
            "frais": be_frais,
            "catastrophe": capital_risque,
        }

    def calculer(self):
        chocs = {
            "mortalite":    self.be["mortalite"]   * self.CHOCS["mortalite"],
            "longevite":    self.be["longevite"]   * self.CHOCS["longevite"],
            "invalidite":   self.be["invalidite"]  * self.CHOCS["invalidite"],
            "rachat":       self.be["rachat"]      * self.CHOCS["rachat_up"],
            "frais":        self.be["frais"]       * self.CHOCS["frais"],
            "catastrophe":  self.be["catastrophe"] * self.CHOCS["catastrophe"],
        }
        v = np.array(list(chocs.values()))
        SCR = np.sqrt(v @ self.CORR @ v)
        print("\\n── SCR Vie détaillé ──")
        for k, val in chocs.items():
            print(f"  SCR {k:<15}: {val:>12,.0f} €")
        print(f"  {\'SCR Vie TOTAL\':<15}: {SCR:>12,.0f} €")
        return round(SCR), chocs


class SolvabiliteII:
    CORR_BSCR = np.array([
        [1.00, 0.00, 0.25],
        [0.00, 1.00, 0.25],
        [0.25, 0.25, 1.00],
    ])

    def __init__(self, scr_nl, scr_vie, scr_marche, fonds_propres, taux_abs=0.15):
        self.scr_nl = scr_nl
        self.scr_vie = scr_vie
        self.scr_marche = scr_marche
        self.fonds_propres = fonds_propres
        self.taux_abs = taux_abs

    def bscr(self):
        v = np.array([self.scr_nl, self.scr_vie, self.scr_marche])
        return round(np.sqrt(v @ self.CORR_BSCR @ v))

    def scr(self):
        return round(self.bscr() * (1 - self.taux_abs))

    def ratio_solvabilite(self):
        return round(self.fonds_propres / self.scr() * 100, 1)

    def rapport(self):
        bscr = self.bscr()
        scr_f = self.scr()
        ratio = self.ratio_solvabilite()
        sep = "=" * 50
        dash = "-" * 40
        print(f"\\n{sep}")
        print(f"{\'RAPPORT SOLVABILITÉ II\':^50}")
        print(f"{sep}")
        print(f"  SCR Non-Vie        : {self.scr_nl:>12,.0f} €")
        print(f"  SCR Vie            : {self.scr_vie:>12,.0f} €")
        print(f"  SCR Marché         : {self.scr_marche:>12,.0f} €")
        print(f"  {dash}")
        print(f"  BSCR               : {bscr:>12,.0f} €")
        print(f"  SCR FINAL          : {scr_f:>12,.0f} €")
        print(f"  Fonds Propres      : {self.fonds_propres:>12,.0f} €")
        print(f"  {dash}")
        emoji = "OK" if ratio >= 100 else "KO"
        print(f"  RATIO SII          : {ratio:>11.1f}% [{emoji}]")
        print(f"{sep}")
        return {"BSCR": bscr, "SCR": scr_f, "Ratio": ratio}
'''

# ── modules/ifrs17.py ─────────────────────────────────────────────────────────
MODULES_IFRS17 = '''\
import numpy as np
import pandas as pd


class GroupeContrats:
    def __init__(self, params: dict):
        self.p = params
        self._valider_params()
        self.v = 1 / (1 + params["taux_actualisation"])
        self.n = params["nb_contrats"]
        self.T = params["duree"]
        self._flux = None
        self._bel = None
        self._ra = None
        self._csm = None

    def _valider_params(self):
        requis = [
            "nb_contrats", "duree", "prime_annuelle", "capital_deces",
            "taux_actualisation", "taux_mortalite", "frais_gestion",
            "frais_acquisition", "cov_risque",
        ]
        for param in requis:
            if param not in self.p:
                raise ValueError(f"Paramètre manquant : {param}")

    def flux_tresorerie(self):
        if self._flux is not None:
            return self._flux
        p = self.p
        survivants = float(self.n)
        records = []
        for t in range(1, self.T + 1):
            deces = survivants * p["taux_mortalite"]
            flux = {
                "annee": t,
                "survivants": survivants,
                "deces": deces,
                "primes": survivants * p["prime_annuelle"],
                "prestations": deces * p["capital_deces"],
                "frais": survivants * p["frais_gestion"],
            }
            flux["flux_net"] = flux["primes"] - flux["prestations"] - flux["frais"]
            flux["flux_actualise"] = flux["flux_net"] * (self.v ** t)
            records.append(flux)
            survivants -= deces
        self._flux = pd.DataFrame(records)
        return self._flux

    def best_estimate(self):
        if self._bel is not None:
            return self._bel
        df = self.flux_tresorerie()
        t_arr = df["annee"].values
        v_arr = self.v ** t_arr
        pv_presta = (df["prestations"].values * v_arr).sum()
        pv_frais  = (df["frais"].values * v_arr).sum()
        pv_primes = (df["primes"].values * v_arr).sum()
        frais_acq = self.n * self.p["frais_acquisition"]
        self._bel = pv_presta + pv_frais + frais_acq - pv_primes
        return self._bel

    def risk_adjustment(self):
        if self._ra is not None:
            return self._ra
        self._ra = abs(self.best_estimate()) * self.p["cov_risque"] * 1.15
        return self._ra

    def csm_initial(self):
        if self._csm is not None:
            return self._csm
        primes_init = self.n * self.p["prime_annuelle"]
        fcf = self.best_estimate() + self.risk_adjustment()
        self._csm = max(primes_init - fcf, 0)
        return self._csm

    def amortissement_csm(self):
        df = self.flux_tresorerie()
        csm = self.csm_initial()
        t_arr = df["annee"].values
        unites = df["survivants"].values * (self.v ** t_arr)
        total = unites.sum()
        taux = unites / total
        releases = csm * taux
        csm_cum = csm - np.cumsum(releases)
        return pd.DataFrame({
            "annee":       t_arr,
            "unites":      np.round(unites, 1),
            "taux_release": np.round(taux * 100, 2),
            "release_CSM": np.round(releases, 2),
            "CSM_restant": np.round(csm_cum, 2),
        })

    def compte_resultat(self):
        df_flux  = self.flux_tresorerie()
        df_amort = self.amortissement_csm()
        bel = self.best_estimate()
        ra  = self.risk_adjustment()
        records = []
        for i in range(min(10, self.T)):
            t = i + 1
            rel_csm = df_amort.loc[i, "release_CSM"]
            rel_ra  = ra / self.T
            charge  = df_flux.loc[i, "prestations"] + df_flux.loc[i, "frais"]
            rev_ass = charge + rel_csm + rel_ra
            res_tech = rel_csm + rel_ra
            prod_fi  = abs(bel) * self.p["taux_actualisation"]
            records.append({
                "Année":              t,
                "Revenus assurance":  round(rev_ass),
                "Charges assurance":  round(charge),
                "Release CSM":        round(rel_csm),
                "Release RA":         round(rel_ra),
                "Résultat technique": round(res_tech),
                "Produits financiers":round(prod_fi),
                "Résultat net":       round(res_tech + prod_fi),
            })
        return pd.DataFrame(records)

    def bilan_ifrs17(self):
        bel = self.best_estimate()
        ra  = self.risk_adjustment()
        csm = self.csm_initial()
        sep  = "=" * 50
        dash = "-" * 38
        print(f"\\n{sep}")
        print(f"{\'PASSIF IFRS 17 — GMM\':^50}")
        print(f"{sep}")
        print(f"  Best Estimate (BEL)  : {bel:>12,.0f} €")
        print(f"  Risk Adjustment (RA) : {ra:>12,.0f} €")
        print(f"  CSM                  : {csm:>12,.0f} €")
        print(f"  {dash}")
        print(f"  TOTAL PASSIF         : {bel+ra+csm:>12,.0f} €")
        print(f"{sep}")
        return {"BEL": bel, "RA": ra, "CSM": csm, "Total": bel + ra + csm}
'''

# ── main.py ───────────────────────────────────────────────────────────────────
MAIN_PY = '''\
from modules.vie import TableMortalite, AssuranceVie
from modules.non_vie import PortefeuilleAuto, ChainLadder
from modules.solvabilite2 import SCR_NonVie, SCR_Vie, SolvabiliteII
from modules.ifrs17 import GroupeContrats
import numpy as np


def main():
    print("=" * 60)
    print("   PROJET ACTUARIEL PYTHON — EXECUTION COMPLETE")
    print("=" * 60)

    # ── MODULE VIE ────────────────────────────────────────────
    print("\\n[VIE]")
    table = TableMortalite()
    av = AssuranceVie(table, taux_tech=0.025)
    prime = av.prime_nette(age=45, duree=20, capital=100_000)
    print(f"Prime nette (45 ans, 20 ans, 100k€) : {prime:,.2f} €")
    print(f"Espérance de vie résiduelle à 45 ans : {av.annuite_viagere(45):.1f} ans")

    # ── MODULE NON-VIE ────────────────────────────────────────
    print("\\n[NON-VIE]")
    ptf = PortefeuilleAuto(n_contrats=5_000)
    ptf.statistiques()
    triangle = np.array([
        [4_500_000, 6_750_000, 7_650_000, 7_875_000],
        [4_950_000, 7_425_000, 8_415_000,       np.nan],
        [5_400_000, 8_100_000,      np.nan,      np.nan],
        [5_850_000,      np.nan,    np.nan,      np.nan],
    ])
    cl = ChainLadder(triangle)
    cl.calculer_ibnr()

    # ── MODULE SOLVABILITÉ II ─────────────────────────────────
    print("\\n[SOLVABILITE II]")
    scr_nl = SCR_NonVie(
        {"Auto RC": 5e6, "Auto Dommages": 3e6, "Incendie": 2e6},
        {"Auto RC": 8e6, "Auto Dommages": 4e6, "Incendie": 3.5e6},
    ).scr_total()
    scr_v, _ = SCR_Vie(15e6, 20e6, 5e6, 20e6, 100e6).calculer()
    sii = SolvabiliteII(scr_nl, scr_v, 3.5e6, 50e6)
    sii.rapport()

    # ── MODULE IFRS 17 ────────────────────────────────────────
    print("\\n[IFRS 17]")
    gc = GroupeContrats({
        "nb_contrats": 1_000,
        "duree": 20,
        "prime_annuelle": 1_380,
        "capital_deces": 100_000,
        "taux_actualisation": 0.025,
        "taux_mortalite": 0.005,
        "frais_gestion": 80,
        "frais_acquisition": 150,
        "cov_risque": 0.05,
    })
    gc.bilan_ifrs17()

    print("\\nCalculs terminés !")
    print("Lance le dashboard avec : streamlit run dashboard.py")


if __name__ == "__main__":
    main()
'''

# ── dashboard.py ──────────────────────────────────────────────────────────────
DASHBOARD_PY = '''\
"""
dashboard.py — Dashboard Streamlit du projet actuariel.
Lancement : streamlit run dashboard.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.vie import TableMortalite, AssuranceVie
from modules.non_vie import PortefeuilleAuto, ChainLadder, Tarification
from modules.solvabilite2 import SCR_NonVie, SCR_Vie, SolvabiliteII
from modules.ifrs17 import GroupeContrats

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Projet Actuariel",
    page_icon="📊",
    layout="wide",
)
st.title("📊 Projet Actuariel — Dashboard Interactif")

tab_vie, tab_nv, tab_sii, tab_ifrs = st.tabs(
    ["🔵 Vie", "🟠 Non-Vie", "🟢 Solvabilité II", "🟣 IFRS 17"]
)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — VIE
# ═══════════════════════════════════════════════════════════════════════════
with tab_vie:
    st.header("Assurance Vie — Table de mortalité & provisionnement")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Paramètres")
        age       = st.slider("Âge assuré",        20, 70, 45)
        duree     = st.slider("Durée du contrat",   5, 40, 20)
        capital   = st.number_input("Capital décès (€)", 10_000, 1_000_000, 100_000, 10_000)
        taux_tech = st.slider("Taux technique (%)", 0.5, 5.0, 2.5, 0.1) / 100

    table = TableMortalite()
    av    = AssuranceVie(table, taux_tech=taux_tech)

    prime = av.prime_nette(age=age, duree=duree, capital=capital)
    ev    = av.annuite_viagere(age)

    with col1:
        st.metric("Prime nette annuelle", f"{prime:,.0f} €")
        st.metric("Espérance de vie résiduelle", f"{ev:.1f} ans")

    with col2:
        df_prov = av.provision_mathematique(age, duree, capital)
        fig = px.area(
            df_prov, x="annee", y="provision",
            title=f"Provision mathématique — Assuré {age} ans, contrat {duree} ans",
            labels={"annee": "Année", "provision": "Provision (€)"},
            color_discrete_sequence=["#1f77b4"],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Table de mortalité (extrait)")
    age_min_t = st.slider("Âge min table", 0, 80, 30, key="tmin")
    age_max_t = st.slider("Âge max table", age_min_t + 5, 120, 80, key="tmax")
    df_table  = table.afficher_table(age_min_t, age_max_t)

    fig2 = px.line(
        df_table, x="age", y="qx",
        title="Taux de mortalité qx",
        labels={"age": "Âge", "qx": "qx"},
        color_discrete_sequence=["#d62728"],
    )
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — NON-VIE
# ═══════════════════════════════════════════════════════════════════════════
with tab_nv:
    st.header("Non-Vie — Portefeuille Auto & Chain-Ladder")

    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Portefeuille")
        n_contrats = st.number_input("Nb contrats", 1_000, 50_000, 10_000, 1_000)
        seed       = st.number_input("Seed aléatoire", 0, 999, 42)

    @st.cache_data
    def get_portefeuille(n, s):
        return PortefeuilleAuto(n_contrats=n, seed=s)

    ptf = get_portefeuille(int(n_contrats), int(seed))
    df  = ptf.df

    with col2:
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Nb contrats",   f"{len(df):,}")
        col_b.metric("Fréq. moyenne", f"{df[\'nb_sinistres\'].mean():.4f}")
        sin_pos = df[df["cout_total"] > 0]["cout_total"]
        col_c.metric("Coût moyen",    f"{sin_pos.mean():,.0f} €")

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(
            df, x="nb_sinistres", nbins=10,
            title="Distribution du nombre de sinistres",
            labels={"nb_sinistres": "Nb sinistres", "count": "Effectif"},
            color_discrete_sequence=["#ff7f0e"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
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

    # Chain-Ladder
    st.subheader("Triangle Chain-Ladder")
    triangle_def = np.array([
        [4_500_000, 6_750_000, 7_650_000, 7_875_000],
        [4_950_000, 7_425_000, 8_415_000, np.nan],
        [5_400_000, 8_100_000, np.nan,    np.nan],
        [5_850_000, np.nan,    np.nan,    np.nan],
    ])
    cl     = ChainLadder(triangle_def)
    df_ibnr = cl.calculer_ibnr()

    fig5 = px.bar(
        df_ibnr, x="annee_surv", y="IBNR",
        title="Provision IBNR par année de survenance",
        labels={"annee_surv": "Année survenance", "IBNR": "IBNR (€)"},
        color_discrete_sequence=["#2ca02c"],
        text_auto=True,
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(df_ibnr, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — SOLVABILITÉ II
# ═══════════════════════════════════════════════════════════════════════════
with tab_sii:
    st.header("Solvabilité II — SCR & Ratio de couverture")

    st.subheader("SCR Non-Vie")
    col1, col2, col3 = st.columns(3)
    with col1:
        p_auto_rc  = st.number_input("Primes Auto RC (M€)",  0.1, 50.0, 5.0, 0.5) * 1e6
        p_auto_do  = st.number_input("Primes Auto Dom (M€)", 0.1, 50.0, 3.0, 0.5) * 1e6
        p_incendie = st.number_input("Primes Incendie (M€)", 0.1, 50.0, 2.0, 0.5) * 1e6
    with col2:
        r_auto_rc  = st.number_input("Prov. Auto RC (M€)",  0.1, 100.0, 8.0, 0.5) * 1e6
        r_auto_do  = st.number_input("Prov. Auto Dom (M€)", 0.1, 100.0, 4.0, 0.5) * 1e6
        r_incendie = st.number_input("Prov. Incendie (M€)", 0.1, 100.0, 3.5, 0.5) * 1e6
    with col3:
        be_deces     = st.number_input("BE Décès (M€)",   1.0, 200.0, 15.0, 1.0) * 1e6
        be_rentes    = st.number_input("BE Rentes (M€)",  1.0, 200.0, 20.0, 1.0) * 1e6
        fonds_propres= st.number_input("Fonds propres (M€)", 1.0, 500.0, 50.0, 1.0) * 1e6

    scr_nl_obj = SCR_NonVie(
        {"Auto RC": p_auto_rc, "Auto Dommages": p_auto_do, "Incendie": p_incendie},
        {"Auto RC": r_auto_rc, "Auto Dommages": r_auto_do, "Incendie": r_incendie},
    )
    df_scr_ligne = scr_nl_obj.scr_par_ligne().reset_index()
    df_scr_ligne.columns = ["Ligne"] + list(df_scr_ligne.columns[1:])

    fig6 = px.bar(
        df_scr_ligne, x="Ligne", y=["SCR_Primes", "SCR_Reserves"],
        barmode="group",
        title="SCR par ligne et composante",
        labels={"value": "SCR (€)", "variable": "Composante"},
    )
    st.plotly_chart(fig6, use_container_width=True)

    scr_nl = scr_nl_obj.scr_total()
    scr_v, chocs = SCR_Vie(be_deces, be_rentes, 5e6, 20e6, 100e6).calculer()
    scr_marche   = 3.5e6
    sii = SolvabiliteII(scr_nl, scr_v, scr_marche, fonds_propres)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("SCR Non-Vie",  f"{scr_nl/1e6:.2f} M€")
    col_m2.metric("SCR Vie",      f"{scr_v/1e6:.2f} M€")
    col_m3.metric("SCR Final",    f"{sii.scr()/1e6:.2f} M€")
    ratio = sii.ratio_solvabilite()
    col_m4.metric("Ratio SII",    f"{ratio:.1f}%", delta=f"{ratio-100:.1f}%")

    # Gauge
    fig7 = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=ratio,
        delta={"reference": 100},
        gauge={
            "axis": {"range": [0, 300]},
            "bar":  {"color": "darkblue"},
            "steps": [
                {"range": [0,   100], "color": "#ff4b4b"},
                {"range": [100, 200], "color": "#ffd700"},
                {"range": [200, 300], "color": "#00cc44"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 100},
        },
        title={"text": "Ratio de Solvabilité (%)"},
    ))
    st.plotly_chart(fig7, use_container_width=True)

    # Waterfall BSCR
    fig8 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["SCR Non-Vie", "SCR Vie", "SCR Marché", "BSCR"],
        y=[scr_nl / 1e6, scr_v / 1e6, scr_marche / 1e6, sii.bscr() / 1e6],
        connector={"line": {"color": "rgb(63,63,63)"}},
    ))
    fig8.update_layout(title="Décomposition BSCR (M€)", yaxis_title="M€")
    st.plotly_chart(fig8, use_container_width=True)

    # SCR Vie détail
    df_chocs = pd.DataFrame(
        [{"Risque": k, "SCR (€)": round(v)} for k, v in chocs.items()]
    )
    fig9 = px.pie(
        df_chocs, values="SCR (€)", names="Risque",
        title="Répartition SCR Vie par risque",
    )
    st.plotly_chart(fig9, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — IFRS 17
# ═══════════════════════════════════════════════════════════════════════════
with tab_ifrs:
    st.header("IFRS 17 — Modèle GMM (General Measurement Model)")

    col1, col2 = st.columns(2)
    with col1:
        nb_contrats = st.number_input("Nb contrats",   100, 10_000, 1_000, 100)
        duree_ifrs  = st.slider("Durée (ans)",          5,      40,    20)
        prime_ann   = st.number_input("Prime annuelle (€)", 100, 10_000, 1_380, 100)
        capital_d   = st.number_input("Capital décès (€)",  10_000, 500_000, 100_000, 10_000)
    with col2:
        taux_act    = st.slider("Taux actualisation (%)", 0.5, 8.0, 2.5, 0.1) / 100
        taux_mort   = st.slider("Taux mortalité (%)",     0.1, 5.0, 0.5, 0.1) / 100
        frais_gest  = st.number_input("Frais de gestion (€/an)", 0, 1_000, 80, 10)
        frais_acq   = st.number_input("Frais acquisition (€)",   0, 1_000, 150, 10)
        cov_risque  = st.slider("CoV risque (%)", 1, 20, 5) / 100

    gc = GroupeContrats({
        "nb_contrats": int(nb_contrats),
        "duree": int(duree_ifrs),
        "prime_annuelle": float(prime_ann),
        "capital_deces": float(capital_d),
        "taux_actualisation": float(taux_act),
        "taux_mortalite": float(taux_mort),
        "frais_gestion": float(frais_gest),
        "frais_acquisition": float(frais_acq),
        "cov_risque": float(cov_risque),
    })

    bel = gc.best_estimate()
    ra  = gc.risk_adjustment()
    csm = gc.csm_initial()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Best Estimate (BEL)", f"{bel:,.0f} €")
    col_b.metric("Risk Adjustment",     f"{ra:,.0f} €")
    col_c.metric("CSM initial",         f"{csm:,.0f} €")
    col_d.metric("Total Passif IFRS17", f"{bel+ra+csm:,.0f} €")

    # Bilan waterfall
    fig10 = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative", "relative", "relative", "total"],
        x=["BEL", "RA", "CSM", "Passif Total"],
        y=[bel, ra, csm, bel + ra + csm],
        connector={"line": {"color": "rgb(63,63,63)"}},
    ))
    fig10.update_layout(title="Passif IFRS 17 — GMM (€)", yaxis_title="€")
    st.plotly_chart(fig10, use_container_width=True)

    # Amortissement CSM
    df_csm = gc.amortissement_csm()
    fig11 = px.bar(
        df_csm.head(min(10, len(df_csm))),
        x="annee", y="release_CSM",
        title="Amortissement du CSM (10 premières années)",
        labels={"annee": "Année", "release_CSM": "Release CSM (€)"},
        color_discrete_sequence=["#9467bd"],
    )
    st.plotly_chart(fig11, use_container_width=True)

    # Compte de résultat
    st.subheader("Compte de résultat IFRS 17")
    df_cr = gc.compte_resultat()
    fig12 = px.bar(
        df_cr, x="Année",
        y=["Revenus assurance", "Charges assurance"],
        barmode="overlay",
        title="Revenus vs Charges d\'assurance",
        labels={"value": "€", "variable": "Poste"},
    )
    st.plotly_chart(fig12, use_container_width=True)

    fig13 = px.line(
        df_cr, x="Année", y="Résultat net",
        title="Résultat net IFRS 17",
        markers=True,
        color_discrete_sequence=["#17becf"],
    )
    st.plotly_chart(fig13, use_container_width=True)

    st.subheader("Tableau — Compte de résultat détaillé")
    st.dataframe(df_cr.set_index("Année"), use_container_width=True)

    # Flux de trésorerie
    st.subheader("Flux de trésorerie actualisés")
    df_flux = gc.flux_tresorerie()
    fig14 = make_subplots(rows=1, cols=2, subplot_titles=["Flux bruts", "Flux nets actualisés"])
    fig14.add_trace(go.Bar(name="Primes",      x=df_flux["annee"], y=df_flux["primes"],      marker_color="#2ca02c"), row=1, col=1)
    fig14.add_trace(go.Bar(name="Prestations", x=df_flux["annee"], y=df_flux["prestations"], marker_color="#d62728"), row=1, col=1)
    fig14.add_trace(go.Scatter(name="Flux net actualisé", x=df_flux["annee"], y=df_flux["flux_actualise"], mode="lines+markers", marker_color="#1f77b4"), row=1, col=2)
    fig14.update_layout(title_text="Flux de trésorerie IFRS 17", barmode="group")
    st.plotly_chart(fig14, use_container_width=True)
'''

# ─────────────────────────────────────────────────────────────────────────────
# Main — create the file tree
# ─────────────────────────────────────────────────────────────────────────────

def main():
    base = Path(".")

    files = {
        base / "modules" / "__init__.py":      MODULES_INIT,
        base / "modules" / "vie.py":           MODULES_VIE,
        base / "modules" / "non_vie.py":       MODULES_NON_VIE,
        base / "modules" / "solvabilite2.py":  MODULES_SOLVABILITE2,
        base / "modules" / "ifrs17.py":        MODULES_IFRS17,
        base / "main.py":                      MAIN_PY,
        base / "dashboard.py":                 DASHBOARD_PY,
    }

    print("Création du projet actuariel…\n")
    for path, content in files.items():
        write_file(path, content)

    print("\n[DONE] Projet cree avec succes! Lancez: python main.py")
    print("       Pour le dashboard : streamlit run dashboard.py")


if __name__ == "__main__":
    main()
