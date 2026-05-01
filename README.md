# Optimal Transport for Financial Time Series Generation

Génération de séries temporelles financières synthétiques par trois approches issues du **transport optimal** et de ses extensions, comparées sur des log-returns journaliers du S&P 500 (top 10 capitalisations).

Trois modèles sont implémentés et benchmarkés sur le même jeu de données rescalé selon le protocole de Huang et al. (2024) — *Schrödinger Bridges for Time Series Generation* :

- **SBTS** — Schrödinger Bridge Time Series, versions non-markovienne (`K=0`) et markovienne (`K=1`)
- **Flow Matching** — entraînement *Conditional Flow Matching* sur un couplage Gaussien → données
- **ICNN** — générateur de Monge basé sur un *Input-Convex Neural Network* (carte de Brenier paramétrée)

Toutes les méthodes sont évaluées sur le même espace de log-returns originaux (les sorties sont *unrescalées* après simulation) afin que la comparaison soit honnête.

---

## Structure du repo

```
.
├── data/                       # données de marché (CSV S&P 500 top 10)
├── models/                     # implémentations des trois modèles
│   ├── sbts_multi.py           # SBTS K=0 (non-markovien)
│   ├── sbts_multi_markov.py    # SBTS K=1 (markovien)
│   ├── flow_matching.py        # Flow Matching (Lipman et al., 2023)
│   └── icnn.py                 # ICNN Monge generator + losses
├── metrics/
│   └── eval_functions.py       # statistiques, scores discriminatif & prédictif, plots
├── SB_main.py                  # pipeline SBTS (sweep h pour K=0 et K=1)
├── FM_main.py                  # pipeline Flow Matching
├── ICNN_main.py                # pipeline ICNN (CLI argparse complet)
├── analyse.ipynb               # comparaison croisée des trois modèles
├── requirements.txt
├── X_synth_*.npy               # sorties générées (returns + prices)
├── fm_sample.png               # exemples de trajectoires générées
├── icnn_sample.png
└── sbts_sample.png
```

---

## Données

Fichier d'entrée : `data/sp500_top10_prices.csv` — prix journaliers de 10 actions du S&P 500.


## Analyse comparative

```bash
jupyter notebook analyse.ipynb
```

Le notebook charge les `.npy` générés par les trois pipelines et produit les visualisations + tableaux comparatifs.


## Référence

Huang, Z., Henry-Labordère, P., et al. *Schrödinger Bridges for Time Series Generation*, 2024 et le GitHub associé

---
