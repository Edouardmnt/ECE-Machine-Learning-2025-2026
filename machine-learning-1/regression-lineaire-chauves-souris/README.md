# Régression linéaire : masse du cerveau et masse corporelle des chauves-souris

**▶ Essayer en ligne :** [huggingface.co/spaces/edouardmnt04/regression-chauves-souris](https://huggingface.co/spaces/edouardmnt04/regression-chauves-souris) (environ 1 minute de chargement au premier lancement)

Application **Streamlit** qui modélise la masse du cerveau (BRW, en mg) en fonction de la masse corporelle (BOW, en g) pour 29 espèces de chauves-souris.

Réalisé en binôme par Chloé Lestic et Édouard Menut (Machine Learning 1, ECE Paris, 2025/2026).

![Droites de régression avec et sans l'espèce atypique](images/droites_regression.png)

## Contenu

1. **Exploration** : aperçu, types et statistiques descriptives du jeu de données.
2. **Régression linéaire simple** (statsmodels OLS) : coefficients, p-values, R², AIC/BIC, graphe des résidus et QQ-plot.
3. **Point influent** : *Pteropus vampyrus* pèse plus de 1 kg, près de 4 fois plus que l'espèce suivante. On compare le modèle avec et sans cette espèce.

## Résultats

| | Pente (mg de cerveau par g) | R² |
|---|---|---|
| Toutes les espèces | 9,0 | 0,950 |
| Sans *Pteropus vampyrus* | 14,5 | 0,978 |

Un seul individu à fort effet de levier suffit à sous-estimer la pente de 38 %. Il faut toujours contrôler les points influents avant d'interpréter une régression.

## Lancer l'application

```bash
streamlit run app.py
```

Le fichier `data/tabBats.txt` est chargé automatiquement. Un autre fichier au même format peut être chargé depuis l'interface.
