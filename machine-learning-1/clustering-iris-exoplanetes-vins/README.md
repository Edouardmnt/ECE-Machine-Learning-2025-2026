# Clustering : Iris, exoplanètes et vins

**▶ Essayer en ligne :** [huggingface.co/spaces/edouardmnt04/clustering-iris-exoplanetes-vins](https://huggingface.co/spaces/edouardmnt04/clustering-iris-exoplanetes-vins) (environ 1 minute de chargement au premier lancement)

Application **Streamlit** en quatre onglets autour de l'apprentissage non supervisé, puis d'un pipeline complet de data science.

Réalisé en binôme par Chloé Lestic et Édouard Menut (Machine Learning 1, ECE Paris, 2025/2026).

![K-Means sur la projection ACP des iris](images/kmeans_iris.png)

## Contenu et résultats

| Onglet | Méthodes | Résultat |
|---|---|---|
| 1. K-Means (Iris) | Standardisation, ACP (95 % de variance sur 2 axes), K-Means k = 3, table de contingence | Silhouette 0,51. *Setosa* est parfaitement isolée, *versicolor* et *virginica* se chevauchent |
| 2. Clustering hiérarchique (Iris) | CAH, dendrogrammes, linkage complet vs moyen | Le linkage moyen l'emporte (silhouette 0,48 contre 0,45) |
| 3. Nombre optimal de clusters (exoplanètes) | K-Means de k = 2 à 10, indices Calinski-Harabasz et Davies-Bouldin | Les deux indices s'accordent sur k = 7, plus que les 5 types étiquetés |
| 4. Pipeline Wine Quality | Nettoyage robuste d'un CSV bruité (valeurs manquantes ajoutées volontairement), imputation, ACP, K-Means / DBSCAN, KNN / arbre / régression logistique | Meilleur F1 pondéré = 0,76 (arbre de décision, bon vin si note ≥ 6) |

## Lancer l'application

```bash
streamlit run app.py
```

Les données sont dans `data/`. L'onglet 4 accepte aussi n'importe quel CSV séparé par `;` avec une colonne `quality`.
