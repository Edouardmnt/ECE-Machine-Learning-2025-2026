# Machine Learning – ECE Paris (2025/2026)

Travaux pratiques réalisés pendant l'année 2025/2026 à l'**ECE Paris** (majeure Data & IA), dans le cadre des modules **Machine Learning 1** et **Machine Learning 2**.

Auteur : **Édouard Menut** (certains TP en binôme avec Chloé Lestic)

## Structure

```
.
├── ML1/                          # Machine Learning 1
│   ├── TP2_Regression/           # Exploration & régressions (statsmodels) – app Streamlit
│   ├── TP4_Clustering/           # K-Means, CAH, nombre optimal de clusters – app Streamlit + rapport
│   ├── TP_SVM/                   # Support Vector Machines (breast cancer, spambase, car, mushroom)
│   └── TP_Naive_Bayes/           # Classification de texte (Rotten Tomatoes) avec Naive Bayes
└── ML2/                          # Machine Learning 2
    ├── TP2_Neural_Networks/      # MLP avec Keras + rétropropagation codée en NumPy
    ├── TP3_RNN/                  # Réseaux récurrents : génération de texte & analyse de sentiments
    └── TP_Note_Regression_Keras/ # TP noté : problème de régression avec Keras
```

## Machine Learning 1

| TP | Sujet | Notions clés | Outils |
|----|-------|--------------|--------|
| [TP2](ML1/TP2_Regression) | Exploration & régressions (données chauves-souris) | Statistiques descriptives, régression linéaire simple/multiple | pandas, statsmodels, Streamlit |
| [TP4](ML1/TP4_Clustering) | Clustering sur Iris, exoplanètes et Wine Quality | ACP, K-Means, clustering hiérarchique, silhouette / Calinski-Harabasz / Davies-Bouldin | scikit-learn, scipy, Streamlit |
| [SVM](ML1/TP_SVM) | Classification avec SVM sur 4 jeux de données | Noyaux linéaire/RBF/poly, normalisation, sélection de variables (chi², mutual info), GridSearchCV | scikit-learn |
| [Naive Bayes](ML1/TP_Naive_Bayes) | Classification de critiques de films | Bag-of-words, Multinomial NB, lissage de Laplace, validation croisée, TF-IDF, n-grammes | scikit-learn |

## Machine Learning 2

| TP | Sujet | Notions clés | Outils |
|----|-------|--------------|--------|
| [TP2.1](ML2/TP2_Neural_Networks/TP2_1_MLP_Keras.ipynb) | Entraîner un MLP (Digits) | Couches denses, fonctions d'activation, optimiseurs | TensorFlow / Keras |
| [TP2.2](ML2/TP2_Neural_Networks/TP2_2_Backpropagation_Numpy.ipynb) | Rétropropagation « à la main » | Descente de gradient, softmax, cross-entropy | NumPy |
| [TP3](ML2/TP3_RNN) | Réseaux de neurones récurrents | Génération de texte caractère par caractère (Shakespeare), classification de sentiments | TensorFlow / Keras |
| [TP noté](ML2/TP_Note_Regression_Keras) | Problème de régression | Préparation des données, réseau de neurones pour la régression, évaluation | TensorFlow / Keras |

## Lancer le code

```bash
pip install -r requirements.txt

# Notebooks
jupyter notebook

# Applications Streamlit (ML1)
streamlit run ML1/TP2_Regression/TP2.py
streamlit run ML1/TP4_Clustering/TP4.py
```

> Pour le TP SVM, placer les fichiers CSV de `ML1/TP_SVM/data/` à côté du notebook (ou adapter les chemins).
> Certains jeux de données (Rotten Tomatoes, Shakespeare, MovieReview, breast cancer) ne sont pas inclus et étaient fournis par l'école.
