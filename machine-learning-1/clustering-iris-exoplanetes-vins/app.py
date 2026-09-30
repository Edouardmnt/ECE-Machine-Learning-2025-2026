import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Clustering : Iris, exoplanètes et vins", layout="centered")
st.title("Clustering et pipeline de data science")
st.caption("K-Means, clustering hiérarchique et choix du nombre de clusters (Iris, exoplanètes), puis pipeline complet sur Wine Quality.")

# Chargement du dataset Iris
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
labels = np.array([iris.target_names[i] for i in iris.target])

# Standardisation
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(data_scaled)


tabs = st.tabs([
    "1. K-Means (Iris)",
    "2. Clustering hiérarchique (Iris)",
    "3. Nombre optimal de clusters (exoplanètes)",
    "4. Pipeline complet (Wine Quality)"
])


#  PARTIE 1 K-MEANS

with tabs[0]:
    st.header("Iris : K-Means (k = 3) sur la projection PCA vs les données originales")

    colA, colB = st.columns(2)
    with colA:
        jeu = st.radio("Jeu de données", ["Projection PCA (X)", "Données originales (data)"], index=0)
    with colB:
        rs_opt = st.selectbox("Initialisation K-Means", ["Alea (None)", "Fixe (42)"], index=0)

    k = 3
    random_state = None if rs_opt == "Alea (None)" else 42

    if jeu == "Projection PCA (X)":
        Z = X_pca
        titre = "K-Means sur X (PCA 2D)"
    else:
        Z = data.values
        titre = "K-Means sur données originales (4D)"

    if st.button("Relancer l'initialisation"):
        pass

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    clusters = kmeans.fit_predict(Z)

    st.subheader("Visualisation des clusters + centroïdes")
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    if Z.shape[1] == 2:
        ax1.scatter(Z[:, 0], Z[:, 1], c=clusters, alpha=0.7)
        if kmeans.cluster_centers_.shape[1] == 2:
            ax1.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                        marker="X", s=160, c='black')
        ax1.set_xlabel("Dim 1")
        ax1.set_ylabel("Dim 2")
    else:
        Zp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Z))
        ax1.scatter(Zp[:, 0], Zp[:, 1], c=clusters, alpha=0.7)
        C = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(kmeans.cluster_centers_))
        ax1.scatter(C[:, 0], C[:, 1], marker="X", s=160, c='black')
        ax1.set_xlabel("PCA visuelle 1")
        ax1.set_ylabel("PCA visuelle 2")
    ax1.set_title(titre)
    st.pyplot(fig1)

    st.subheader("Visualisation des labels réels")
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    palette = {name: i for i, name in enumerate(np.unique(labels))}
    if Z.shape[1] == 2:
        ax2.scatter(Z[:, 0], Z[:, 1], c=[palette[l] for l in labels], alpha=0.7)
    else:
        Zp = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Z))
        ax2.scatter(Zp[:, 0], Zp[:, 1], c=[palette[l] for l in labels], alpha=0.7)
    ax2.set_title("Répartition par espèce réelle")
    ax2.set_xlabel("PCA visuelle 1")
    ax2.set_ylabel("PCA visuelle 2")
    st.pyplot(fig2)

    st.subheader("Tableau de contingence Clusters vs Espèces")
    ct = pd.crosstab(pd.Series(clusters, name="Cluster"), pd.Series(labels, name="Espèce"))
    st.dataframe(ct)

    sil = silhouette_score(Z, clusters)
    st.metric("Indice de silhouette", f"{sil:.3f}")

    st.divider()
    st.markdown("""
    **Interprétation**
    - La PCA à 2 composantes conserve ~95 % de la variance : les clusters y sont très lisibles.
    - *Setosa* est parfaitement isolée ; *versicolor* et *virginica*, morphologiquement proches, se chevauchent partiellement.
    - En initialisation aléatoire, « Relancer l'initialisation » montre que la partition varie légèrement selon les centroïdes de départ : fixer la graine rend les résultats reproductibles.
    """)

#  PARTIE 2 – CLUSTERING HIÉRARCHIQUE
with tabs[1]:
    st.header("Clustering hiérarchique ascendant sur les données Iris")

    st.markdown("""
    On effectue un **clustering hiérarchique ascendant** sur les données brutes (sans labels).  
    On compare deux stratégies de liaison :  
    - **Linkage complet** (complete)  
    - **Linkage moyen** (average)
    """)

    # Linkage complet
    st.subheader("Linkage complet")
    Z_complete = linkage(data_scaled, method="complete")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    dendrogram(Z_complete, truncate_mode="level", p=5, ax=ax1)
    ax1.set_title("Dendrogramme (linkage = complete)")
    ax1.set_xlabel("Échantillons regroupés")
    ax1.set_ylabel("Distance")
    st.pyplot(fig1)

    cluster_complete = AgglomerativeClustering(n_clusters=3, linkage="complete")
    labels_complete = cluster_complete.fit_predict(data_scaled)

    st.write("**Tableau de contingence (linkage complet)**")
    tab_complete = pd.crosstab(pd.Series(labels_complete, name="Cluster"),
                               pd.Series(labels, name="Espèce réelle"))
    st.dataframe(tab_complete)

    sil_complete = silhouette_score(data_scaled, labels_complete)
    st.metric("Indice de silhouette (linkage complet)", f"{sil_complete:.3f}")

    #  Linkage moyen
    st.subheader("Linkage moyen")
    Z_average = linkage(data_scaled, method="average")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    dendrogram(Z_average, truncate_mode="level", p=5, ax=ax2)
    ax2.set_title("Dendrogramme (linkage = average)")
    ax2.set_xlabel("Échantillons regroupés")
    ax2.set_ylabel("Distance")
    st.pyplot(fig2)

    cluster_average = AgglomerativeClustering(n_clusters=3, linkage="average")
    labels_average = cluster_average.fit_predict(data_scaled)

    st.write("**Tableau de contingence (linkage moyen)**")
    tab_average = pd.crosstab(pd.Series(labels_average, name="Cluster"),
                              pd.Series(labels, name="Espèce réelle"))
    st.dataframe(tab_average)

    sil_average = silhouette_score(data_scaled, labels_average)
    st.metric("Indice de silhouette (linkage moyen)", f"{sil_average:.3f}")

    #  Comparaison
    st.subheader("Comparaison des deux méthodes")
    st.write(f"""
    - Linkage complet : **silhouette = {sil_complete:.3f}**  
    - Linkage moyen   : **silhouette = {sil_average:.3f}**
    """)

    if sil_complete > sil_average:
        st.success("➡ Le linkage complet produit une meilleure cohésion inter-clusters selon l'indice de silhouette.")
    elif sil_complete < sil_average:
        st.success("➡ Le linkage moyen donne un regroupement plus homogène selon l'indice de silhouette.")
    else:
        st.info("Les deux méthodes offrent des performances similaires.")

    st.divider()
    st.markdown("""
    **Interprétation** :  
    - Le dendrogramme montre comment les échantillons se regroupent progressivement.  
    - Une coupe à 3 branches correspond à la création de 3 clusters (les 3 espèces d’Iris).  
    - Le tableau de contingence montre la correspondance entre clusters et espèces réelles.  
    - L’indice de silhouette confirme la qualité globale de chaque partition.
    """)


#  PARTIE 3  NOMBRE OPTIMAL DE CLUSTERS (planete.csv)
with tabs[2]:
    st.header("Nombre optimal de clusters sur des compositions atmosphériques d’exoplanètes")

    st.markdown("""
    Dans cet exercice, on cherche à **déterminer le nombre optimal de clusters** pour le jeu de données
    `planete.csv`, représentant des compositions atmosphériques d’exoplanètes.  
    On utilise deux indices :
    - **Calinski–Harabasz (CH)** : grand = bonne séparation entre clusters et forte cohésion interne.  
    - **Davies–Bouldin (DB)** : petit = meilleure compacité et séparation.  
    """)


    try:
        df_planete = pd.read_csv(DATA_DIR / "planete.csv", sep=";")
    except Exception as e:
        st.error("Impossible de charger data/planete.csv.")
        st.stop()

    # Retrait de la colonne des labels
    if "Type" in df_planete.columns:
        X_planete = df_planete.drop(columns=["Type"]).values
    else:
        X_planete = df_planete.values

    st.subheader("Aperçu des données")
    st.dataframe(df_planete.head())

    #  Conversion en float (sécurité)
    X_planete = X_planete.astype(float)

    #  Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_planete)

    #  Calcul des indices pour K=2..10
    ch_scores = []
    db_scores = []
    k_values = range(2, 11)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled)
        labels_k = kmeans.labels_
        ch_scores.append(calinski_harabasz_score(X_scaled, labels_k))
        db_scores.append(davies_bouldin_score(X_scaled, labels_k))

    #  Affichage des courbes
    st.subheader("Méthode du coude : évolution des indices")

    fig3, ax3 = plt.subplots(1, 2, figsize=(12, 5))

    ax3[0].plot(k_values, ch_scores, marker='o')
    ax3[0].set_title("Indice de Calinski–Harabasz (CH)")
    ax3[0].set_xlabel("Nombre de clusters (k)")
    ax3[0].set_ylabel("Score CH")
    ax3[0].grid(True)

    ax3[1].plot(k_values, db_scores, marker='o', color='orange')
    ax3[1].set_title("Indice de Davies–Bouldin (DB)")
    ax3[1].set_xlabel("Nombre de clusters (k)")
    ax3[1].set_ylabel("Score DB")
    ax3[1].grid(True)

    st.pyplot(fig3)

    #  Déterminer le meilleur K
    best_ch_k = k_values[np.argmax(ch_scores)]
    best_db_k = k_values[np.argmin(db_scores)]

    st.success(f"Selon Calinski–Harabasz, le nombre optimal de clusters est **k = {best_ch_k}**")
    st.info(f"Selon Davies–Bouldin, le nombre optimal de clusters est **k = {best_db_k}**")

    st.divider()
    st.markdown("""
    **Explication :**  
    - La méthode du *coude* consiste à observer la forme des courbes CH et DB :  
      - Pour **CH**, on choisit le *k* où le score atteint un maximum ou se stabilise.  
      - Pour **DB**, on choisit le *k* où le score atteint un minimum.  
    - Cela permet de trouver un compromis entre **cohésion interne** et **séparation entre clusters**.  
    - Le nombre de clusters optimal correspond au point où ajouter de nouveaux groupes n’apporte plus d’amélioration significative.
    """)

    if "Type" in df_planete.columns:
        n_types = df_planete["Type"].nunique()
        st.markdown(f"""
        **Résultat** : les deux indices convergent vers le même *k*, ce qui renforce ce choix.
        Le fichier contient pourtant **{n_types} types** étiquetés (dont un quasi vide) : les indices internes
        mesurent la compacité géométrique des groupes, pas la correspondance avec des catégories métier.
        Certains types d'atmosphère sont donc hétérogènes et se scindent en plusieurs sous-groupes.
        """)

#  PARTIE 4 – PIPELINE COMPLET

with tabs[3]:
    st.header("Pipeline complet d’analyse de données (Wine Quality)")

    st.markdown("""
    **Objectif :**  
    Appliquer un pipeline complet de *data science* sur le jeu de données **Wine Quality**,  
    contenant des mesures physico-chimiques de 4 898 vins blancs et une note de qualité (`quality`).  
    Des valeurs ont été volontairement supprimées du fichier pour travailler la détection et l'imputation
    des données manquantes. Le chargement fonctionne avec n'importe quel CSV séparé par `;` contenant une colonne `quality`.  
    Étapes : prétraitement, visualisations, clustering et apprentissage supervisé.
    """)

    #  1. Chargement  du jeu de données
    st.subheader("1. Chargement du jeu de données")

    uploaded = st.file_uploader("Charger un autre fichier CSV (séparateur `;`, optionnel)", type=["csv"])
    df_wine = None

    def _load_wine_df(file_or_path):
        """Lecture robuste d'un CSV avec nettoyage intelligent des valeurs vides."""
        import re

        # Lecture brute en texte
        df = pd.read_csv(file_or_path, sep=';', dtype=str, engine='python', quotechar='"', na_filter=False)

        # Nettoyage des chaînes
        def clean_str(x):
            if not isinstance(x, str):
                return x
            # Supprimer caractères invisibles et espaces
            x = x.replace('\xa0', ' ').replace('\u200b', '').strip()
            x = x.strip('"').strip("'")
            # Nettoyage des nombres avec virgules
            if re.match(r"^\d+,\d+$", x):
                x = x.replace(',', '.')
            # Transformer tous les placeholders en NaN
            if x in {"", " ", "NA", "N/A", "Na", "na", "Null", "null", "NONE", "None", "?", "-", "–", "—", ".", "..", "..."}:
                return np.nan
            return x

        df = df.map(clean_str)

        # Conversion numérique (quand c’est possible)
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        # Forcer les colonnes numériques à devenir NaN si erreurs
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                # Essai de conversion manuelle
                try:
                    df[col] = df[col].apply(lambda s: float(s) if str(s).replace('.', '', 1).isdigit() else s)
                except Exception:
                    pass

        return df

    #  Lecture
    if uploaded is not None:
        df_wine = _load_wine_df(uploaded)
        st.success("Fichier chargé.")
    else:
        try:
            df_wine = _load_wine_df(DATA_DIR / "winequality-white.csv")
            st.info("Jeu de données par défaut : `data/winequality-white.csv`.")
        except Exception as e:
            st.warning("Aucune donnée chargée.")
            st.stop()

    st.write(f"**Dimensions :** {df_wine.shape[0]} lignes × {df_wine.shape[1]} colonnes")
    st.dataframe(df_wine.head())

    # === 2. Exploration & valeurs manquantes ===
    st.subheader("2. Exploration et valeurs manquantes")

    import io
    buffer = io.StringIO()
    df_wine.info(buf=buffer)
    st.text(buffer.getvalue())

    st.markdown("#### Statistiques descriptives")
    numeric_cols = df_wine.select_dtypes(include=[np.number])
    if numeric_cols.shape[1] > 0:
        st.dataframe(numeric_cols.describe().T.style.format(precision=3))
    else:
        st.warning("Aucune colonne numérique détectée après conversion.")

    st.markdown("#### Valeurs manquantes")
    na_counts = df_wine.isna().sum().rename("Valeurs manquantes")
    st.dataframe(na_counts)
    total_nan = int(df_wine.isna().sum().sum())
    st.caption(f"Total de valeurs manquantes détectées : **{total_nan}**")

    fig_na, ax_na = plt.subplots(figsize=(8, 3))
    sns.heatmap(df_wine.isna(), cbar=False, ax=ax_na)
    ax_na.set_title("Carte des valeurs manquantes")
    st.pyplot(fig_na)

    # 3. Prétraitement
    st.subheader("3. Prétraitement des données")

    st.markdown("""
    - Les valeurs manquantes détectées sont imputées (médiane pour numérique).  
    - La colonne **quality** est considérée comme variable cible.  
    """)

    df_clean = df_wine.copy()
    for col in df_clean.columns:
        if df_clean[col].isna().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])

    if "quality" not in df_clean.columns:
        st.error("Colonne cible `quality` introuvable.")
        st.stop()

    X = df_clean.drop(columns=["quality"])
    y = df_clean["quality"]

    scaler4 = StandardScaler()
    X_scaled = scaler4.fit_transform(X)
    st.success("Imputation (médiane) et standardisation effectuées.")

    # 4. Visualisation
    st.subheader("4. Visualisation des données")

    st.markdown("#### Distribution de la qualité des vins")
    fig_dist, ax_dist = plt.subplots(figsize=(5, 3))
    sns.countplot(x=y, ax=ax_dist, color="darkred")
    ax_dist.set_xlabel("Quality")
    ax_dist.set_ylabel("Nombre de vins")
    st.pyplot(fig_dist)

    st.markdown("#### Matrice de corrélation")
    fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(df_clean.corr(), cmap="vlag", center=0, annot=False, ax=ax_corr)
    st.pyplot(fig_corr)

    st.markdown("#### Visualisation PCA (2D)")
    pca4 = PCA(n_components=2)
    X_pca4 = pca4.fit_transform(X_scaled)
    fig_pca4, ax_pca4 = plt.subplots(figsize=(6, 4))
    scatter = ax_pca4.scatter(X_pca4[:, 0], X_pca4[:, 1], c=y, cmap="viridis", alpha=0.6)
    ax_pca4.set_xlabel("Composante principale 1")
    ax_pca4.set_ylabel("Composante principale 2")
    ax_pca4.set_title("Projection PCA (colorée par qualité)")
    st.pyplot(fig_pca4)

    #5. Clustering non supervisé
    st.subheader("5. Clustering non supervisé")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        k = st.slider("Nombre de clusters pour K-Means", 2, 10, 3)
    with col_c2:
        eps = st.slider("Paramètre ε pour DBSCAN", 0.1, 3.0, 1.0, 0.1)

    # K-Means
    kmeans4 = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_scaled)
    labels_km4 = kmeans4.labels_
    sil_km4 = silhouette_score(X_scaled, labels_km4)
    st.write(f"**K-Means (k={k}) → Silhouette = {sil_km4:.3f}**")

    fig_km4, ax_km4 = plt.subplots(figsize=(6, 4))
    ax_km4.scatter(X_pca4[:, 0], X_pca4[:, 1], c=labels_km4, cmap="tab10", alpha=0.7)
    ax_km4.set_title("K-Means (projection PCA 2D)")
    st.pyplot(fig_km4)

    # DBSCAN
    from sklearn.cluster import DBSCAN
    db4 = DBSCAN(eps=eps, min_samples=5).fit(X_scaled)
    labels_db4 = db4.labels_
    unique_labels = len(set(labels_db4)) - (1 if -1 in labels_db4 else 0)
    if unique_labels > 1:
        sil_db4 = silhouette_score(X_scaled, labels_db4)
        st.write(f"**DBSCAN (ε={eps}) → Silhouette = {sil_db4:.3f}**")
    else:
        st.warning("DBSCAN a produit un seul cluster ou trop de bruit. Silhouette non définie.")

    fig_db4, ax_db4 = plt.subplots(figsize=(6, 4))
    ax_db4.scatter(X_pca4[:, 0], X_pca4[:, 1], c=labels_db4, cmap="tab10", alpha=0.7)
    ax_db4.set_title("DBSCAN (projection PCA 2D)")
    st.pyplot(fig_db4)

    # 6. Apprentissage supervisé
    st.subheader("6. Apprentissage supervisé")

    st.markdown("On cherche à prédire la qualité du vin (multiclasses ou binaire).")

    mode_sup = st.radio("Mode de classification", ["Multiclasses", "Binaire (quality ≥ seuil)"], index=1)
    if mode_sup == "Binaire (quality ≥ seuil)":
        seuil = st.slider("Seuil de 'bon vin'", 3, 8, 6)
        y_sup = (y >= seuil).astype(int)
    else:
        y_sup = y.copy()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression

    X_train4, X_test4, y_train4, y_test4 = train_test_split(
        X_scaled, y_sup, test_size=0.3, random_state=42, stratify=y_sup
    )

    f1_scores = {}
    models4 = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Arbre de Décision": DecisionTreeClassifier(random_state=42),
        "Régression Logistique": LogisticRegression(max_iter=2000)
    }

    for name, model in models4.items():
        st.markdown(f"#### {name}")
        model.fit(X_train4, y_train4)
        y_pred4 = model.predict(X_test4)
        f1 = f1_score(y_test4, y_pred4, average="weighted")
        f1_scores[name] = f1
        st.write(f"F1-score = **{f1:.3f}**")
        st.text(classification_report(y_test4, y_pred4))

        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(confusion_matrix(y_test4, y_pred4), annot=True, fmt='d', cmap="Blues", ax=ax_cm)
        ax_cm.set_title(f"Matrice de confusion – {name}")
        st.pyplot(fig_cm)

    st.subheader("7. Synthèse")
    best_model = max(f1_scores, key=f1_scores.get)
    st.markdown(f"""
    - **Données** : {df_wine.shape[0]} vins, {total_nan} valeurs manquantes détectées puis imputées par la médiane.
    - **Clustering** : silhouette K-Means = {sil_km4:.3f}. Les vins ne forment pas de groupes naturellement séparés
      dans l'espace physico-chimique ; DBSCAN, sensible à la densité, isole surtout du bruit.
    - **Supervisé** : meilleur modèle = **{best_model}** (F1 pondéré = {f1_scores[best_model]:.3f}).
      Les notes extrêmes étant rares, la qualité est plus facile à prédire en binaire (bon / moins bon)
      qu'en multiclasses.
    """)
