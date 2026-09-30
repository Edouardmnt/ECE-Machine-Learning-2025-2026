import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm

from pathlib import Path

st.set_page_config(page_title="Régression linéaire - chauves-souris", layout="centered")
st.title("Masse du cerveau vs masse corporelle chez les chauves-souris")
st.markdown(
    "Régression linéaire simple (statsmodels) de la masse du cerveau **BRW** (mg) "
    "en fonction de la masse corporelle **BOW** (g) sur 29 espèces de chauves-souris, "
    "puis étude de l'influence d'une espèce atypique."
)

DEFAULT_FILE = Path(__file__).parent / "data" / "tabBats.txt"

# Fichier fourni par défaut, remplaçable par un autre fichier au même format
uploaded = st.file_uploader("Charger un autre fichier (optionnel)", type=["txt", "csv"])
data_file = uploaded if uploaded is not None else DEFAULT_FILE

if data_file is not None:
    try:
        # Lecture du  fichier en tenant compte des guillemets et espaces avec pandas 
        df = pd.read_csv(data_file, sep=r"\s+", quotechar='"')

        st.header("Partie A : Exploration des données")
        st.subheader("Aperçu des premières lignes")
        st.write(df.head())

        # afficher les types de colonnes
        st.subheader("Types de colonnes")
        st.write(df.dtypes.astype(str))

        # afficher les statistiques descriptives
        st.subheader("Statistiques descriptives")
        st.write(df.describe())

        # selectionner les colonnes pertinentes
        if set(['Species','BOW','BRW']).issubset(df.columns):
            data = df[['Species','BOW','BRW']].dropna()
            st.subheader("Variables pertinentes (Species, BOW, BRW)")
            st.write(data.head())

            # partie B 
            st.header("Partie B : Régression linéaire simple")

            # affichage du nuage de points avec BOW en abscisse et BRW en ordonnée
            st.subheader("1. Nuage de points")
            fig, ax = plt.subplots()
            ax.scatter(data['BOW'], data['BRW'], alpha=0.6)
            ax.set_xlabel("BOW (masse corporelle)")
            ax.set_ylabel("BRW (masse du cerveau)")
            ax.set_title("Nuage de points : BRW en fonction de BOW")
            st.pyplot(fig)
            st.markdown("La relation est clairement croissante et linéaire. Un point isolé se détache "
                        "très nettement : *Pteropus vampyrus* (roussette de Malaisie, plus de 1 kg).")

            # ajustement du modèle de régression linéaire
            st.subheader("2. Ajustement du modèle de régression linéaire")  
            model = smf.ols('BRW ~ BOW', data=data).fit()

            # résumé du modèle
            st.subheader("2-3. Résumé du modèle de régression")

            coeffs = pd.DataFrame({
                "Coefficient": model.params,
                "Std. Error": model.bse,
                "t-value": model.tvalues,
                "p-value": model.pvalues
            })
            st.markdown("### Coefficients estimés")
            st.dataframe(coeffs)

            st.markdown(" Statistiques globales du modèle")
            col1, col2, col3 = st.columns(3)
            col1.metric("R²", f"{model.rsquared:.3f}")
            col2.metric("R² ajusté", f"{model.rsquared_adj:.3f}")
            col3.metric("F-statistic", f"{model.fvalue:.2f}")

            col1, col2 = st.columns(2)
            col1.metric("AIC", f"{model.aic:.1f}")
            col2.metric("BIC", f"{model.bic:.1f}")

            st.markdown(f"""
            - **Nombre d'observations** : {int(model.nobs)}
            - **Signification globale (Prob > F)** : {model.f_pvalue:.2e}
            """)

            st.markdown(f"""
            **Interprétation**
            - Chaque gramme de masse corporelle supplémentaire est associé à **{model.params['BOW']:.1f} mg**
              de cerveau en plus (p-value = {model.pvalues['BOW']:.1e}, effet très significatif).
            - Le modèle explique **{model.rsquared:.0%}** de la variance de la masse du cerveau.
            - Ce R² élevé est cependant tiré par un seul point extrême : il faut vérifier les résidus.
            """)

            # analyse des résidus
            st.subheader("Analyse des résidus")
            fitted = model.fittedvalues
            resid = model.resid

            fig, axes = plt.subplots(1, 2, figsize=(10,4))
            axes[0].scatter(fitted, resid, alpha=0.6)
            axes[0].axhline(0, color='k', linestyle='--')
            axes[0].set_xlabel("Valeurs ajustées")
            axes[0].set_ylabel("Résidus")
            axes[0].set_title("Résidus vs valeurs ajustées")

            sm.qqplot(resid, line='45', fit=True, ax=axes[1])
            axes[1].set_title("QQ-plot des résidus")
            st.pyplot(fig)

            # tracer la droite de régression
            st.subheader("4. Droite de régression sur le nuage de points")
            fig, ax = plt.subplots()
            ax.scatter(data['BOW'], data['BRW'], alpha=0.6, label="Données")
            x_vals = np.linspace(data['BOW'].min(), data['BOW'].max(), 100)
            y_vals = model.params['Intercept'] + model.params['BOW'] * x_vals
            ax.plot(x_vals, y_vals, color='red', linewidth=2, label="Régression")
            ax.set_xlabel("BOW")
            ax.set_ylabel("BRW")
            ax.set_title("Droite de régression BRW ~ BOW")
            ax.legend()
            st.pyplot(fig)

            # analyse sans Pteropus vampyrus
            st.header("Partie C : Analyse sans l'espèce atypique Pteropus vampyrus")

            # retrait de l'espèce
            tab2 = data[data['Species'] != 'Pteropus vampyrus']

            # comparaison visuelle des nuages de points
            st.subheader("1-2. Comparaison des nuages de points")
            fig, axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)
            axes[0].scatter(data['BOW'], data['BRW'], alpha=0.6)
            axes[0].set_title("Avec Pteropus vampyrus")
            axes[0].set_xlabel("BOW")
            axes[0].set_ylabel("BRW")

            axes[1].scatter(tab2['BOW'], tab2['BRW'], alpha=0.6, color='orange')
            axes[1].set_title("Sans Pteropus vampyrus")
            axes[1].set_xlabel("BOW")

            st.pyplot(fig)

            # ajustement d'un second modèle
            model2 = smf.ols('BRW ~ BOW', data=tab2).fit()

            st.subheader("Comparaison des résultats")
            coeffs2 = pd.DataFrame({
                "Coefficient": model2.params,
                "Std. Error": model2.bse,
                "t-value": model2.tvalues,
                "p-value": model2.pvalues
            })
            st.markdown(" Coefficients sans Pteropus vampyrus")
            st.dataframe(coeffs2)

            col1, col2, col3 = st.columns(3)
            col1.metric("R² (avec)", f"{model.rsquared:.3f}")
            col2.metric("R² (sans)", f"{model2.rsquared:.3f}")
            col3.metric("ΔR²", f"{model2.rsquared - model.rsquared:.3f}")

            # superposition des deux droites de régression
            st.subheader("4. Superposition des droites de régression")
            fig, ax = plt.subplots()
            ax.scatter(data['BOW'], data['BRW'], alpha=0.6, label="Données")

            # droite avec Pteropus vampyrus
            y_vals1 = model.params['Intercept'] + model.params['BOW'] * x_vals
            ax.plot(x_vals, y_vals1, color='red', linewidth=2, label="Avec P. vampyrus")

            # droite sans Pteropus vampyrus
            y_vals2 = model2.params['Intercept'] + model2.params['BOW'] * x_vals
            ax.plot(x_vals, y_vals2, color='blue', linewidth=2, linestyle="--", label="Sans P. vampyrus")

            ax.set_xlabel("BOW")
            ax.set_ylabel("BRW")
            ax.set_title("Comparaison des droites de régression")
            ax.legend()
            st.pyplot(fig)

            st.markdown(f"""
            **Effet de l'espèce atypique**
            - *Pteropus vampyrus* pèse près de 4 fois plus que l'espèce suivante : c'est un point à fort **effet de levier**.
            - Sans elle, la pente passe de **{model.params['BOW']:.1f}** à **{model2.params['BOW']:.1f}** mg/g
              et le R² de **{model.rsquared:.3f}** à **{model2.rsquared:.3f}**.
            - Un seul individu extrême suffisait à tirer la droite vers le bas : il faut toujours
              contrôler les points influents avant d'interpréter une régression.
            """)

        else:
            st.warning(f"Colonnes disponibles : {df.columns.tolist()}")

    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
else:
    st.info("Veuillez charger le fichier tabBats.txt.")