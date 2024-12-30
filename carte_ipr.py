import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
import io
from pyproj import Transformer
import geopy
from geopy.geocoders import Nominatim

# Afficher le logo dans la barre latérale
st.sidebar.image("logo_aqua.PNG", use_container_width=True)


# Charger les données
@st.cache_data
def load_data():
    return pd.read_csv('ipr_data_with_trends.csv')

data = load_data()

# Convertir les coordonnées Lambert en latitude/longitude
transformer = Transformer.from_crs("epsg:2154", "epsg:4326")
data[['Latitude', 'Longitude']] = data.apply(
    lambda row: pd.Series(transformer.transform(row['CoordXStationMesureEauxSurface'], row['CoordYStationMesureEauxSurface'])),
    axis=1
)

# Garder uniquement les dernières mesures pour l'affichage sur la carte
latest_data = data.sort_values('DateDebutOperationPrelBio').groupby('LbStationMesureEauxSurface').tail(1)

# Ajouter une colonne pour la tendance linéaire
# Calcul de la pente pour chaque station
station_trends = data.groupby('LbStationMesureEauxSurface')['Score_Numeric'].apply(
    lambda scores: np.polyfit(range(len(scores)), scores, 1)[0] if len(scores) > 1 else np.nan
)

# Mapper la tendance en description textuelle
def interpret_trend(slope):
    if pd.isna(slope):
        return "Non significative"
    elif slope < 0:
        return "Amélioration"
    elif slope > 0:
        return "Détérioration"
    else:
        return "Stable"

data['Tendance'] = data['LbStationMesureEauxSurface'].map(station_trends).map(interpret_trend)

# Fonction pour afficher la courbe des scores IPR
# et une ligne de tendance pointillée

def plot_ipr_curve(station_data, station_name):
    fig, ax = plt.subplots()
    station_data = station_data.sort_values('DateDebutOperationPrelBio')

    # Tracé des points de mesure IPR avec valeurs numériques
    dates = pd.to_datetime(station_data['DateDebutOperationPrelBio'])
    scores = station_data['Score_Numeric']
    ax.plot(dates, scores, marker='o', label='IPR')
    for i, txt in enumerate(scores):
        ax.annotate(f"{txt:.2f}", (dates.iloc[i], scores.iloc[i]), textcoords="offset points", xytext=(0,5), ha='center')

    # Calcul et affichage de la tendance si plusieurs points
    if len(scores) > 1:
        coeffs = np.polyfit(range(len(scores)), scores, 1)
        trend = np.polyval(coeffs, range(len(scores)))
        ax.plot(dates, trend, linestyle='--', color='red', label='Tendance linéaire')

    # Ajouter les limites de qualité avec des lignes discrètes en gris
    ax.axhline(y=7, color='grey', linestyle=':')
    ax.axhline(y=16, color='grey', linestyle=':')
    ax.axhline(y=25, color='grey', linestyle=':')
    ax.axhline(y=36, color='grey', linestyle=':')

    ax.set_title(f"Courbe IPR pour {station_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Score IPR")
    ax.legend(loc='upper right', labels=['IPR', 'Tendance linéaire'])
    plt.tight_layout()

    # Convert plot to image and display it
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    return buffer

# Menu déroulant pour choisir une carte

# Menu déroulant pour choisir une carte

selected_map = st.selectbox(
    "Choisissez une carte // Choose a map :",
    ["Scores IPR des rivières", "Carte ROE des ouvrages des rivières"]
)


### Carte IPR des rivières ###

if selected_map == "Scores IPR des rivières":
    # Carte de France pour les scores IPR
    st.title("Carte des scores IPR des rivières de France métropolitaine")

    # Ajouter une légende expliquant l'IPR
    st.markdown(
        """**Indice Poisson Rivière (IPR)** :\
        - Évalue la qualité écologique des cours d'eau à partir de la faune piscicole.\
        - Les classes de qualité :\
          - **≤ 7** : Qualité excellente\
          - **7-16** : Bonne qualité\
          - **16-25** : Qualité médiocre\
          - **25-36** : Qualité mauvaise\
          - **> 36** : Très mauvaise qualité\

          Source des données : [Naïades](https://naiades.eaufrance.fr/france-entiere#/)\
          46 174 observations. Dernière mise à jour : année 2023"""
    )

    # Définir les couleurs pour chaque catégorie
    color_scale = {
        "Excellent": "blue",
        "Bon": "green",
        "Médiocre": "yellow",
        "Mauvais": "orange",
        "Très Mauvais": "red"
    }
    latest_data["Couleur"] = latest_data["Qualité_Estimée"].map(color_scale)

    # Créer une carte interactive avec Plotly
    fig = px.scatter_mapbox(
        latest_data,
        lat='Latitude',
        lon='Longitude',
        hover_name='LbStationMesureEauxSurface',
        hover_data={
            "Date de dernière observation": latest_data["DateDebutOperationPrelBio"]
        },
        color="Qualité_Estimée",
        title="Scores IPR des stations",
        color_discrete_map=color_scale,
        category_orders={"Qualité_Estimée": ["Excellent", "Bon", "Médiocre", "Mauvais", "Très Mauvais"]}
    )
    fig.update_layout(
        mapbox=dict(center={"lat": 46.603354, "lon": 1.888334}, zoom=5), 
        mapbox_style="open-street-map",
        height=800  # Augmenter la hauteur de la carte
    )

    # Afficher la carte
    st.plotly_chart(fig)

    # Interaction avec la carte
    station_selected = st.selectbox("Choisissez une station :", data['LbStationMesureEauxSurface'].unique())
    station_data = data[data['LbStationMesureEauxSurface'] == station_selected]

    if not station_data.empty:
        st.write(f"### Détails pour la station : {station_selected}")
        st.write(f"Qualité actuelle : {station_data['Qualité_Estimée'].iloc[-1]}")
        st.write(f"Tendance : {station_data['Tendance'].iloc[0]}")
        st.write(f"Date de la dernière mesure : {station_data['DateDebutOperationPrelBio'].iloc[-1]}")

        # Afficher la courbe des scores IPR
        st.image(plot_ipr_curve(station_data, station_selected))

        # Ajouter un bouton pour télécharger les données sources
        csv = station_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger les données sources",
            data=csv,
            file_name=f"{station_selected}_data.csv",
            mime='text/csv'
        )


### Carte ROE des ouvrages des rivières ###

if selected_map == "Carte ROE des ouvrages des rivières":
    st.title("Carte ROE des ouvrages des rivières de France métropolitaine")

  # Fonction de géocodage indépendante
def get_city_coordinates(city_name):
    geolocator = Nominatim(user_agent="streamlit_app")
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        st.warning(f"Ville '{city_name}' introuvable. Vérifiez l'orthographe.")
        return None, None

### Carte ROE des ouvrages des rivières ###

# Charger les données ROE
if selected_map == "Carte ROE des ouvrages des rivières":
    # Charger les données uniquement si elles ne sont pas déjà définies
    if 'roe_data' not in locals():
        roe_data = pd.read_csv('roe_2024_valid_with_categories.csv', low_memory=False)

        # Convertir Lambert 93 en Latitude/Longitude si nécessaire
        if 'Latitude' not in roe_data.columns or 'Longitude' not in roe_data.columns:
            transformer = Transformer.from_crs("epsg:2154", "epsg:4326")
            roe_data[['Latitude', 'Longitude']] = roe_data.apply(
                lambda row: pd.Series(transformer.transform(row['CoordYPoin'], row['CoordXPoin']))
                if not pd.isnull(row['CoordXPoin']) and not pd.isnull(row['CoordYPoin']) else pd.Series([None, None]),
                axis=1
            )

    # Fonction pour géocoder une ville
    def get_city_coordinates(city_name):
        geolocator = Nominatim(user_agent="streamlit_app")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.warning(f"Ville '{city_name}' introuvable. Vérifiez l'orthographe.")
            return None, None

    # Ajouter un champ de recherche
    city_name = st.text_input("Recherchez une ville pour centrer la carte :", "")

    city_lat, city_lon = None, None
    if city_name:
        city_lat, city_lon = get_city_coordinates(city_name)


# Créer la carte interactive
fig = px.scatter_mapbox(
    roe_data.dropna(subset=['Latitude', 'Longitude']),
    lat='Latitude',
    lon='Longitude',
    hover_name='NomPrincip',
    hover_data={
        "N° de ROE": roe_data['CdObstEcou'],
        "Type d'ouvrage": roe_data['LbTypeOuvr'],
        "Hauteur de chute": roe_data['LbHautChut'],
        "Cours d'eau": roe_data['NomEntiteH'],
        "Commune": roe_data['LbCommune'],
        "Département": roe_data['LbDepartem'],
        "Bassin": roe_data['NomCircAdm']
    },
    color='TypeOuvrageSimplifie',  # Utilisation de la catégorie simplifiée
    labels={"TypeOuvrageSimplifie": "Type d'ouvrages"},  # Renommer pour la légende
    text='NomPrincip',  # Texte affiché sur la carte à côté des points
    title="Carte des ouvrages des rivières",
    mapbox_style="open-street-map",
    height=800
)

# Modifier la taille des points
fig.update_traces(
    marker=dict(size=10),  # Points plus gros
    textfont=dict(size=10),  # Taille du texte
    textposition="top right"  # Position du texte par rapport au point
)

# Déplacer la légende sous la carte
fig.update_layout(
    legend=dict(
        title="Type d'ouvrages",  # Titre de la légende
        orientation="h",  # Orientation horizontale
        yanchor="bottom",
        y=-0.2,  # Position verticale (sous la carte)
        xanchor="center",
        x=0.5  # Centrer horizontalement
    )
)

# Centrer la carte sur la ville si trouvée
if city_lat is not None and city_lon is not None:
    fig.update_layout(mapbox=dict(center={"lat": city_lat, "lon": city_lon}, zoom=10))
else:
    fig.update_layout(mapbox=dict(center={"lat": 46.603354, "lon": 1.888334}, zoom=5))

# Afficher la carte
st.plotly_chart(fig)

st.title("Tri croisé et export des données")

# Filtrer par rivière
selected_river = st.selectbox(
    "Choisissez une rivière :", 
    options=["Tous"] + sorted(roe_data['NomEntiteH'].dropna().unique())
)

# Filtrer par département
selected_department = st.selectbox(
    "Choisissez un département :", 
    options=["Tous"] + sorted(roe_data['LbDepartem'].dropna().unique())
)

# Filtrer par commune
selected_city = st.selectbox(
    "Choisissez une commune :", 
    options=["Tous"] + sorted(roe_data['LbCommune'].dropna().unique())
)

# Filtrer par type d'obstacle
selected_obstacle = st.selectbox(
    "Choisissez un type d'obstacle :", 
    options=["Tous"] + sorted(roe_data['LbTypeOuvr'].dropna().unique())
)

# Filtrer par hauteur de chute
selected_height = st.select_slider(
    "Choisissez une plage de hauteur de chute :", 
    options=sorted(roe_data['LbHautChut'].dropna().unique())
)

# Filtrer par bassin
selected_basin = st.selectbox(
    "Choisissez un bassin :", 
    options=["Tous"] + sorted(roe_data['NomCircAdm'].dropna().unique())
)

# Appliquer les filtres croisés
filtered_data = roe_data.copy()

if selected_river != "Tous":
    filtered_data = filtered_data[filtered_data['NomEntiteH'] == selected_river]

if selected_department != "Tous":
    filtered_data = filtered_data[filtered_data['LbDepartem'] == selected_department]

if selected_city != "Tous":
    filtered_data = filtered_data[filtered_data['LbCommune'] == selected_city]

if selected_obstacle != "Tous":
    filtered_data = filtered_data[filtered_data['LbTypeOuvr'] == selected_obstacle]

if selected_height != "Tous":
    filtered_data = filtered_data[filtered_data['LbHautChut'] == selected_height]

if selected_basin != "Tous":
    filtered_data = filtered_data[filtered_data['NomCircAdm'] == selected_basin]

# Afficher un aperçu des données filtrées
st.write("### Données filtrées :")
st.dataframe(filtered_data)

# Ajouter un bouton pour télécharger les données filtrées
csv = filtered_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Télécharger les données filtrées",
    data=csv,
    file_name="donnees_filtrees.csv",
    mime='text/csv'
)
