import streamlit as st
import pandas as pd
import numpy as np
import requests
import math
import uszipcode
search = uszipcode.search.SearchEngine()

st.set_page_config(layout="wide")
st.title('A Location-Based Analysis System for Sears')
st.subheader('Problem Statement')

"""Sears is a pioneer in the retail industry and operates in a multitude of domains, selling millions of products on a daily basis and continuously adding thousands of new products to its product line.
In order to effectively evaluate the performance of retail stores, it is essential to have a consistent analysis system in place factoring the local market and competition.

Given a list of store location data, build a system that will generate a ranking of the locations based on the following -

   -- Distance from nearest Walmart, Costco, Target, and Home Depot stores. Each of these distances will have its own weightage that has to be accounted for in the ranking algorithm.\n
   -- Population density in and around the location.\n
   -- Economy around the location.\n

The weightage of these parameters needs to be configurable in the system."""

st.subheader('Load store location data')
data = pd.read_csv(
    "https://raw.githubusercontent.com/Pushpit07/Tech-Rush_AI-Ranking/main/locn_data.csv")

df = data[["locn_nbr", "locn_cty", "locn_st_cd",
           "zip_cd", "zip_pls_4", "locn_addr"]]

st.session_state.df = df

st.dataframe(df)


st.subheader(
    'Add coordinates, population density, and economy data for locations given in the dataset')


def get_coordinates_from_address(address):
    address = address.replace("#", "")
    url = 'https://nominatim.openstreetmap.org/search.php?q=' + \
        address.replace(" ", "+") + "&format=json"
    response = requests.get(url).json()
    return response


START_INDEX = 0
END_INDEX = 100


@st.cache_data
def get_coordinates(df):
    for index, row in df.iterrows():
        if (index >= START_INDEX and index < END_INDEX):
            try:
                address = row['locn_addr'].strip() + " " + row['locn_cty'].strip() + \
                    " " + row['locn_st_cd'].strip() + " " + str(row['zip_cd'])
                response = get_coordinates_from_address(address)

                if not response:
                    address = row['locn_cty'].strip(
                    ) + " " + row['locn_st_cd'].strip() + " " + str(row['zip_cd'])
                    response = get_coordinates_from_address(address)

                if not response:
                    address = row['locn_addr'].strip(
                    ) + " " + row['locn_st_cd'].strip() + " " + str(row['zip_cd'])
                    response = get_coordinates_from_address(address)

                if not response:
                    address = row['locn_st_cd'].strip() + " " + \
                        str(row['zip_cd'])
                    response = get_coordinates_from_address(address)

                if (response):
                    df.at[index, 'latitude'] = response[0]['lat']
                    df.at[index, 'longitude'] = response[0]['lon']
                    df.at[index, 'display_name'] = response[0]['display_name']

                zipcode = search.by_zipcode(row['zip_cd'])
                # Checking for non std postal codes
                if not zipcode.population:
                    # Checking for non std zipcodes like postal boxes
                    res = search.by_city_and_state(
                        city=zipcode.major_city, state=zipcode.state)
                    if (len(res)) > 0:
                        zipcode = res[0]

                df.at[index, 'population_density'] = zipcode.population_density
                df.at[index, 'median_household_income'] = zipcode.median_household_income

            except Exception as e:
                print("An exception occurred:", e)
                continue

    df['normalised_population_density'] = (((df['population_density'] - df['population_density'].min()) / (
        df['population_density'].max() - df['population_density'].min())) * 100)
    df['normalised_economy'] = (((df['median_household_income'] - df['median_household_income'].min()) / (
        df['median_household_income'].max() - df['median_household_income'].min())) * 100)

    return df


df = get_coordinates(df)
df = df.dropna()
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)


df.to_csv('with_coordinates.csv', index=False)
# df = pd.read_csv("./with_coordinates.csv")

# Removing outliers
df = df[df['latitude'] > 15]
# Removing outliers
df = df[df['latitude'] < 70]
# Removing outliers
df = df[df['longitude'] < -66]
# Removing outliers
df = df[df['longitude'] > -170]

st.dataframe(df)

st.subheader('Plotting all locations on the map')
st.map(df)


st.subheader('Calculating distance from nearest stores')

# The haversine formula is used to  calculate the distance between two geographic coordinates on the earth.
# This formula calculates the great-circle distance between two points on a sphere given their longitudes and latitudes.


def haversine_distance(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    # Calculate the haversine distance
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * \
        math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of the earth in kilometers
    return c * r


@st.cache_data
def get_distance_from_nearest_stores(df):
    # Load walmart stores location data
    walmart_stores_data = pd.read_csv(
        "https://raw.githubusercontent.com/Pushpit07/Tech-Rush_AI-Ranking/main/walmartstoreloaction.csv")

    for index, row in df.iterrows():
        distances_to_walmarts = []
        if (index >= START_INDEX and index < END_INDEX):
            for idx, walmart_store in walmart_stores_data.iterrows():
                distances_to_walmarts.append(haversine_distance(float(row['latitude']), float(
                    row['longitude']), float(walmart_store['latitude']), float(walmart_store['longitude'])))
            min_dist = min(distances_to_walmarts)
            df.at[index, 'distance_from_nearest_walmart'] = min_dist

    # Load costco stores location data
    costco_stores_data = pd.read_csv(
        "https://raw.githubusercontent.com/Pushpit07/Tech-Rush_AI-Ranking/main/costco_store.csv")

    for index, row in df.iterrows():
        distances_to_costcos = []
        if (index >= START_INDEX and index < END_INDEX):
            for idx, costco_store in costco_stores_data.iterrows():
                distances_to_costcos.append(haversine_distance(float(row['latitude']), float(
                    row['longitude']), float(costco_store['latitude']), float(costco_store['longitude'])))
            min_dist = min(distances_to_costcos)
            df.at[index, 'distance_from_nearest_costco'] = min_dist

    # Load target stores location data
    target_stores_data = pd.read_csv(
        "https://raw.githubusercontent.com/Pushpit07/Tech-Rush_AI-Ranking/main/target.csv", encoding='latin-1')

    for index, row in df.iterrows():
        distances_to_targets = []
        if (index >= START_INDEX and index < END_INDEX):
            for idx, target_store in target_stores_data.iterrows():
                distances_to_targets.append(haversine_distance(float(row['latitude']), float(
                    row['longitude']), float(target_store['Address.Latitude']), float(target_store['Address.Longitude'])))
            min_dist = min(distances_to_targets)
            df.at[index, 'distance_from_nearest_target'] = min_dist

    return df


df = get_distance_from_nearest_stores(df)


df['normalised_distance_from_nearest_walmart'] = (((df['distance_from_nearest_walmart'] - df['distance_from_nearest_walmart'].min()) / (
    df['distance_from_nearest_walmart'].max() - df['distance_from_nearest_walmart'].min())) * 100)
df['normalised_distance_from_nearest_costco'] = (((df['distance_from_nearest_costco'] - df['distance_from_nearest_costco'].min()) / (
    df['distance_from_nearest_costco'].max() - df['distance_from_nearest_costco'].min())) * 100)
df['normalised_distance_from_nearest_target'] = (((df['distance_from_nearest_target'] - df['distance_from_nearest_target'].min()) / (
    df['distance_from_nearest_target'].max() - df['distance_from_nearest_target'].min())) * 100)

st.dataframe(df)

st.session_state.df = df


# Weightage for each parameter
DISTANCE_WEIGHT = 0.4
POPULATION_DENSITY_WEIGHT = 0.3
ECONOMY_WEIGHT = 0.3


st.subheader('Enter weightage for each parameter')

st.write("Default weightage for each parameter is " + str(DISTANCE_WEIGHT) + " for distance, " +
         str(POPULATION_DENSITY_WEIGHT) + " for population density, and " + str(ECONOMY_WEIGHT) + " for economy.")


# Session state callbacks
def callback_distance_weight():
    # The callback renormalizes the values that were not updated.
    remain = 1 - st.session_state.DISTANCE_WEIGHT
    # This is the proportions of X2, X3 and X4 in remain
    sum = st.session_state.POPULATION_DENSITY_WEIGHT + st.session_state.ECONOMY_WEIGHT
    # This is the normalisation step
    st.session_state.POPULATION_DENSITY_WEIGHT = st.session_state.POPULATION_DENSITY_WEIGHT/sum*remain
    st.session_state.ECONOMY_WEIGHT = st.session_state.ECONOMY_WEIGHT/sum*remain


def callback_pop_density_weight():
    remain = 1 - st.session_state.POPULATION_DENSITY_WEIGHT
    sum = st.session_state.DISTANCE_WEIGHT + st.session_state.ECONOMY_WEIGHT
    st.session_state.DISTANCE_WEIGHT = st.session_state.DISTANCE_WEIGHT/sum*remain
    st.session_state.ECONOMY_WEIGHT = st.session_state.ECONOMY_WEIGHT/sum*remain


def callback_economy_weight():
    remain = 1 - st.session_state.ECONOMY_WEIGHT
    sum = st.session_state.DISTANCE_WEIGHT + \
        st.session_state.POPULATION_DENSITY_WEIGHT
    st.session_state.DISTANCE_WEIGHT = st.session_state.DISTANCE_WEIGHT/sum*remain
    st.session_state.POPULATION_DENSITY_WEIGHT = st.session_state.POPULATION_DENSITY_WEIGHT/sum*remain


DISTANCE_WEIGHT = st.slider(
    'Distance weight', min_value=0.0, max_value=1.0, value=DISTANCE_WEIGHT, step=0.01, key="DISTANCE_WEIGHT", on_change=callback_distance_weight)
POPULATION_DENSITY_WEIGHT = st.slider(
    'Population density weight', min_value=0.0, max_value=1.0, value=POPULATION_DENSITY_WEIGHT, step=0.01, key="POPULATION_DENSITY_WEIGHT", on_change=callback_pop_density_weight)
ECONOMY_WEIGHT = st.slider(
    'Economy weight', min_value=0.0, max_value=1.0, value=ECONOMY_WEIGHT, step=0.01, key="ECONOMY_WEIGHT", on_change=callback_economy_weight)


st.code(
    "NOTE: Ranking data and map will be updated automatically after changing the weightage")

NEAREST_WALMART_DISTANCE_WEIGHT = 0.5
NEAREST_COSTCO_DISTANCE_WEIGHT = 0.3
NEAREST_TARGET_DISTANCE_WEIGHT = 0.2


def callback_walmart_distance_weight():
    remain = 1 - st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT
    sum = st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT + \
        st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT
    st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT = st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT/sum*remain
    st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT = st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT/sum*remain


def callback_costco_distance_weight():
    remain = 1 - st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT
    sum = st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT + \
        st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT
    st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT = st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT/sum*remain
    st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT = st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT/sum*remain


def callback_target_distance_weight():
    remain = 1 - st.session_state.NEAREST_TARGET_DISTANCE_WEIGHT
    sum = st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT + \
        st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT
    st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT = st.session_state.NEAREST_WALMART_DISTANCE_WEIGHT/sum*remain
    st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT = st.session_state.NEAREST_COSTCO_DISTANCE_WEIGHT/sum*remain


col1, col2 = st.columns(2)
with col1:
    st.subheader('Enter weightage for Walmart, Costco, and Target stores')
    NEAREST_WALMART_DISTANCE_WEIGHT = st.slider(
        'Nearest Walmart Distance weight', min_value=0.0, max_value=1.0, value=NEAREST_WALMART_DISTANCE_WEIGHT, step=0.01, key="NEAREST_WALMART_DISTANCE_WEIGHT", on_change=callback_walmart_distance_weight)
    NEAREST_COSTCO_DISTANCE_WEIGHT = st.slider(
        'Nearest Costco Distance weight', min_value=0.0, max_value=1.0, value=NEAREST_COSTCO_DISTANCE_WEIGHT, step=0.01, key="NEAREST_COSTCO_DISTANCE_WEIGHT", on_change=callback_costco_distance_weight)
    NEAREST_TARGET_DISTANCE_WEIGHT = st.slider(
        'Nearest Target Distance weight', min_value=0.0, max_value=1.0, value=NEAREST_TARGET_DISTANCE_WEIGHT, step=0.01, key="NEAREST_TARGET_DISTANCE_WEIGHT", on_change=callback_target_distance_weight)


st.subheader('Enter the number of top results to obtain')
NUMBER_OF_TOP_RESULTS = 5
NUMBER_OF_TOP_RESULTS = st.radio("Choose a number", index=2, options=[
                                 1, 3, 5, 10], key="NUMBER_OF_TOP_RESULTS", horizontal=True)


def calculate_score(df):
    for index, row in df.iterrows():
        if (index >= START_INDEX and index < END_INDEX):
            try:
                # Assigning a score based on greatest population density, higher economy, and low distances from Walmart and Costco
                score_of_distance_from_stores = (df.at[index, 'normalised_distance_from_nearest_walmart'] * NEAREST_WALMART_DISTANCE_WEIGHT +
                                                 df.at[index, 'normalised_distance_from_nearest_costco'] * NEAREST_COSTCO_DISTANCE_WEIGHT +
                                                 df.at[index, 'normalised_distance_from_nearest_target'] * NEAREST_TARGET_DISTANCE_WEIGHT) * DISTANCE_WEIGHT
                df.at[index, 'score'] = df.at[index, 'normalised_population_density'] * POPULATION_DENSITY_WEIGHT + \
                    df.at[index, 'normalised_economy'] * \
                    ECONOMY_WEIGHT - score_of_distance_from_stores

            except Exception as e:
                print("An exception occurred", e)
                continue

    df['normalised_score'] = (
        ((df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())) * 100)

    df['rank'] = df['normalised_score'].rank(ascending=False)

    st.subheader('Top ' + str(NUMBER_OF_TOP_RESULTS) + ' ranked locations')

    top_df = df.nsmallest(NUMBER_OF_TOP_RESULTS, "rank")
    st.dataframe(top_df)

    st.subheader('Top ' + str(NUMBER_OF_TOP_RESULTS) +
                 ' ranked locations on map')
    st.map(top_df)

    st.subheader('Score of each location')
    st.dataframe(df)

    return df


calculate_score(st.session_state.df)
