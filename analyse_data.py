import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas

from data import COLUMN_TO_PREDICT


def analyse(df: pandas.DataFrame) -> None:
    # 1. Data Overview
    print(df.info())
    print(df.describe())

    # 2. Descriptive Statistics
    # Calculating mean, median for price and surface
    print("Mean price:", df[COLUMN_TO_PREDICT].mean())
    print("Median price:", df[COLUMN_TO_PREDICT].median())
    print("Mean surface:", df['surface'].mean())
    print("Median surface:", df['surface'].median())

    # 3. Distribution Analysis
    # Histogram for price
    sns.histplot(df[COLUMN_TO_PREDICT], kde=True)
    plt.title('Price Distribution')
    plt.show()

    # Histogram for surface
    sns.histplot(df['surface'], kde=True)
    plt.title('Surface Distribution')
    plt.show()

    # 5. Category Analysis
    # Boxplot for price by number of rooms
    sns.boxplot(x='room', y=COLUMN_TO_PREDICT, data=df)
    plt.title('Price Distribution by Number of Rooms')
    plt.show()

    # Boxplot for price by elevator presence
    sns.boxplot(x='elevator', y=COLUMN_TO_PREDICT, data=df)
    plt.title('Price Distribution by Elevator Presence')
    plt.show()

    # 6. Location Analysis
    # Scatter plot for price by latitude and longitude
    sns.scatterplot(x='location.lon', y='location.lat', size=COLUMN_TO_PREDICT, hue=COLUMN_TO_PREDICT, data=df)
    plt.title('Price Distribution by Location')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    # Normalize the price for color mapping
    price_normalized = (df[COLUMN_TO_PREDICT] - df[COLUMN_TO_PREDICT].min()) / (df[COLUMN_TO_PREDICT].max() - df[COLUMN_TO_PREDICT].min())

    plt.figure(figsize=(10, 6))
    plt.scatter(df['location.lon'], df['location.lat'], alpha=0.6, c=price_normalized, cmap='viridis', s=df[COLUMN_TO_PREDICT] / 10000)
    plt.colorbar(label=COLUMN_TO_PREDICT)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Price Based on Location')
    plt.show()

    # Assuming `df` is your DataFrame and it has 'lat', 'lon', and 'price' columns

    show_scatter_mapbox(df)


def show_scatter_mapbox(df: pandas.DataFrame):
    fig = px.scatter_mapbox(df, lat="location.lat", lon="location.lon", color="price", size="price",
                            color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                            mapbox_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()


def analyse_v2(df: pandas.DataFrame) -> None:
    # 1. Data Overview
    print(df.info())
    print(df.describe())

    # 2. Descriptive Statistics
    # Calculating mean, median for price and surface
    print("Mean Valeurs fonciere:", df[COLUMN_TO_PREDICT].mean())
    print("Median Valeurs fonciere:", df[COLUMN_TO_PREDICT].median())
    print("Mean Surface reelle bati:", df['Surface reelle bati'].mean())
    print("Median Surface reelle bati:", df['Surface reelle bati'].median())

    sns.boxplot(x='Surface reelle bati', y=COLUMN_TO_PREDICT, data=df)
    plt.title('Price Distribution by Surface reelle bati')
    plt.show()

    sns.boxplot(x='Type local', y=COLUMN_TO_PREDICT, data=df)
    plt.title('Price Distribution by Type local')
    plt.show()
