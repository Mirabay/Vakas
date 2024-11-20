"This script generates graphs for the data analysis of the CSV files"
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_distribution(df):
    # Plot the data
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot column A
    sns.countplot(x="A", data=df, ax=ax[0], color="blue")
    ax[0].set_title("Cama A")

    # Plot column B
    sns.countplot(x="B", data=df, ax=ax[1], color="green")
    ax[1].set_title("Cama B")

    # Plot column C
    sns.countplot(x="C", data=df, ax=ax[2], color="red")
    ax[2].set_title("Cama C")

    # Set labels
    for a in ax:
        a.set_ylabel("Conteo")

    plt.xlabel("Categorías")
    plt.tight_layout()
    plt.show()


def plot_most_used_bed(df):
    bed_counts = (
        df[["A", "B", "C"]]
        .apply(lambda x: (x == "vaca_acostada").sum())
        .sort_values(ascending=False)
    )
    print("Número de vacas acostadas por cada cama:")
    print(bed_counts)
    bed_counts_percentage = (bed_counts / bed_counts.sum() * 100).dropna()
    plt.figure(figsize=(8, 8))
    plt.pie(
        bed_counts_percentage,
        labels=bed_counts_percentage.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Ditribución de vacas acostadas por cama")
    plt.tight_layout()
    plt.show()


def plot_most_empty_bed(df):
    bed_counts = (
        df[["A", "B", "C"]]
        .apply(lambda x: (x == "cama_vacia").sum())
        .sort_values(ascending=False)
    )
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.barplot(
        x=bed_counts.values,
        y=bed_counts.index,
        orient="h",
    )
    sns.despine()
    plt.title("Número de camas vacías por cada cama")
    plt.ylabel("Cama")
    plt.xlabel("Conteo")
    plt.tight_layout()
    plt.show()

    def plot_empty_bed_percentage(df):
        bed_counts = (
            df[["A", "B", "C"]]
            .apply(lambda x: (x == "cama_vacia").sum())
            .sort_values(ascending=False)
        )
        bed_counts_percentage = (bed_counts / bed_counts.sum() * 100).dropna()
        plt.figure(figsize=(8, 8))
        plt.pie(
            bed_counts_percentage,
            labels=bed_counts_percentage.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel"),
        )
        plt.title("Distribución porcentual de camas vacías por cama")
        plt.tight_layout()
        plt.show()

    def plot_least_used_bed(df):
        bed_counts = (
            df[["A", "B", "C"]]
            .apply(lambda x: (x == "vaca_acostada").sum())
            .sort_values(ascending=True)
        )
        print("Número de vacas acostadas por cada cama (de menor a mayor):")
        print(bed_counts)
        bed_counts_percentage = (bed_counts / bed_counts.sum() * 100).dropna()
        plt.figure(figsize=(8, 8))
        plt.pie(
            bed_counts_percentage,
            labels=bed_counts_percentage.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=sns.color_palette("pastel"),
        )
        plt.title("Distribución de vacas acostadas por cama (menos usada)")
        plt.tight_layout()
        plt.show()


def plot_bed_subdivisions(df):
    bed_counts = df[["A", "B", "C"]].apply(pd.Series.value_counts).fillna(0)
    bed_counts = bed_counts.transpose()
    sns.set_theme(style="whitegrid", palette="pastel")
    bed_counts.plot(
        kind="barh",
        stacked=True,
        figsize=(10, 8),
        color=sns.color_palette("pastel", n_colors=3),
    )
    plt.title("Distribución total predecida de imágenes por camas")
    plt.xlabel("Cantidad de imágenes")
    plt.ylabel("Cama")
    plt.legend(title="Categoría")
    plt.gca().invert_yaxis()  # Invert the order of the bars
    plt.tight_layout()
    plt.show()


def plot_bed_usage_over_time(df, column_name):
    df = df.copy()
    category_mapping = {"vaca_acostada": 1, "vaca_parada": 2, "cama_vacia": 3}
    reverse_category_mapping = {v: k for k, v in category_mapping.items()}
    df[column_name] = df[column_name].map(category_mapping)

    # Filter data for one day
    one_day_df = df.loc["2024-02-08"]

    # Plot the data for one day
    one_day_df[column_name].plot()
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M"))
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=1))
    plt.gcf().autofmt_xdate()

    plt.yticks(
        ticks=list(reverse_category_mapping.keys()),
        labels=list(reverse_category_mapping.values()),
    )
    plt.title(f"Uso de cama {column_name} en el día 2024-02-08")
    plt.xlabel("Hora")
    plt.ylabel("Estado de la cama")
    plt.grid(False)  # Remove the grid
    plt.show()


def plot_hourly_total_vaca_acostada(df):
    # Extraer la hora del índice
    df["Hour"] = df.index.hour

    # Contar ocurrencias de "vaca_acostada" por hora para cada cama
    vaca_acostada_counts = (
        df[["A", "B", "C"]]
        .apply(lambda x: x == "vaca_acostada")
        .groupby(df["Hour"])
        .sum()
    )

    # Sumar entre todas las camas para obtener el total por hora
    vaca_acostada_counts["Total"] = vaca_acostada_counts.sum(axis=1)

    # Reiniciar índice para graficar
    vaca_acostada_counts = vaca_acostada_counts.reset_index()

    # Configurar las horas para el eje x
    hours = vaca_acostada_counts["Hour"]

    # Graficar histogramas para cada cama
    for bed in ["A", "B", "C"]:
        plt.figure(figsize=(12, 6))
        plt.bar(
            hours,
            vaca_acostada_counts[bed],
            color="skyblue",
            edgecolor="black",
            alpha=0.7,
            width=0.8,
            label=f"Cama {bed}",
        )
        plt.title(f"Distribución horaria de vacas acostadas en cama {bed}")
        plt.xlabel("Hora del día")
        plt.ylabel(f"Cantidad de vacas acostadas en cama {bed}")
        plt.xticks(range(0, 24))  # Asegurar que todas las horas aparecen en el eje x
        plt.legend()
        plt.grid(False)  # Remove the grid
        plt.tight_layout()
        plt.show()

    # Graficar histograma combinado para el total
    plt.figure(figsize=(12, 6))
    plt.bar(
        hours,
        vaca_acostada_counts["Total"],
        color="orange",
        edgecolor="black",
        alpha=0.7,
        width=0.8,
        label="Total",
    )
    plt.title("Distribución horaria total de vacas acostadas (todas las camas)")
    plt.xlabel("Hora del día")
    plt.ylabel("Cantidad total de vacas acostadas")
    plt.xticks(range(0, 24))  # Asegurar que todas las horas aparecen en el eje x
    plt.legend()
    plt.grid(False)  # Remove the grid
    plt.tight_layout()
    plt.show()

    # Imprimir la hora con mayor cantidad de vacas acostadas en total
    max_hour_total = vaca_acostada_counts.loc[vaca_acostada_counts["Total"].idxmax()]
    print(
        f"La hora con más vacas acostadas en total es {int(max_hour_total['Hour'])} "
        f"con un total de {int(max_hour_total['Total'])} vacas."
    )


# Read the CSV file called predictions
df = pd.read_csv(
    "C:\\Users\\urigo\\Documents\\Vakas\\data_analysis\\predictionsFull-Dataset.csv"
)

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Set timestamp as index
df.set_index("Timestamp", inplace=True)


plot_bed_subdivisions(df)
plot_bed_usage_over_time(df, "A")
plot_bed_usage_over_time(df, "B")
plot_bed_usage_over_time(df, "C")
plot_most_used_bed(df)
plot_hourly_total_vaca_acostada(df)
