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
    sns.set_theme(style="whitegrid", palette="pastel")
    sns.barplot(
        x=bed_counts.values,
        y=bed_counts.index,
        orient="h",
    )
    sns.despine()
    plt.title("Número de vacas acostadas por cada cama")
    plt.ylabel("Cama")
    plt.xlabel("Conteo")
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


def plot_bed_subdivisions(df):
    bed_counts = df[["A", "B", "C"]].apply(pd.Series.value_counts).fillna(0)
    bed_counts = bed_counts.transpose()
    bed_counts.plot(
        kind="bar", stacked=True, figsize=(10, 8), color=["blue", "green", "red"]
    )
    plt.title("Distribución de categorías por cama")
    plt.ylabel("Conteo")
    plt.xlabel("Cama")
    plt.legend(title="Categoría")
    plt.tight_layout()
    plt.show()


# Read the CSV file called predictions
df = pd.read_csv(
    "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/data_analysis/predictions.csv"
)

# Convert timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Set timestamp as index
df.set_index("Timestamp", inplace=True)

plot_most_used_bed(df)
plot_most_empty_bed(df)
plot_bed_subdivisions(df)
