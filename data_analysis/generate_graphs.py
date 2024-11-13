import pandas as pd
import matplotlib.pyplot as plt


def plot_distribution(df):
    # Plot the data
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Plot column A
    df["A"].value_counts().sort_index().plot(
        ax=ax[0], kind="bar", color="blue", title="Cama A"
    )

    # Plot column B
    df["B"].value_counts().sort_index().plot(
        ax=ax[1], kind="bar", color="green", title="Cama B"
    )

    # Plot column C
    df["C"].value_counts().sort_index().plot(
        ax=ax[2], kind="bar", color="red", title="Cama C"
    )

    # Set labels
    for a in ax:
        a.set_ylabel("Conteo")

    plt.xlabel("Categorías")
    plt.tight_layout()
    plt.show()


def plot_most_used_bed(df):
    bed_counts = df[["A", "B", "C"]].apply(lambda x: (x == "vaca_acostada").sum())
    bed_counts.plot(
        kind="bar", color=["blue", "green", "red"], title="Vaca Acostada en Cada Cama"
    )
    plt.ylabel("Conteo")
    plt.xlabel("Cama")
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
