import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Load the dataset
train = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /31st Jan-3rd Feb/X_train.csv")

# Create a PDF file to save all analysis
with PdfPages("data_analysis_report.pdf") as pdf:
    # Set the style for plots
    sns.set(style="whitegrid")

    # Basic information
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.5, str(train.info()), fontsize=10, family="monospace")
    plt.axis("off")
    plt.title("Dataset Information")
    pdf.savefig()
    plt.close()

    # Summary statistics
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.5, str(train.describe()), fontsize=10, family="monospace")
    plt.axis("off")
    plt.title("Summary Statistics")
    pdf.savefig()
    plt.close()

    # Missing values
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.5, str(train.isnull().sum()), fontsize=10, family="monospace")
    plt.axis("off")
    plt.title("Missing Values")
    pdf.savefig()
    plt.close()

    # Histograms for numerical features
    train.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features")
    pdf.savefig()
    plt.close()

    # Bar plots for categorical features (if any)
    categorical_features = train.select_dtypes(include=["object"]).columns
    for col in categorical_features:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=train, x=col)
        plt.title(f"Bar Plot of {col}")
        pdf.savefig()
        plt.close()

    # Scatter plots for numerical features
    numerical_features = train.select_dtypes(include=["int64", "float64"]).columns
    for i in range(len(numerical_features)):
        for j in range(i + 1, len(numerical_features)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=train, x=numerical_features[i], y=numerical_features[j], hue="target")
            plt.title(f"{numerical_features[i]} vs {numerical_features[j]}")
            pdf.savefig()
            plt.close()

    # Box plots for numerical vs categorical features
    for num_col in numerical_features:
        for cat_col in categorical_features:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=train, x=cat_col, y=num_col, hue="target")
            plt.title(f"{num_col} vs {cat_col}")
            pdf.savefig()
            plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[numerical_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    pdf.savefig()
    plt.close()

    # Pair plot
    sns.pairplot(train[numerical_features + ["target"]], hue="target", diag_kind="kde")
    plt.suptitle("Pair Plot of Numerical Features", y=1.02)
    pdf.savefig()
    plt.close()

    # Target distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=train, x="target")
    plt.title("Distribution of Target Variable")
    pdf.savefig()
    plt.close()

print("Analysis saved to data_analysis_report.pdf")