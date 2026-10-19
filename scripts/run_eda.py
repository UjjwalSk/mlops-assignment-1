"""Generate comprehensive, deterministic EDA figures and statistical summaries."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.heart_risk.data import load_processed_data

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / "data" / "raw" / "processed.cleveland.data"
    if not raw_path.exists():
        from scripts.prepare_data import main as prep_main
        prep_main()

    data = load_processed_data(raw_path)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    # 1. Age distribution and Target Balance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(data=data, x="age", hue="target", multiple="stack", ax=axes[0], palette=["#2b5c8f", "#d95f02"], bins=15)
    axes[0].set_title("Age Distribution by Heart Disease Status", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Patient Count")

    counts = data["target"].value_counts().rename(index={0: "Absence (0)", 1: "Presence (1)"})
    colors = ["#2b5c8f", "#d95f02"]
    bars = axes[1].bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black")
    axes[1].set_title("Binary Target Class Balance", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Number of Patients")
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(data)) * 100
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, height + 3, f"{height} ({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, max(counts.values) + 25)
    fig.tight_layout()
    fig.savefig(OUT / "distribution_and_balance.png", dpi=200)
    plt.close(fig)

    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = data.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("UCI Heart Disease Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "correlation_heatmap.png", dpi=200)
    plt.close(fig)

    # 3. Categorical / Clinical Feature Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    cp_map = {1: "Typ. Angina", 2: "Atyp. Angina", 3: "Non-anginal", 4: "Asymptomatic"}
    temp_data = data.copy()
    temp_data["cp_label"] = temp_data["cp"].map(cp_map)
    sns.countplot(data=temp_data, x="cp_label", hue="target", ax=axes[0, 0], palette=["#2b5c8f", "#d95f02"])
    axes[0, 0].set_title("Chest Pain Type (CP) vs Disease", fontweight="bold")
    axes[0, 0].set_xlabel("Chest Pain Type")
    axes[0, 0].set_ylabel("Count")

    thal_map = {3: "Normal", 6: "Fixed Defect", 7: "Reversable Defect"}
    temp_data["thal_label"] = temp_data["thal"].map(thal_map)
    sns.countplot(data=temp_data, x="thal_label", hue="target", ax=axes[0, 1], palette=["#2b5c8f", "#d95f02"])
    axes[0, 1].set_title("Thallium Stress Test (Thal) vs Disease", fontweight="bold")
    axes[0, 1].set_xlabel("Thallium Category")
    axes[0, 1].set_ylabel("Count")

    exang_map = {0: "No", 1: "Yes"}
    temp_data["exang_label"] = temp_data["exang"].map(exang_map)
    sns.countplot(data=temp_data, x="exang_label", hue="target", ax=axes[1, 0], palette=["#2b5c8f", "#d95f02"])
    axes[1, 0].set_title("Exercise Induced Angina (Exang) vs Disease", fontweight="bold")
    axes[1, 0].set_xlabel("Exercise Angina")
    axes[1, 0].set_ylabel("Count")

    sns.countplot(data=temp_data, x="ca", hue="target", ax=axes[1, 1], palette=["#2b5c8f", "#d95f02"])
    axes[1, 1].set_title("Major Vessels Colored by Fluoroscopy (CA) vs Disease", fontweight="bold")
    axes[1, 1].set_xlabel("Number of Major Vessels (0-3)")
    axes[1, 1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(OUT / "clinical_features_distribution.png", dpi=200)
    plt.close(fig)

    # 4. Numerical Feature Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    num_features = [
        ("trestbps", "Resting Blood Pressure (mm Hg)", axes[0, 0]),
        ("chol", "Serum Cholesterol (mg/dl)", axes[0, 1]),
        ("thalach", "Maximum Heart Rate Achieved (bpm)", axes[1, 0]),
        ("oldpeak", "ST Depression (Oldpeak)", axes[1, 1]),
    ]
    for col, title, ax in num_features:
        sns.boxplot(
            data=data,
            x="target",
            y=col,
            hue="target",
            legend=False,
            ax=ax,
            palette=["#2b5c8f", "#d95f02"],
            width=0.4,
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks([0, 1], ["Absence (0)", "Presence (1)"])
        ax.set_xlabel("Disease Status")

    fig.tight_layout()
    fig.savefig(OUT / "numerical_boxplots.png", dpi=200)
    plt.close(fig)

    # 5. Data Quality and Summary Statistics
    summary = pd.DataFrame({
        "Metric": ["Total Rows", "Features Count", "Missing Values Count", "Negative Class Count (0)", "Positive Class Count (1)", "Disease Prevalence Rate"],
        "Value": [
            str(len(data)),
            str(len(data.columns) - 1),
            str(int(data.isna().sum().sum())),
            str(int((data.target == 0).sum())),
            str(int((data.target == 1).sum())),
            f"{float(data.target.mean()):.4f} ({float(data.target.mean())*100:.1f}%)",
        ]
    })
    summary.to_csv(OUT / "eda_summary.csv", index=False)
    print("EDA Visualizations and Summary generated successfully:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
