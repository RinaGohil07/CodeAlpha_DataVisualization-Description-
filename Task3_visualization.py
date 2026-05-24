# ==============================================================
#   CodeAlpha Internship | Task 3: Data Visualization
#   Project  : Titanic Survival — Story Through Data
#   Dataset  : titanic.csv
#   Author   : (Your Name)
#   Libraries: pandas, matplotlib, seaborn
# ==============================================================
#
#  WHAT THIS SCRIPT DOES:
#    Creates a COMPELLING visual data story with:
#    - A stunning 9-chart storytelling dashboard
#    - Custom color themes & professional design
#    - Annotations and text insights on charts
#    - A separate "infographic-style" summary poster
#
#  HOW TO RUN:
#    pip install pandas matplotlib seaborn
#    python visualization.py
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── SETTINGS ─────────────────────────────────────────────────
DATA_FILE      = "titanic.csv"
OUTPUT_STORY   = "titanic_visual_story.png"
OUTPUT_POSTER  = "titanic_infographic.png"

# Premium Color Palette
C_DARK    = "#0F172A"
C_NAVY    = "#1E3A5F"
C_BLUE    = "#2563EB"
C_MID     = "#3B82F6"
C_LIGHT   = "#93C5FD"
C_PALE    = "#DBEAFE"
C_BG      = "#EEF2FF"
C_CARD    = "#F8FAFF"
C_ACCENT  = "#F59E0B"    # gold accent
C_RED     = "#EF4444"
C_GREEN   = "#10B981"

PALETTE_FULL = [C_NAVY, C_BLUE, C_MID, C_LIGHT, C_PALE,
                "#1D4ED8","#1E40AF","#172554","#BFDBFE","#60A5FA"]
# ─────────────────────────────────────────────────────────────


def load_and_prepare(filepath):
    """Load, clean and engineer features for visualization."""
    df = pd.read_csv(filepath)
    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df.drop(columns=["Cabin"], inplace=True, errors="ignore")

    # Feature Engineering
    df["AgeGroup"] = pd.cut(df["Age"],
        bins=[0,12,18,35,60,100],
        labels=["Child\n(0-12)","Teen\n(13-18)","Young Adult\n(19-35)",
                "Adult\n(36-60)","Senior\n(60+)"])
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["FamilySizeGroup"] = df["FamilySize"].apply(
        lambda x: "Alone" if x==1 else ("Small\n(2-4)" if x<=4 else "Large\n(5+)")
    )
    df["Pclass_Label"] = df["Pclass"].map({1:"1st Class\n(Rich)", 2:"2nd Class\n(Middle)", 3:"3rd Class\n(Poor)"})
    df["Sex_Label"]    = df["Sex"].str.capitalize()
    df["Survived_Label"] = df["Survived"].map({1:"Survived", 0:"Did Not Survive"})
    df["Embarked_Label"] = df["Embarked"].map({"S":"Southampton","C":"Cherbourg","Q":"Queenstown"})

    return df


def plot_visual_story(df):
    """
    Creates a 9-panel visual story of the Titanic disaster.
    Each chart tells a different part of the story.
    """
    print("  Creating Visual Story Dashboard...")

    fig = plt.figure(figsize=(24, 20), facecolor=C_BG)
    fig.suptitle(
        "THE TITANIC STORY  —  A Data Visualization Journey\n"
        "CodeAlpha Internship  |  Task 3: Data Visualization",
        fontsize=20, fontweight="bold", color=C_DARK, y=0.995
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    # ────────────────────────────────────────────────────────
    # Chart 1: SURVIVAL DONUT CHART
    # ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    survived = df["Survived"].sum()
    died     = len(df) - survived
    wedge_colors = [C_GREEN, C_RED]
    wedges, texts, autos = ax1.pie(
        [survived, died],
        labels=["Survived", "Did Not\nSurvive"],
        autopct="%1.1f%%",
        colors=wedge_colors,
        startangle=90,
        pctdistance=0.70,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=3),
        textprops={"fontsize": 9, "color": C_DARK, "fontweight": "bold"}
    )
    for auto in autos:
        auto.set_fontsize(10); auto.set_fontweight("bold"); auto.set_color("white")
    # Center label
    ax1.text(0, 0, f"{len(df)}\nPassengers", ha="center", va="center",
             fontsize=10, fontweight="bold", color=C_DARK)
    ax1.set_title("Overall Survival", fontsize=12, fontweight="bold",
                  color=C_NAVY, pad=10)
    ax1.set_facecolor(C_CARD)

    # ────────────────────────────────────────────────────────
    # Chart 2: SURVIVAL RATE BY GENDER + CLASS (Grouped)
    # ────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pivot = df.pivot_table(values="Survived", index="Pclass",
                           columns="Sex", aggfunc="mean") * 100
    x = np.arange(len(pivot.index))
    w = 0.32
    bars_f = ax2.bar(x - w/2, pivot["female"], w,
                     label="Female", color=C_BLUE, edgecolor="white", linewidth=0.8)
    bars_m = ax2.bar(x + w/2, pivot["male"], w,
                     label="Male", color=C_LIGHT, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["1st Class","2nd Class","3rd Class"], fontsize=9)
    ax2.set_title("Survival Rate: Class & Gender", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax2.set_ylabel("Survival Rate (%)", fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9, framealpha=0.5)
    ax2.set_facecolor(C_CARD)
    ax2.spines[["top","right"]].set_visible(False)
    for bar in list(bars_f) + list(bars_m):
        if bar.get_height() > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{bar.get_height():.0f}%", ha="center", fontsize=7.5,
                     fontweight="bold", color=C_DARK)

    # ────────────────────────────────────────────────────────
    # Chart 3: AGE DISTRIBUTION — SURVIVED vs NOT
    # ────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    surv_ages  = df[df["Survived"]==1]["Age"]
    dead_ages  = df[df["Survived"]==0]["Age"]
    ax3.hist(dead_ages,  bins=20, alpha=0.70, color=C_RED,   label="Did Not Survive", edgecolor="white")
    ax3.hist(surv_ages,  bins=20, alpha=0.70, color=C_GREEN, label="Survived",        edgecolor="white")
    ax3.set_title("Age Distribution: Survived vs Not", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax3.set_xlabel("Age (years)", fontsize=9)
    ax3.set_ylabel("Count", fontsize=9)
    ax3.legend(fontsize=9, framealpha=0.5)
    ax3.set_facecolor(C_CARD)
    ax3.spines[["top","right"]].set_visible(False)
    # Highlight children
    ax3.axvspan(0, 12, alpha=0.12, color=C_ACCENT, label="Children zone")
    ax3.text(6, ax3.get_ylim()[1]*0.85, "Children\nzone", ha="center",
             fontsize=7.5, color=C_ACCENT, fontweight="bold")

    # ────────────────────────────────────────────────────────
    # Chart 4: FARE vs AGE SCATTER PLOT
    # ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    colors_scatter = df["Survived"].map({1: C_GREEN, 0: C_RED})
    ax4.scatter(df["Age"], df["Fare"], c=colors_scatter,
                alpha=0.55, s=25, edgecolors="none")
    ax4.set_title("Fare vs Age (Color = Survival)", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax4.set_xlabel("Age", fontsize=9)
    ax4.set_ylabel("Fare Paid (GBP)", fontsize=9)
    ax4.set_yscale("log")
    survived_patch = mpatches.Patch(color=C_GREEN, label="Survived")
    died_patch     = mpatches.Patch(color=C_RED,   label="Did Not Survive")
    ax4.legend(handles=[survived_patch, died_patch], fontsize=9, framealpha=0.5)
    ax4.set_facecolor(C_CARD)
    ax4.spines[["top","right"]].set_visible(False)

    # ────────────────────────────────────────────────────────
    # Chart 5: SURVIVAL BY AGE GROUP
    # ────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    age_surv = (df.groupby("AgeGroup", observed=True)["Survived"]
                  .agg(["mean","count"]))
    age_surv["mean"] = age_surv["mean"] * 100
    bar_colors = [C_GREEN if v > 40 else C_RED for v in age_surv["mean"]]
    bars5 = ax5.bar(age_surv.index, age_surv["mean"],
                    color=bar_colors, edgecolor="white", linewidth=0.8, width=0.6, alpha=0.9)
    ax5.axhline(38.4, color=C_ACCENT, linestyle="--", linewidth=2, label="Overall avg (38.4%)")
    for bar, val in zip(bars5, age_surv["mean"]):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f"{val:.1f}%", ha="center", fontsize=9,
                 color=C_DARK, fontweight="bold")
    ax5.set_title("Survival Rate by Age Group", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax5.set_ylabel("Survival Rate (%)", fontsize=9)
    ax5.set_ylim(0, 72)
    ax5.legend(fontsize=8)
    ax5.set_facecolor(C_CARD)
    ax5.spines[["top","right"]].set_visible(False)

    # ────────────────────────────────────────────────────────
    # Chart 6: FAMILY SIZE vs SURVIVAL
    # ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    fam_surv = (df.groupby("FamilySizeGroup")["Survived"]
                  .mean() * 100).reindex(["Alone","Small\n(2-4)","Large\n(5+)"])
    bars6 = ax6.bar(fam_surv.index, fam_surv.values,
                    color=[C_NAVY, C_BLUE, C_LIGHT],
                    edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars6, fam_surv.values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=11,
                 color=C_DARK, fontweight="bold")
    ax6.set_title("Survival Rate by Family Size", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax6.set_ylabel("Survival Rate (%)", fontsize=9)
    ax6.set_ylim(0, 75)
    ax6.set_facecolor(C_CARD)
    ax6.spines[["top","right"]].set_visible(False)

    # ────────────────────────────────────────────────────────
    # Chart 7: PASSENGER CLASS BREAKDOWN (Stacked Bar)
    # ────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    class_data = df.groupby(["Pclass_Label","Survived_Label"]).size().unstack(fill_value=0)
    class_data.plot(kind="barh", stacked=True, ax=ax7,
                    color=[C_RED, C_GREEN], edgecolor="white", linewidth=0.8)
    ax7.set_title("Passenger Count by Class & Survival", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax7.set_xlabel("Number of Passengers", fontsize=9)
    ax7.set_ylabel("")
    ax7.legend(fontsize=9, framealpha=0.5)
    ax7.set_facecolor(C_CARD)
    ax7.spines[["top","right"]].set_visible(False)
    ax7.tick_params(labelsize=8)

    # ────────────────────────────────────────────────────────
    # Chart 8: EMBARKATION PORT ANALYSIS
    # ────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    port_data = df.groupby("Embarked_Label").agg(
        Total=("Survived","count"),
        Survived=("Survived","sum")
    )
    port_data["Survival_Rate"] = (port_data["Survived"] / port_data["Total"] * 100).round(1)
    x8 = np.arange(len(port_data))
    w8 = 0.32
    b1 = ax8.bar(x8 - w8/2, port_data["Total"],    w8, label="Total",    color=C_LIGHT, edgecolor="white")
    b2 = ax8.bar(x8 + w8/2, port_data["Survived"], w8, label="Survived", color=C_BLUE,  edgecolor="white")
    ax8.set_xticks(x8)
    ax8.set_xticklabels(port_data.index, fontsize=9)
    for i, (_, row) in enumerate(port_data.iterrows()):
        ax8.text(i + w8/2, row["Survived"] + 2,
                 f"{row['Survival_Rate']}%", ha="center",
                 fontsize=8, fontweight="bold", color=C_DARK)
    ax8.set_title("Embarkation Port: Total vs Survived", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax8.set_ylabel("Count", fontsize=9)
    ax8.legend(fontsize=9, framealpha=0.5)
    ax8.set_facecolor(C_CARD)
    ax8.spines[["top","right"]].set_visible(False)

    # ────────────────────────────────────────────────────────
    # Chart 9: CORRELATION HEATMAP (Advanced)
    # ────────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    num_cols = ["Survived","Pclass","Age","SibSp","Parch","Fare","FamilySize"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(corr, ax=ax9, mask=mask, cmap=cmap, center=0,
                annot=True, fmt=".2f", annot_kws={"size": 7.5, "weight":"bold"},
                linewidths=0.8, linecolor="white",
                cbar_kws={"shrink": 0.75, "label": "Correlation"})
    ax9.set_title("Feature Correlation Heatmap", fontsize=12,
                  fontweight="bold", color=C_NAVY, pad=10)
    ax9.tick_params(labelsize=8)

    plt.savefig(OUTPUT_STORY, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Visual Story saved  ->  {OUTPUT_STORY}")


def plot_infographic(df):
    """
    Creates a clean infographic-style poster with key stats.
    This is great for LinkedIn posts!
    """
    print("  Creating Infographic Poster...")

    fig, ax = plt.subplots(figsize=(12, 16), facecolor=C_DARK)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)
    ax.axis("off")

    def stat_box(ax, x, y, number, label, color, size=2.8):
        rect = mpatches.FancyBboxPatch((x-size/2, y-0.8), size, 1.8,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="white",
            linewidth=1.5, alpha=0.92)
        ax.add_patch(rect)
        ax.text(x, y+0.55, number, ha="center", va="center",
                fontsize=22, fontweight="bold", color="white")
        ax.text(x, y-0.25, label, ha="center", va="center",
                fontsize=9, color="white", alpha=0.9)

    # Title
    ax.text(5, 15.3, "T I T A N I C", ha="center", fontsize=38,
            fontweight="bold", color="white")
    ax.text(5, 14.6, "DATA VISUALIZATION  |  CodeAlpha Internship",
            ha="center", fontsize=11, color=C_LIGHT, alpha=0.85)
    ax.axhline(14.2, xmin=0.05, xmax=0.95, color=C_BLUE, linewidth=1.5, alpha=0.5)

    # Key Stats Row 1
    total    = len(df)
    survived = df["Survived"].sum()
    died     = total - survived
    stat_box(ax, 1.8, 13.0,  f"{total}",    "Total Passengers", C_NAVY)
    stat_box(ax, 5.0, 13.0,  f"{survived}", "Survived",         C_GREEN)
    stat_box(ax, 8.2, 13.0,  f"{died}",     "Did Not Survive",  C_RED)

    # Key Stats Row 2
    female_surv = f"{df[df['Sex']=='female']['Survived'].mean()*100:.0f}%"
    male_surv   = f"{df[df['Sex']=='male']['Survived'].mean()*100:.0f}%"
    avg_age     = f"{df['Age'].mean():.1f} yrs"
    stat_box(ax, 1.8, 11.0, female_surv, "Female Survival", C_BLUE)
    stat_box(ax, 5.0, 11.0, male_surv,   "Male Survival",   C_MID)
    stat_box(ax, 8.2, 11.0, avg_age,     "Average Age",     C_NAVY)

    # Key Stats Row 3
    cls1 = f"{df[df['Pclass']==1]['Survived'].mean()*100:.0f}%"
    cls2 = f"{df[df['Pclass']==2]['Survived'].mean()*100:.0f}%"
    cls3 = f"{df[df['Pclass']==3]['Survived'].mean()*100:.0f}%"
    stat_box(ax, 1.8, 9.0, cls1, "1st Class\nSurvival", "#F59E0B")
    stat_box(ax, 5.0, 9.0, cls2, "2nd Class\nSurvival", "#F59E0B")
    stat_box(ax, 8.2, 9.0, cls3, "3rd Class\nSurvival", "#F59E0B")

    # Divider
    ax.axhline(8.0, xmin=0.05, xmax=0.95, color=C_BLUE, linewidth=1, alpha=0.4)

    # Key Findings
    ax.text(5, 7.5, "KEY FINDINGS", ha="center", fontsize=14,
            fontweight="bold", color="white")
    findings = [
        ("Women had 3x higher survival rate than men",          C_GREEN),
        ("1st class passengers survived at 2x rate of 3rd class", C_ACCENT),
        ("Children under 12 had the highest survival rate",       C_LIGHT),
        ("Small families (2-4) survived better than solo travelers", C_MID),
        ("Higher fare strongly correlates with survival chance",  C_PALE),
    ]
    for i, (text, color) in enumerate(findings):
        y_pos = 6.8 - i * 0.9
        ax.plot(0.7, y_pos, "o", color=color, markersize=10)
        ax.text(1.1, y_pos, text, va="center", fontsize=10,
                color="white", alpha=0.9)

    # Footer
    ax.axhline(1.2, xmin=0.05, xmax=0.95, color=C_BLUE, linewidth=1, alpha=0.4)
    ax.text(5, 0.7, "Task 3: Data Visualization  |  CodeAlpha Data Analytics Internship",
            ha="center", fontsize=9, color=C_LIGHT, alpha=0.7)

    plt.tight_layout(pad=0)
    plt.savefig(OUTPUT_POSTER, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Infographic saved   ->  {OUTPUT_POSTER}")


# =============================================================
#  MAIN
# =============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  CodeAlpha Internship | Task 3: Data Visualization")
    print("  Dataset: Titanic Passenger Survival")
    print("="*60)

    print("\n  Loading and preparing data...")
    df = load_and_prepare(DATA_FILE)
    print(f"  Dataset loaded: {len(df)} passengers, {len(df.columns)} features")

    # 1. 9-Chart Story Dashboard
    plot_visual_story(df)

    # 2. Infographic Poster
    plot_infographic(df)

    print(f"\n  YOUR PROJECT FILES:")
    print(f"    visualization.py          <- source code")
    print(f"    {DATA_FILE}               <- dataset")
    print(f"    {OUTPUT_STORY}  <- 9-chart dashboard")
    print(f"    {OUTPUT_POSTER} <- infographic poster")
    print("\n  Upload all to GitHub repo: CodeAlpha_DataVisualization")
    print("  Use the infographic as your LinkedIn post image!")
    print("="*60 + "\n")
