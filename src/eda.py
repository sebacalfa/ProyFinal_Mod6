from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import TARGET


def build_eda_decisions(df: pd.DataFrame, figures_dir: str | Path):
    """Genera análisis selectivos; cada fila declara una decisión accionable."""
    # Convierte la ruta y crea la carpeta donde se guardarán los gráficos.
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Acumula cada hallazgo junto con la decisión técnica que produce.
    rows = []

    # Calcula la proporción de cada clase para detectar desbalance en el target.
    target_share = df[TARGET].value_counts(normalize=True).sort_index()
    target_share.plot.bar(title="Distribución del target", ylabel="Proporción", rot=0)
    plt.tight_layout(); plt.savefig(figures_dir / "01_target_balance.png", dpi=150); plt.close()

    # Obtiene la proporción de la clase menos frecuente.
    minority = float(target_share.min())
    rows.append({"analisis": "Balance de clases", "resultado": f"clase minoritaria={minority:.2%}",
                 "decision": "Partición estratificada, class_weight='balanced' y F1/ROC-AUC además de accuracy."})

    # Mide el porcentaje de valores faltantes en cada variable predictora.
    missing = df.drop(columns=TARGET).isna().mean().sort_values(ascending=False)
    # Grafica solo las variables que realmente contienen valores faltantes.
    missing[missing > 0].plot.bar(title="Faltantes por feature", ylabel="Proporción")
    plt.tight_layout(); plt.savefig(figures_dir / "02_missingness.png", dpi=150); plt.close()
    rows.append({"analisis": "Valores faltantes", "resultado": f"máximo={missing.max():.2%} en {missing.idxmax()}",
                 "decision": "No eliminar filas; imputar dentro del pipeline usando solo el conjunto de entrenamiento."})

    # Comprueba que cada categoría educativa tenga un único código numérico.
    education_consistency = df.groupby("education")["education-num"].nunique().max()
    rows.append({"analisis": "Redundancia education", "resultado": f"máximo de códigos por categoría={education_consistency}",
                 "decision": "Conservar ambas representaciones para el modelo de árboles y vigilar su importancia; evitar crear una tercera copia."})

    # Calcula qué porcentaje de las variables de capital es exactamente cero.
    # Estos ceros representan ausencia de ganancia o pérdida, no datos faltantes.
    zero_gain = float(df["capital-gain"].eq(0).mean())
    zero_loss = float(df["capital-loss"].eq(0).mean())
    rows.append({"analisis": "Ceros en variables de capital", "resultado": f"gain={zero_gain:.2%}; loss={zero_loss:.2%}",
                 "decision": "No tratar ceros como faltantes; comparar modelos lineales y de árboles ante esta asimetría."})

    # Agrupa las horas semanales en intervalos y calcula la tasa de ingreso >50K.
    income_by_hours = df.assign(hours_band=pd.cut(df["hours-per-week"], [0, 30, 40, 60, 100])) \
        .groupby("hours_band", observed=True)[TARGET].apply(lambda s: s.eq(">50K").mean())
    income_by_hours.plot.bar(title="Ingreso >50K por horas semanales", ylabel="Proporción", rot=0)
    plt.tight_layout(); plt.savefig(figures_dir / "03_income_by_hours.png", dpi=150); plt.close()
    rows.append({"analisis": "Horas de trabajo", "resultado": "la tasa >50K cambia entre bandas de horas",
                 "decision": "Conservar hours-per-week y comparar modelos lineales con modelos que capturen relaciones no lineales."})

    # Devuelve una tabla resumida de análisis, resultados y decisiones.
    return pd.DataFrame(rows)
