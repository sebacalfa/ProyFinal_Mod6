import pandas as pd


def ejecutar_data_quality_gates(
    data,
    target="income",
    minimum_rows=30_000,
    max_duplicate_rate=0.0,
    min_target_proportion=0.10
):
    """
    Ejecuta validaciones automáticas sobre el dataset preparado para modelado.

    Si alguna validación falla, detiene el pipeline mediante ValueError.
    También devuelve una tabla con el resultado de cada regla.
    """

    resultados = []

    def registrar(regla, condicion, detalle):
        resultados.append({
            "regla": regla,
            "estado": "APROBADA" if condicion else "FALLIDA",
            "detalle": detalle
        })

    # ---------------------------------------------------------
    # Configuración esperada para el dataset Adult
    # ---------------------------------------------------------

    columnas_esperadas = {
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "income"
    }

    columnas_enteras = [
        "age",
        "fnlwgt",
        "education-num",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "income"
    ]

    # ---------------------------------------------------------
    # Gate 1: esquema de columnas
    # ---------------------------------------------------------

    columnas_actuales = set(data.columns)
    faltantes = columnas_esperadas - columnas_actuales
    adicionales = columnas_actuales - columnas_esperadas

    esquema_valido = not faltantes and not adicionales

    registrar(
        regla="Esquema de columnas",
        condicion=esquema_valido,
        detalle=(
            f"Faltantes: {sorted(faltantes) or 'ninguna'}; "
            f"adicionales: {sorted(adicionales) or 'ninguna'}"
        )
    )

    # ---------------------------------------------------------
    # Gate 2: cantidad mínima de observaciones
    # ---------------------------------------------------------

    cantidad_filas = len(data)

    registrar(
        regla="Cantidad mínima de filas",
        condicion=cantidad_filas >= minimum_rows,
        detalle=f"{cantidad_filas:,} filas; mínimo requerido: {minimum_rows:,}"
    )

    # ---------------------------------------------------------
    # Gate 3: valores nulos
    # ---------------------------------------------------------

    total_nulos = int(data.isna().sum().sum())

    registrar(
        regla="Ausencia de valores nulos",
        condicion=total_nulos == 0,
        detalle=f"{total_nulos} valores nulos encontrados"
    )

    # ---------------------------------------------------------
    # Gate 4: valores faltantes representados como texto
    # ---------------------------------------------------------

    columnas_texto = data.select_dtypes(include=["object", "string"]).columns

    if len(columnas_texto) > 0:
        valores_texto = data[columnas_texto].apply(
            lambda columna: columna.astype("string").str.strip()
        )

        mascaras_invalidas = valores_texto.isin(
            ["?", "", "NA", "N/A", "null", "None"]
        )

        total_placeholders = int(mascaras_invalidas.sum().sum())
    else:
        total_placeholders = 0

    registrar(
        regla="Ausencia de placeholders de faltantes",
        condicion=total_placeholders == 0,
        detalle=f"{total_placeholders} valores como '?', vacío, NA o null"
    )

    # ---------------------------------------------------------
    # Gate 5: filas duplicadas
    # ---------------------------------------------------------

    cantidad_duplicados = int(data.duplicated().sum())
    proporcion_duplicados = data.duplicated().mean()

    registrar(
        regla="Proporción de duplicados",
        condicion=proporcion_duplicados <= max_duplicate_rate,
        detalle=(
            f"{cantidad_duplicados} duplicados "
            f"({proporcion_duplicados:.2%}); "
            f"máximo permitido: {max_duplicate_rate:.2%}"
        )
    )

    # ---------------------------------------------------------
    # Gate 6: tipos numéricos esperados
    # ---------------------------------------------------------

    columnas_no_numericas = [
        columna
        for columna in columnas_enteras
        if columna not in data.columns
        or not pd.api.types.is_numeric_dtype(data[columna])
    ]

    registrar(
        regla="Tipos numéricos",
        condicion=len(columnas_no_numericas) == 0,
        detalle=(
            "Columnas con tipo incorrecto: "
            f"{columnas_no_numericas or 'ninguna'}"
        )
    )

    # ---------------------------------------------------------
    # Gate 7: dominios de variables binarias
    # ---------------------------------------------------------

    dominios_invalidos = {}

    for columna in ["income", "sex"]:
        if columna in data.columns:
            valores = set(data[columna].dropna().unique())

            if not valores.issubset({0, 1}):
                dominios_invalidos[columna] = sorted(
                    valores,
                    key=lambda valor: str(valor)
                )

    registrar(
        regla="Dominios binarios",
        condicion=len(dominios_invalidos) == 0,
        detalle=f"Valores inválidos: {dominios_invalidos or 'ninguno'}"
    )

    # ---------------------------------------------------------
    # Gate 8: rangos de negocio
    # ---------------------------------------------------------

    rangos = {
        "age": (17, 90),
        "fnlwgt": (1, None),
        "education-num": (1, 16),
        "capital-gain": (0, 99_999),
        "capital-loss": (0, 4_356),
        "hours-per-week": (1, 99)
    }

    valores_fuera_de_rango = {}

    for columna, (limite_inferior, limite_superior) in rangos.items():
        if columna not in data.columns:
            valores_fuera_de_rango[columna] = "columna ausente"
            continue

        mascara_invalida = data[columna] < limite_inferior

        if limite_superior is not None:
            mascara_invalida |= data[columna] > limite_superior

        cantidad_invalida = int(mascara_invalida.sum())

        if cantidad_invalida > 0:
            valores_fuera_de_rango[columna] = cantidad_invalida

    registrar(
        regla="Rangos de negocio",
        condicion=len(valores_fuera_de_rango) == 0,
        detalle=(
            "Filas fuera de rango por columna: "
            f"{valores_fuera_de_rango or 'ninguna'}"
        )
    )

    # ---------------------------------------------------------
    # Gate 9: consistencia education / education-num
    # ---------------------------------------------------------

    if {"education", "education-num"}.issubset(data.columns):
        cantidad_codigos = data.groupby(
            "education",
            dropna=False
        )["education-num"].nunique()

        educaciones_inconsistentes = cantidad_codigos[
            cantidad_codigos != 1
        ].index.tolist()
    else:
        educaciones_inconsistentes = ["columnas ausentes"]

    registrar(
        regla="Consistencia education / education-num",
        condicion=len(educaciones_inconsistentes) == 0,
        detalle=(
            "Categorías inconsistentes: "
            f"{educaciones_inconsistentes or 'ninguna'}"
        )
    )

    # ---------------------------------------------------------
    # Gate 10: calidad de la variable objetivo
    # ---------------------------------------------------------

    if target in data.columns:
        target_sin_nulos = data[target].notna().all()
        clases_target = set(data[target].dropna().unique())
        target_binario = clases_target == {0, 1}
        proporciones_target = data[target].value_counts(normalize=True)
        proporcion_minoritaria = (
            proporciones_target.min()
            if len(proporciones_target) > 0
            else 0
        )
    else:
        target_sin_nulos = False
        target_binario = False
        clases_target = set()
        proporcion_minoritaria = 0

    target_valido = (
        target_sin_nulos
        and target_binario
        and proporcion_minoritaria >= min_target_proportion
    )

    registrar(
        regla="Calidad de la variable objetivo",
        condicion=target_valido,
        detalle=(
            f"Clases: {sorted(clases_target, key=str)}; "
            f"proporción minoritaria: {proporcion_minoritaria:.2%}; "
            f"mínimo requerido: {min_target_proportion:.2%}"
        )
    )

    # ---------------------------------------------------------
    # Reporte y decisión final
    # ---------------------------------------------------------

    reporte = pd.DataFrame(resultados)

    print("=" * 75)
    print("REPORTE DE DATA QUALITY GATES")
    print("=" * 75)

    display(reporte)

    reglas_fallidas = reporte[reporte["estado"] == "FALLIDA"]

    if not reglas_fallidas.empty:
        nombres = reglas_fallidas["regla"].tolist()

        raise ValueError(
            "El entrenamiento fue detenido porque fallaron "
            f"{len(nombres)} reglas de calidad: {nombres}"
        )

    print(
        f"\nTodas las reglas fueron aprobadas "
        f"({len(reporte)}/{len(reporte)})."
    )
    print("El dataset está autorizado para continuar al entrenamiento.")

    return reporte