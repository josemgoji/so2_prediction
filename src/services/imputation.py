import warnings

import numpy as np
import pandas as pd
from pmdarima import auto_arima
import plotly.graph_objects as go

warnings.filterwarnings("ignore", category=FutureWarning)


class ImputationResult:
    """Contiene el DataFrame imputado y un DataFrame de logs."""
    def __init__(self, df: pd.DataFrame, logs: pd.DataFrame = None, arima_params: dict | None = None):
        self.df = df
        self.logs = logs
        self.arima_params = arima_params or {}


class ImputationService:
    """
    Imputación de series temporales:
    - Interpolación para gaps pequeños (<= small_gap_threshold)
    - Selección del mejor fragmento para ajustar ARIMA (estrategia configurable)
    - Imputación del resto con parámetros ARIMA fijados
    - Gaps > chunk_size se imputan por trozos de tamaño chunk_size, recalculando con la serie extendida
    
    Estrategias de selección de fragmentos:
    - "length": Selecciona el fragmento más largo (comportamiento original)
    - "hybrid": Selecciona basado en múltiples criterios:
      * Length (40%): Longitud del fragmento normalizada
      * Completeness (25%): Porcentaje de datos no nulos
      * Density (20%): Densidad de datos con penalización por fragmentación
      * SARIMA fit (15%): Calidad del ajuste ARIMA evaluada con validación cruzada
    """

    def __init__(
        self,
        small_gap_threshold: int = 9,  # 9
        select_gap_threshold: int = 24,  # 72  #24
        chunk_size: int = 25,  # 73 # 25
        min_obs_for_arima: int = 50,  # mínimo de observaciones no nulas para ajustar ARIMA
        use_fragment_min: bool = True,  # si True, usa mínimo del fragmento; si False, usa 0 como límite
        fragment_selection_strategy: str = "length",  # "length" o "hybrid" (length + completeness + density + sarima_fit)
    ):
        self.small_gap_threshold = small_gap_threshold
        self.select_gap_threshold = select_gap_threshold
        self.chunk_size = chunk_size
        self.min_obs_for_arima = min_obs_for_arima
        self.use_fragment_min = use_fragment_min
        self.fragment_selection_strategy = fragment_selection_strategy
        

    # ----------------------------
    # Utilidades de detección de gaps y fragmentos
    # ----------------------------
    @staticmethod
    def _find_gaps_mask(is_nan: pd.Series) -> list[tuple[int, int]]:
        """Devuelve lista de (start_idx, end_idx_no_inclusivo) para runs de NaN consecutivos."""
        gaps = []
        i = 0
        N = len(is_nan)
        while i < N:
            if is_nan.iloc[i]:
                s = i
                while i < N and is_nan.iloc[i]:
                    i += 1
                e = i
                gaps.append((s, e))
            else:
                i += 1
        return gaps

    @staticmethod
    def _find_blocks_mask(is_nan: pd.Series) -> list[tuple[int, int]]:
        """Devuelve lista de (start_idx, end_idx_no_inclusivo) para runs observados (no-NaN) consecutivos."""
        blocks = []
        i = 0
        N = len(is_nan)
        while i < N:
            if not is_nan.iloc[i]:
                s = i
                while i < N and not is_nan.iloc[i]:
                    i += 1
                e = i
                blocks.append((s, e))
            else:
                i += 1
        return blocks

    def _split_fragments_by_large_gaps(self, df: pd.DataFrame, col: str) -> list[tuple[int, int]]:
        """
        Divide la serie en fragmentos separados por gaps > select_gap_threshold.
        Devuelve lista de (start_idx, end_idx_no_inclusivo) de fragmentos.
        """
        is_nan = df[col].isna()
        gaps = self._find_gaps_mask(is_nan)

        # puntos de corte: gaps estrictamente mayores a select_gap_threshold
        cut_points = [0]
        for s, e in gaps:
            gap_len = e - s
            if gap_len > self.select_gap_threshold:
                cut_points.extend([s, e])
        cut_points.append(len(df))

        # construir fragmentos a partir de cut_points consecutivos (pares)
        fragments = []
        for i in range(0, len(cut_points) - 1, 2):
            a = cut_points[i]
            b = cut_points[i + 1]
            # evitar fragmentos vacíos
            if b - a > 0:
                fragments.append((a, b))
        # si cut_points tuvo longitud impar por estructura, cubrir tramo final
        if len(cut_points) % 2 == 1:
            a = cut_points[-1]
            b = len(df)
            if b - a > 0:
                fragments.append((a, b))

        # limpiar posibles solapes/duplicados y ordenar
        clean = []
        for s, e in fragments:
            if not clean or s >= clean[-1][1]:
                clean.append((s, e))
        return clean

    def _calculate_fragment_density(self, fragment_data: pd.Series) -> float:
        """
        Calcula la densidad de datos en un fragmento.
        Mide qué tan concentrados están los datos no nulos.
        """
        if len(fragment_data) == 0:
            return 0.0
        
        # Contar observaciones no nulas
        non_null_count = fragment_data.notna().sum()
        if non_null_count == 0:
            return 0.0
        
        # Calcular densidad como proporción de datos no nulos
        density = non_null_count / len(fragment_data)
        
        # Penalizar fragmentos con gaps muy dispersos
        # Si hay muchos gaps pequeños dispersos, la densidad efectiva es menor
        is_na = fragment_data.isna()
        if is_na.any():
            # Encontrar gaps
            gap_starts = is_na & ~is_na.shift(1, fill_value=False)
            gap_count = gap_starts.sum()
            
            # Penalizar por número de gaps (fragmentación)
            fragmentation_penalty = min(0.2, gap_count * 0.01)  # Máximo 20% de penalización
            density = density * (1 - fragmentation_penalty)
        
        return max(0.0, min(1.0, density))  # Asegurar rango [0,1]

    def _evaluate_sarima_fit(self, fragment_data: pd.Series) -> float:
        """
        Evalúa la calidad del ajuste SARIMA en un fragmento.
        Usa validación cruzada temporal simple con parámetros adaptativos.
        """
        if len(fragment_data) < 20:  # Mínimo para validación
            return 0.0
        
        try:
            # Interpolar gaps pequeños dentro del fragmento
            frag_interp = self.impute_small_gaps(fragment_data.to_frame(name='temp'), 'temp')['temp']
            y = frag_interp.dropna().astype(float).values
            
            if len(y) < 15:  # Mínimo absoluto
                return 0.0
            
            # Dividir en train/test temporal (80/20)
            split_point = int(len(y) * 0.8)
            train = y[:split_point]
            test = y[split_point:]
            
            if len(test) < 3:  # Test muy pequeño
                return 0.0
            
            # Determinar límites adaptativos basados en la cantidad de datos
            n_train = len(train)
            
            # Reglas adaptativas para evitar modelos demasiado complejos:
            # - Para < 50 datos: máximo (1,1,1)
            # - Para 50-100 datos: máximo (2,1,2) 
            # - Para 100-200 datos: máximo (3,2,3)
            # - Para > 200 datos: máximo (4,2,4)
            if n_train < 50:
                max_p, max_d, max_q = 1, 1, 1
            elif n_train < 100:
                max_p, max_d, max_q = 2, 1, 2
            elif n_train < 200:
                max_p, max_d, max_q = 3, 2, 3
            else:
                max_p, max_d, max_q = 4, 2, 4
            
            # Verificar estacionariedad básica antes de ajustar
            # Si la serie es muy volátil, usar diferenciación más conservadora
            train_std = np.std(train)
            train_mean = np.mean(train)
            cv = train_std / (train_mean + 1e-8)  # Coeficiente de variación
            
            if cv > 1.0:  # Serie muy volátil
                max_d = min(max_d, 1)  # Limitar diferenciación
            
            # Ajustar ARIMA con parámetros adaptativos
            model = auto_arima(
                train, 
                seasonal=False, 
                stepwise=True, 
                suppress_warnings=True,
                error_action='ignore',  # Ignorar errores de ajuste
                max_p=max_p,
                max_d=max_d, 
                max_q=max_q,
                start_p=0,
                start_q=0,
                start_d=0,
                # Parámetros adicionales para estabilidad
                max_order=min(10, n_train // 10),  # Orden total máximo
                information_criterion='aic',  # Usar AIC para selección
                n_fits=10,  # Número de modelos a probar
                with_intercept=True  # Incluir intercepto
            )
            
            # Verificar que el modelo se ajustó correctamente
            if hasattr(model, 'arima_res_') and model.arima_res_ is not None:
                predictions = model.predict(n_periods=len(test))
                
                # Verificar que las predicciones son válidas
                if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                    return 0.1  # Score bajo si predicciones inválidas
                
                # Calcular MAPE
                mape = np.mean(np.abs((test - predictions) / (test + 1e-8))) * 100
                
                # Convertir MAPE a score (menor MAPE = mejor score)
                if mape < 10:
                    fit_score = 1.0
                elif mape < 20:
                    fit_score = 0.8
                elif mape < 30:
                    fit_score = 0.6
                else:
                    fit_score = max(0.0, 0.4 - (mape - 30) * 0.01)
                
                return fit_score
            else:
                # Modelo no se ajustó correctamente
                return 0.1
                
        except Exception:
            # Log del error para debugging (opcional)
            # print(f"Error en _evaluate_sarima_fit: {str(e)}")
            return 0.1

    def _select_best_fragment(self, df: pd.DataFrame, col: str) -> tuple[pd.Series, tuple[int, int]]:
        """
        Selecciona el mejor fragmento según la estrategia configurada.
        - "length": Selecciona el fragmento más largo (comportamiento original)
        - "hybrid": Selecciona basado en length + completeness + density + sarima_fit
        
        Devuelve (serie_fragmento, (start_idx, end_idx_no_inclusivo)).
        """
        fragments = self._split_fragments_by_large_gaps(df, col)
        
        if not fragments:
            # Sin cortes -> toda la serie
            return df[col], (0, len(df))

        if self.fragment_selection_strategy == "length":
            # Estrategia original: seleccionar el fragmento más largo
            best_fragment = max(fragments, key=lambda x: x[1] - x[0])
            s, e = best_fragment
            
            print(f"\n=== FRAGMENTO SELECCIONADO (Estrategia: {self.fragment_selection_strategy}) ===")
            print(f"Índices: {s} a {e-1}")
            print(f"Fechas: {df.index[s]} a {df.index[e-1]}")
            print(f"Longitud: {e-s} observaciones")
            print("=" * 50)
            
            return df[col].iloc[s:e], (s, e)
        
        elif self.fragment_selection_strategy == "hybrid":
            # Estrategia híbrida: evaluar múltiples criterios
            fragment_scores = []
            
            print(f"\n=== EVALUACIÓN DE FRAGMENTOS (Estrategia: {self.fragment_selection_strategy}) ===")
            print(f"Total de fragmentos encontrados: {len(fragments)}")
            print()
            
            for i, (s, e) in enumerate(fragments):
                fragment_data = df[col].iloc[s:e]
                
                # 1. Length score (normalizado por min_obs_for_arima)
                length_score = min(1.0, len(fragment_data) / self.min_obs_for_arima)
                
                # 2. Completeness score (% de datos no nulos)
                completeness_score = fragment_data.notna().sum() / len(fragment_data)
                
                # 3. Density score (densidad con penalización por fragmentación)
                density_score = self._calculate_fragment_density(fragment_data)
                
                # 4. SARIMA fit score (calidad del ajuste)
                sarima_fit_score = self._evaluate_sarima_fit(fragment_data)
                
                # Puntuación combinada con pesos
                # Pesos: 40% length, 25% completeness, 20% density, 15% sarima_fit
                total_score = (
                    0.6 * length_score +
                    0.1 * completeness_score +
                    0.2 * density_score +
                    0.1 * sarima_fit_score
                )
                
                fragment_scores.append((total_score, (s, e)))
                
                # Print informativo para testing
                print(f"Fragmento {i+1}: {df.index[s]} a {df.index[e-1]}")
                print(f"  Longitud: {len(fragment_data)} obs | Length score: {length_score:.3f}")
                print(f"  Completitud: {completeness_score:.3f} | Density score: {density_score:.3f}")
                print(f"  SARIMA fit score: {sarima_fit_score:.3f}")
                print(f"  PUNTUACIÓN TOTAL: {total_score:.3f}")
                print()
            
            # Seleccionar el fragmento con mayor puntuación
            best_fragment = max(fragment_scores, key=lambda x: x[0])[1]
            s, e = best_fragment
            
            print("=== FRAGMENTO SELECCIONADO ===")
            print(f"Índices: {s} a {e-1}")
            print(f"Fechas: {df.index[s]} a {df.index[e-1]}")
            print(f"Longitud: {e-s} observaciones")
            print(f"Puntuación: {max(fragment_scores, key=lambda x: x[0])[0]:.3f}")
            print("=" * 50)
            
            return df[col].iloc[s:e], (s, e)
        
        else:
            # Fallback a estrategia original si se especifica una estrategia inválida
            best_fragment = max(fragments, key=lambda x: x[1] - x[0])
            s, e = best_fragment
            return df[col].iloc[s:e], (s, e)


    # ----------------------------
    # Interpolación de gaps pequeños
    # ----------------------------
    def impute_small_gaps(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Imputa gaps pequeños (<= small_gap_threshold) por interpolación lineal.
        """
        is_na = df[col].isna()
        group = (is_na != is_na.shift()).cumsum()
        group_sizes = is_na.groupby(group).transform('sum')
        mask = (is_na) & (group_sizes <= self.small_gap_threshold)

        df_imputed = df.copy()
        interpolated = df[col].interpolate()
        df_imputed.loc[mask, col] = interpolated[mask]
        return df_imputed

    # ----------------------------
    # Simulación ARIMA (parámetros fijos)
    # ----------------------------
    @staticmethod
    def _simula_arima_serie_param(serie: np.ndarray, n_simular: int, p: int, d: int, q: int,
                                  ar_coefs: list[float], ma_coefs: list[float], min_value: float = 0.0) -> list[float]:
        """Simula n pasos adelante usando parámetros ARIMA fijos sobre la serie observada."""
        serie_proc = pd.Series(serie, copy=True)
        for _ in range(d):
            serie_proc = serie_proc.diff().dropna().reset_index(drop=True)

        P = p
        Q = q
        maxpq = max(P, Q)

        # construir residuales históricos bajo ARMA(p,q) sobre la serie diferenciada
        errores = [0.0] * maxpq
        for t in range(maxpq, len(serie_proc)):
            pred_ar = sum(ar_coefs[i] * serie_proc.iloc[t - i - 1] for i in range(P)) if P > 0 else 0.0
            pred_ma = sum(ma_coefs[i] * errores[t - i - 1]           for i in range(Q)) if Q > 0 else 0.0
            pred = pred_ar + pred_ma
            err = float(serie_proc.iloc[t] - pred)
            errores.append(err)

        errores_hist = np.array(errores[maxpq:]) if len(errores) > maxpq else np.array([0.0])
        media_err = float(np.mean(errores_hist))
        std_err = float(np.std(errores_hist)) if errores_hist.size > 1 else 1.0  # fallback

        # colas de estado para simulación
        y_state = list(serie_proc.iloc[-maxpq:]) if maxpq > 0 else []
        e_state = list(errores_hist[-maxpq:])    if maxpq > 0 else []

        # rellenar si faltan
        while len(y_state) < P:
            y_state.insert(0, 0.0)
        while len(e_state) < Q:
            e_state.insert(0, 0.0)

        sims = []
        for _ in range(n_simular):
            pred_ar = sum(ar_coefs[i] * y_state[-i-1] for i in range(P)) if P > 0 else 0.0
            pred_ma = sum(ma_coefs[i] * e_state[-i-1] for i in range(Q)) if Q > 0 else 0.0
            pred = pred_ar + pred_ma
            e_new = np.random.normal(loc=media_err, scale=max(std_err, 1e-8))
            y_new = pred + e_new
            y_state.append(float(y_new))
            e_state.append(float(e_new))
            sims.append(float(y_new))

        # des-diferenciar
        nuevos = sims
        if d > 0:
            ultimos_original = serie[-d:]
            for i in range(d):
                nuevos = list(np.cumsum(np.concatenate(([ultimos_original[i]], nuevos))))[:-1]

        # recorte a valores - nunca permitir valores negativos
        nuevos = [max(min_value, x) for x in nuevos]
        return nuevos

    # ----------------------------
    # Ajuste ARIMA (auto_arima) a partir del mejor fragmento
    # ----------------------------
    def _fit_arima_from_best_fragment(self, df: pd.DataFrame, col: str) -> dict:
        """Devuelve dict con p,d,q, ar_coefs, ma_coefs, fragmento (índices)."""
        frag_series, (s, e) = self._select_best_fragment(df, col)

        # antes de ajustar, interpolamos SOLO gaps pequeños dentro del fragmento
        frag_series_interp = self.impute_small_gaps(frag_series.to_frame(name=col), col)[col]
        y = frag_series_interp.dropna().astype(float).values

        if len(y) < self.min_obs_for_arima:
            raise ValueError(
                f"No hay suficientes observaciones en el mejor fragmento para ajustar ARIMA "
                f"(mínimo requerido={self.min_obs_for_arima}, disponibles={len(y)})."
            )

        modelo = auto_arima(y, seasonal=False, stepwise=True, suppress_warnings=True)
        p, d, q = modelo.order
        ar = list(modelo.arparams()) if p > 0 else []
        ma = list(modelo.maparams()) if q > 0 else []
        
        # Calcular el valor mínimo del fragmento para usar como límite inferior
        fragment_min_value = float(frag_series_interp.min())

        return {
            "p": p, "d": d, "q": q,
            "ar_coefs": ar,
            "ma_coefs": ma,
            "fragment_index": (s, e),
            "fragment_start_date": df.index[s],
            "fragment_end_date": df.index[e - 1],
            "fragment_length": e - s,
            "fragment_min_value": fragment_min_value
        }

    # ----------------------------
    # Imputación de gaps con parámetros fijos y troceo para gaps grandes
    # ----------------------------
    def _impute_with_fixed_arima(self, df: pd.DataFrame, col: str, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Imputa todos los gaps restantes con parámetros ARIMA fijos.
        - gaps <= small_gap_threshold ya están imputados por interpolación
        - gaps entre small_gap_threshold y chunk_size: 1 sola simulación
        - gaps > chunk_size: simulación en trozos de chunk_size, actualizando la serie para la siguiente tanda
        """
        p = params["p"]
        d = params["d"]
        q = params["q"]
        ar = params["ar_coefs"]
        ma = params["ma_coefs"]
        fragment_min_value = params["fragment_min_value"]

        df_imp = df.copy()
        logs = []

        is_nan = df_imp[col].isna()
        gaps = self._find_gaps_mask(is_nan)

        for s, e in gaps:
            gap_len = e - s
            if gap_len <= self.small_gap_threshold:
                # estos deberían estar ya imputados; si no, los imputamos rápido con interpolación local
                local = df_imp[col].iloc[max(0, s-3):min(len(df_imp), e+3)].interpolate()
                df_imp.iloc[s:e, df_imp.columns.get_loc(col)] = local.iloc[(s - max(0, s-3)):(s - max(0, s-3) + gap_len)].values
                logs.append({
                    "gap_inicio": df_imp.index[s],
                    "gap_fin": df_imp.index[e-1],
                    "gap_len": gap_len,
                    "metodo": f"interpolate_fallback(<= {self.small_gap_threshold})"
                })
                continue

            # serie observada disponible hasta el inicio del gap (ya con pequeños imputados)
            pos = s
            restante = gap_len
            modo = "trozos" if gap_len > self.chunk_size else "directo"

            while restante > 0:
                paso = min(self.chunk_size, restante)
                serie_obs = df_imp[col].iloc[:pos].dropna().values.astype(float)

                # sanity check para d/p/q
                if len(serie_obs) < max(p, q) + d + 5:
                    # si no hay suficiente historial, hacemos interpolación local como fallback
                    local = df_imp[col].iloc[max(0, s-3):min(len(df_imp), e+3)].interpolate()
                    df_imp.iloc[s:e, df_imp.columns.get_loc(col)] = local.iloc[(s - max(0, s-3)):(s - max(0, s-3) + gap_len)].values
                    logs.append({
                        "gap_inicio": df_imp.index[s],
                        "gap_fin": df_imp.index[e-1],
                        "gap_len": gap_len,
                        "metodo": "interpolate_fallback_historia_insuficiente"
                    })
                    restante = 0
                    break

                # Determinar el límite inferior según configuración
                limite_inferior = fragment_min_value if self.use_fragment_min else 0.0
                nuevos = self._simula_arima_serie_param(serie_obs, paso, p, d, q, ar, ma, limite_inferior)
                df_imp.iloc[pos:pos+paso, df_imp.columns.get_loc(col)] = nuevos

                logs.append({
                    "gap_inicio": df_imp.index[s] if pos == s else df_imp.index[pos],
                    "gap_fin": df_imp.index[min(e-1, pos+paso-1)],
                    "gap_len": paso,
                    "metodo": f"ARIMA_fixed_params_{modo}",
                    "p": p, "d": d, "q": q,
                    "valor_minimo_fragmento": fragment_min_value,
                    "limite_inferior_aplicado": limite_inferior
                })

                pos += paso
                restante -= paso

        return df_imp, pd.DataFrame(logs)

    # ----------------------------
    # API principal
    # ----------------------------
    def impute(
        self,
        df: pd.DataFrame,
        col: str = 'target',
        start_date: str | None = None,
        end_date: str | None = None
    ) -> ImputationResult:
        """
        Flujo completo:
        1) (Opcional) Filtrado por fechas
        2) Interpolación de gaps pequeños (<= small_gap_threshold)
        3) Selección del fragmento más largo y ajuste ARIMA
        4) Imputación del resto con parámetros fijos (trozos de chunk_size si gap > chunk_size)
        """
        if start_date or end_date:
            df_work = df.loc[start_date:end_date].copy()
        else:
            df_work = df.copy()

        if not isinstance(df_work.index, pd.DatetimeIndex):
            raise ValueError("El DataFrame debe tener un DatetimeIndex.")

        # Paso 1: imputar gaps pequeños por interpolación
        df_small = self.impute_small_gaps(df_work, col)

        # Paso 2: elegir mejor fragmento y ajustar ARIMA
        params = self._fit_arima_from_best_fragment(df_small, col)

        # Paso 3: imputar el resto con parámetros fijos (troceo para gaps grandes)
        df_final, logs = self._impute_with_fixed_arima(df_small, col, params)

        # Agregar información de fragmentos al inicio de los logs
        fragment_summary = {
            "gap_inicio": "FRAGMENT_SUMMARY",
            "gap_fin": "FRAGMENT_SUMMARY", 
            "gap_len": 1,
            "metodo": "Fragmento seleccionado por longitud",
            "p": params["p"],
            "d": params["d"], 
            "q": params["q"],
            "fragmento_seleccionado": f"{params['fragment_start_date']} a {params['fragment_end_date']}",
            "fragmento_longitud": params["fragment_length"],
            "valor_minimo_fragmento": params["fragment_min_value"],
            "use_fragment_min": self.use_fragment_min,
            "limite_inferior_imputacion": params["fragment_min_value"] if self.use_fragment_min else 0.0
        }
        
        # Insertar el resumen al inicio de los logs
        logs_with_summary = pd.concat([pd.DataFrame([fragment_summary]), logs], ignore_index=True)

        return ImputationResult(df_final, logs_with_summary, arima_params=params)

    # ----------------------------
    # Plot helper
    # ----------------------------
    def plot_imputation(self, original_df: pd.DataFrame, imputed_df: pd.DataFrame, col: str = 'target', pollutant: str = "SO2", station: str = "", save_path: str = None):
        """
        Grafica original vs imputados (marca en rojo lo que antes era NaN y ahora no lo es).
        """
        imputados_mask = original_df[col].isna() & imputed_df[col].notna()
        imputados_plot = np.full_like(imputed_df[col], np.nan, dtype=np.float64)
        imputados_plot[imputados_mask] = imputed_df[col][imputados_mask]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=original_df.index, y=original_df[col],
            mode='lines', name='Original',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=imputed_df.index, y=imputados_plot,
            mode='lines', name='Imputados',
            line=dict(color='red', width=2)
        ))
        fig.update_layout(
            title=f'Serie original y valores imputados {station} {pollutant}',
            xaxis_title='Fecha',
            yaxis_title=pollutant,
            legend=dict(x=0.01, y=0.99)
        )
        if save_path:
            fig.write_html(save_path)
        return fig
