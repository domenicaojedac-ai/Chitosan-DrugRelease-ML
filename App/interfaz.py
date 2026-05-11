import os
import joblib
import numpy as np
import pandas as pd
import tkinter as tk

from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) 
# ----- MODELO DE EE% -----
EE_MODEL_PATH = os.path.join(BASE_DIR, "Models", "best_EE_model.pkl")

# ----- MODELO DE CURVA -----
CURVE_MODEL_PATH = os.path.join(BASE_DIR, "Models", "xgboost_weibull_method1.joblib")

# ----- BASES DE DATOS -----
EE_DATA_FILE_PATH = os.path.join(BASE_DIR, "Data", "Data_base9.2_EE.xlsx")
CURVE_DATA_FILE_PATH = os.path.join(BASE_DIR, "Data", "Data_base9.1_CURVE.xlsx")



def to_float(x, default=np.nan):
    try:
        s = str(x).strip()
        if s == "" or s.lower() in ["nan", "none", "unknown"]:
            return default
        s = s.replace(",", ".")
        return float(s)
    except:
        return default


def is_finite_number(v):
    return isinstance(v, (int, float, np.floating)) and np.isfinite(v)


def weibull(t, A, k, b):
    t = np.clip(np.asarray(t, dtype=float), 0, None)
    arg = t / np.clip(float(k), 1e-6, None)
    with np.errstate(over="ignore", invalid="ignore"):
        term = np.exp(-np.power(arg, float(b)))
    return float(A) * (1.0 - term)

# PARA CARGAR EL MODELO DE CURVA
# =========================================================================
class SmoothedTargetEncoder:
    def __init__(self, col, target_cols, smoothing=10):
        self.col = col
        self.target_cols = target_cols
        self.smoothing = smoothing
        self.maps = {}
        self.global_means = {}

    def fit(self, df):
        for t_col in self.target_cols:
            self.global_means[t_col] = float(df[t_col].mean())
            stats = df.groupby(self.col)[t_col].agg(['count', 'mean'])
            numerator = (stats['count'] * stats['mean'] + self.smoothing * self.global_means[t_col])
            denominator = (stats['count'] + self.smoothing)
            self.maps[t_col] = (numerator / denominator).to_dict()
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for t_col in self.target_cols:
            new_col = f"{self.col}_enc_{t_col}"
            out[new_col] = df[self.col].map(self.maps[t_col]).fillna(self.global_means[t_col])
        return out


def ensure_pickle_compatibility():
    try:
        import __main__
        if not hasattr(__main__, "SmoothedTargetEncoder"):
            __main__.SmoothedTargetEncoder = SmoothedTargetEncoder
    except Exception:
        pass

    try:
        from sklearn.compose import _column_transformer
        if not hasattr(_column_transformer, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass

            _RemainderColsList.__module__ = _column_transformer.__name__
            _RemainderColsList.__qualname__ = "_RemainderColsList"
            _column_transformer._RemainderColsList = _RemainderColsList
    except Exception:
        pass


class ToolTip:
    def __init__(self, widget, text_func):
        self.widget = widget
        self.text_func = text_func
        self.tip_window = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        text = self.text_func() if callable(self.text_func) else str(self.text_func)
        if not text or self.tip_window is not None:
            return

        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(
            self.tip_window,
            text=text,
            justify="left",
            background="#fff8dc",
            relief="solid",
            borderwidth=1,
            padding=(7, 5),
            wraplength=320
        )
        label.pack()

    def hide(self, event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


# APP
# =========================================================================
class HydrogelApp:
    TEXTS = {
        "es": {
            "app_title": "Simulador de Hidrogeles: Curva de Liberacion + EE%",
            "language": "Idioma:",
            "raw_material": "1. MATERIA PRIMA",
            "drug": "Farmaco:",
            "drug_props": "Propiedades del farmaco",
            "polymer": "Quitosano (Matriz):",
            "tab_opt": "Optimizador de formulaciones",
            "tab_curve": "Curva de Liberacion",
            "tab_ee": "Eficiencia EE%",
            "formulation_vars": "Variables de Formulacion",
            "graph_config": "Configuracion de grafica",
            "time_max": "Tiempo Max (h):",
            "release_max": "Liberacion Max (%):",
            "generate_curve": "GENERAR CURVA",
            "weibull_params": "Parametros Weibull:\nA = {A}\nb = {b}\nk = {k}",
            "calc_ee": "CALCULAR EE%",
            "ee_intro": "Introduce los parametros y pulsa Calcular",
            "ee_done": "EE% estimada por el modelo entrenado",
            "objective": "Objetivo",
            "target_release": "Liberacion objetivo (%):",
            "target_time": "Tiempo (h):",
            "iterations": "Iteraciones:",
            "constraints": "Restricciones (vacio = libre)",
            "start_search": "INICIAR BUSQUEDA",
            "ready": "Listo",
            "optimizing": "Optimizando...",
            "evaluated": "Evaluados {done}/{total} | mejor error: {err:.3f}",
            "invalid_candidates": "Sin candidatos validos",
            "finished": "Finalizado | mejor error: {err:.3f}",
            "best_result": "MEJOR RESULTADO",
            "predicted_release": "Liberacion predicha = {y:.2f}% a t = {t} h",
            "goal": "Meta = {target:.2f}%",
            "error": "Error = {err:.2f}",
            "requested_iterations": "Iteraciones solicitadas = {iterations}",
            "valid_candidates": "Candidatos validos = {evaluated}",
            "suggested_formulation": "Formulacion sugerida:",
            "warning": "Advertencia",
            "error_title": "Error",
            "check_target_time": "Revisa el objetivo y el tiempo.",
            "check_iterations": "Revisa el numero de iteraciones.",
            "iteration_range": "Usa entre 100 y 30000 iteraciones.",
            "no_candidates": "No se pudo evaluar candidatos.",
            "tmax_warning": "El tiempo maximo debe ser mayor que 0.",
            "curve_error": "La prediccion de la curva fallo:\n\n{error}",
            "ee_error": "La prediccion de EE% fallo:\n\n{error}",
            "model_load_error": "No se pudieron cargar los modelos:\n\n{error}",
            "numeric_required": "{field} debe ser numerico.",
            "range_required": "{field} debe estar entre {lo:g} y {hi:g}.",
            "x_time": "Tiempo (h)",
            "y_release": "% Liberado",
        },
        "en": {
            "app_title": "Hydrogel Simulator: Release Curve + EE%",
            "language": "Language:",
            "raw_material": "1. RAW MATERIAL",
            "drug": "Drug:",
            "drug_props": "Drug properties",
            "polymer": "Chitosan (Matrix):",
            "tab_opt": "Formulation optimizer",
            "tab_curve": "Release Curve",
            "tab_ee": "EE% Efficiency",
            "formulation_vars": "Formulation Variables",
            "graph_config": "Graph Settings",
            "time_max": "Max Time (h):",
            "release_max": "Max Release (%):",
            "generate_curve": "GENERATE CURVE",
            "weibull_params": "Weibull Parameters:\nA = {A}\nb = {b}\nk = {k}",
            "calc_ee": "CALCULATE EE%",
            "ee_intro": "Enter the parameters and press Calculate",
            "ee_done": "EE% estimated by the trained model",
            "objective": "Objective",
            "target_release": "Target release (%):",
            "target_time": "Time (h):",
            "iterations": "Iterations:",
            "constraints": "Constraints (empty = free)",
            "start_search": "START SEARCH",
            "ready": "Ready",
            "optimizing": "Optimizing...",
            "evaluated": "Evaluated {done}/{total} | best error: {err:.3f}",
            "invalid_candidates": "No valid candidates",
            "finished": "Finished | best error: {err:.3f}",
            "best_result": "BEST RESULT",
            "predicted_release": "Predicted release = {y:.2f}% at t = {t} h",
            "goal": "Target = {target:.2f}%",
            "error": "Error = {err:.2f}",
            "requested_iterations": "Requested iterations = {iterations}",
            "valid_candidates": "Valid candidates = {evaluated}",
            "suggested_formulation": "Suggested formulation:",
            "warning": "Warning",
            "error_title": "Error",
            "check_target_time": "Check the target and time.",
            "check_iterations": "Check the number of iterations.",
            "iteration_range": "Use between 100 and 30000 iterations.",
            "no_candidates": "No candidates could be evaluated.",
            "tmax_warning": "The maximum time must be greater than 0.",
            "curve_error": "Release curve prediction failed:\n\n{error}",
            "ee_error": "EE% prediction failed:\n\n{error}",
            "model_load_error": "Models could not be loaded:\n\n{error}",
            "numeric_required": "{field} must be numeric.",
            "range_required": "{field} must be between {lo:g} and {hi:g}.",
            "x_time": "Time (h)",
            "y_release": "% Released",
        },
    }

    FIELD_LABELS = {
        "es": {
            "Drug_MW_g_mol": "Peso molecular",
            "Drug_LogP": "LogP",
            "Drug_HBD": "Donadores H",
            "Drug_HBA": "Aceptores H",
            "Drug_pKa": "pKa",
            "CS_MW_kDa": "Peso molecular CS",
            "CS_DD_percent": "Desacetilacion CS (%)",
            "CS_Conc_wv_percent": "Concentracion CS (% p/v)",
            "Crosslinker_Name": "Reticulante",
            "Prep_Method": "Metodo de preparacion",
            "Geometry": "Geometria",
            "Medium_pH": "pH del medio",
            "Temp_C": "Temperatura (C)",
            "Crosslinker_Conc_percent": "Concentracion reticulante (%)",
            "CS_Crosslinker_Ratio": "Relacion quitosano/reticulante",
            "Drug_Mass_Initial_mg": "Masa inicial de farmaco (mg)",
            "Matrix_Mass": "Masa de matriz",
        },
        "en": {
            "Drug_MW_g_mol": "Molecular weight",
            "Drug_LogP": "LogP",
            "Drug_HBD": "H-bond donors",
            "Drug_HBA": "H-bond acceptors",
            "Drug_pKa": "pKa",
            "CS_MW_kDa": "CS molecular weight",
            "CS_DD_percent": "CS deacetylation (%)",
            "CS_Conc_wv_percent": "CS concentration (% w/v)",
            "Crosslinker_Name": "Crosslinker",
            "Prep_Method": "Preparation method",
            "Geometry": "Geometry",
            "Medium_pH": "Medium pH",
            "Temp_C": "Temperature (C)",
            "Crosslinker_Conc_percent": "Crosslinker concentration (%)",
            "CS_Crosslinker_Ratio": "Chitosan/crosslinker ratio",
            "Drug_Mass_Initial_mg": "Initial drug mass (mg)",
            "Matrix_Mass": "Matrix mass",
        },
    }

    FIELD_TOOLTIPS = {
        "es": {
            "Drug_MW_g_mol": "Peso molecular del farmaco en g/mol. Se llena automaticamente segun el farmaco seleccionado.",
            "Drug_LogP": "Coeficiente de particion octanol/agua del farmaco. Indica afinidad lipofila o hidrofila.",
            "Drug_HBD": "Numero de grupos del farmaco que pueden donar puentes de hidrogeno.",
            "Drug_HBA": "Numero de grupos del farmaco que pueden aceptar puentes de hidrogeno.",
            "Drug_pKa": "Constante de acidez del farmaco usada para estimar su estado de ionizacion.",
            "CS_MW_kDa": "Peso molecular del quitosano usado como matriz, en kDa.",
            "CS_DD_percent": "Porcentaje de desacetilacion del quitosano.",
            "CS_Conc_wv_percent": "Concentracion de quitosano en la formulacion, expresada como porcentaje peso/volumen.",
            "Crosslinker_Name": "Agente reticulante usado para formar la red del hidrogel.",
            "Prep_Method": "Metodo de preparacion o fabricacion de la formulacion.",
            "Geometry": "Forma fisica final del sistema: perlas, esfera, membrana, pelicula, etc.",
            "Medium_pH": "pH del medio donde se evalua la liberacion o la eficiencia.",
            "Temp_C": "Temperatura del ensayo o proceso, en grados Celsius.",
            "Crosslinker_Conc_percent": "Porcentaje de concentracion del reticulante en la formulacion.",
            "CS_Crosslinker_Ratio": "Relacion entre quitosano y reticulante. Por ejemplo, 3 significa 3 partes de quitosano por 1 de reticulante.",
            "Drug_Mass_Initial_mg": "Cantidad inicial de farmaco incorporada en la formulacion, en miligramos.",
            "Matrix_Mass": "Masa total de la matriz o hidrogel usada como referencia para el calculo.",
            "entry_tmax": "Tiempo final que se mostrara en la grafica de liberacion.",
            "entry_ymax": "Valor maximo del eje Y para visualizar el porcentaje de liberacion.",
            "entry_target": "Porcentaje de liberacion que quieres alcanzar en el tiempo objetivo.",
            "entry_ttime": "Tiempo especifico en horas donde se evaluara la meta de liberacion.",
            "entry_iterations": "Cantidad de candidatos aleatorios que revisara el optimizador. Mas iteraciones puede mejorar el resultado, pero tarda mas.",
        },
        "en": {
            "Drug_MW_g_mol": "Drug molecular weight in g/mol. It is filled automatically from the selected drug.",
            "Drug_LogP": "Drug octanol/water partition coefficient. It indicates lipophilic or hydrophilic tendency.",
            "Drug_HBD": "Number of drug groups that can donate hydrogen bonds.",
            "Drug_HBA": "Number of drug groups that can accept hydrogen bonds.",
            "Drug_pKa": "Drug acidity constant used to estimate its ionization state.",
            "CS_MW_kDa": "Molecular weight of the chitosan used as matrix, in kDa.",
            "CS_DD_percent": "Chitosan deacetylation percentage.",
            "CS_Conc_wv_percent": "Chitosan concentration in the formulation, expressed as weight/volume percentage.",
            "Crosslinker_Name": "Crosslinking agent used to form the hydrogel network.",
            "Prep_Method": "Preparation or manufacturing method for the formulation.",
            "Geometry": "Final physical shape of the system: beads, sphere, membrane, film, etc.",
            "Medium_pH": "pH of the medium used for release or efficiency evaluation.",
            "Temp_C": "Assay or process temperature in degrees Celsius.",
            "Crosslinker_Conc_percent": "Crosslinker concentration percentage in the formulation.",
            "CS_Crosslinker_Ratio": "Ratio between chitosan and crosslinker. For example, 3 means 3 parts chitosan per 1 part crosslinker.",
            "Drug_Mass_Initial_mg": "Initial amount of drug loaded into the formulation, in milligrams.",
            "Matrix_Mass": "Total matrix or hydrogel mass used as calculation reference.",
            "entry_tmax": "Final time displayed on the release curve graph.",
            "entry_ymax": "Maximum Y-axis value used to visualize release percentage.",
            "entry_target": "Release percentage you want to reach at the target time.",
            "entry_ttime": "Specific time in hours where the release target is evaluated.",
            "entry_iterations": "Number of random candidates checked by the optimizer. More iterations may improve the result, but take longer.",
        },
    }

    def t(self, key, **kwargs):
        lang = getattr(self, "language", "es")
        text = self.TEXTS.get(lang, self.TEXTS["es"]).get(key, key)
        return text.format(**kwargs) if kwargs else text

    def field_label(self, key):
        lang = getattr(self, "language", "es")
        return self.FIELD_LABELS.get(lang, self.FIELD_LABELS["es"]).get(key, key)

    def tooltip_text(self, key):
        lang = getattr(self, "language", "es")
        return self.FIELD_TOOLTIPS.get(lang, self.FIELD_TOOLTIPS["es"]).get(key, "")

    def register_text(self, widget, key, option="text"):
        self.translatable_widgets.append((widget, key, option))
        widget.configure(**{option: self.t(key)})
        return widget

    def register_field_label(self, widget, field_key):
        self.field_label_widgets.append((widget, field_key))
        widget.configure(text=self.field_label(field_key))
        self.add_tooltip(widget, field_key)
        return widget

    def add_tooltip(self, widget, key):
        ToolTip(widget, lambda key=key: self.tooltip_text(key))
        return widget

    def apply_language(self):
        self.root.title(self.t("app_title"))

        for widget, key, option in self.translatable_widgets:
            try:
                widget.configure(**{option: self.t(key)})
            except tk.TclError:
                pass

        for widget, key in self.field_label_widgets:
            try:
                widget.configure(text=self.field_label(key))
            except tk.TclError:
                pass

        if hasattr(self, "notebook"):
            self.notebook.tab(self.tab_opt, text=self.t("tab_opt"))
            self.notebook.tab(self.tab_curve, text=self.t("tab_curve"))
            self.notebook.tab(self.tab_ee, text=self.t("tab_ee"))

        if hasattr(self, "lbl_curve_params"):
            params = getattr(self, "current_curve_params", None)
            if params is None:
                self.lbl_curve_params.config(text=self.t("weibull_params", A="---", b="---", k="---"))
            else:
                A, b, k = params
                self.lbl_curve_params.config(
                    text=self.t("weibull_params", A=f"{A:.4f}", b=f"{b:.4f}", k=f"{k:.4f}")
                )
        if hasattr(self, "lbl_ee_info") and self.lbl_ee_info.cget("text") in [
            self.TEXTS["es"]["ee_intro"],
            self.TEXTS["en"]["ee_intro"],
        ]:
            self.lbl_ee_info.config(text=self.t("ee_intro"))
        if hasattr(self, "lbl_opt_status") and self.lbl_opt_status.cget("text") in [
            self.TEXTS["es"]["ready"],
            self.TEXTS["en"]["ready"],
        ]:
            self.lbl_opt_status.config(text=self.t("ready"))
        if hasattr(self, "last_curve_plot"):
            self.render_curve_plot(*self.last_curve_plot)
        if hasattr(self, "last_opt_result"):
            best, target, t_obj, iterations, evaluated = self.last_opt_result
            self.txt_res.delete(1.0, tk.END)
            self.txt_res.insert(tk.END, self.format_opt_result(best, target, t_obj, iterations, evaluated))
            self.render_opt_curve(best["cand"], target, t_obj, best["y"])

    def on_language_change(self, event=None):
        selected = self.cb_language.get()
        self.language = "en" if selected == "English" else "es"
        self.apply_language()

    def __init__(self, root):
        self.root = root
        self.language = "es"
        self.translatable_widgets = []
        self.field_label_widgets = []
        self.root.title(self.t("app_title"))
        self.root.geometry("1380x920")

        # -------------------------------------------------------------
        # 1) CARGAR MODELOS
        # -------------------------------------------------------------
        try:
            self.load_curve_model(CURVE_MODEL_PATH)
            self.load_ee_model(EE_MODEL_PATH)
        except Exception as e:
            messagebox.showerror(self.t("error_title"), self.t("model_load_error", error=e))
            root.destroy()
            return

       
        # 2) CARGAR BASES 
        # -------------------------------------------------------------
        self.drug_db = {}
        self.available_crosslinkers = ["TPP"]
        self.available_methods = ["Ionic Gelation"]
        self.available_geometries = ["Beads", "Sphere", "Membrane", "Film"]

        self.load_reference_data(EE_DATA_FILE_PATH)
        self.load_reference_data(CURVE_DATA_FILE_PATH)

        if not self.drug_db:
            self.drug_db = {
                "Metronidazole": {
                    "Drug_MW_g_mol": 171.15,
                    "Drug_LogP": -0.1,
                    "Drug_HBD": 1,
                    "Drug_HBA": 5,
                    "Drug_pKa": 2.6,
                },
                "Ibuprofen": {
                    "Drug_MW_g_mol": 206.28,
                    "Drug_LogP": 3.5,
                    "Drug_HBD": 1,
                    "Drug_HBA": 2,
                    "Drug_pKa": 4.9,
                },
            }

        
        # 3) CONFIG UI
        # -------------------------------------------------------------
        self.polymer_defaults = {
            "CS_MW_kDa": 150,
            "CS_DD_percent": 85,
            "CS_Conc_wv_percent": 2.0,
        }

        self.form_defaults = {
            "Crosslinker_Name": "TPP",
            "Prep_Method": "Ionic Gelation",
            "Geometry": "Beads",
            "Medium_pH": 7.4,
            "Temp_C": 37.0,
            "Crosslinker_Conc_percent": 0.5,
            "CS_Crosslinker_Ratio": 3.0,
            "Drug_Mass_Initial_mg": 10.0,
            "Matrix_Mass": 100.0,
            "Theorical_loading": 5.0,
        }

        
        # 4)  ENTRADAS COMUNES
        # -------------------------------------------------------------
        panel_left = ttk.LabelFrame(root, padding=10)
        self.register_text(panel_left, "raw_material")
        panel_left.pack(side="left", fill="y", padx=10, pady=10)

        lang_row = ttk.Frame(panel_left)
        lang_row.pack(fill="x", pady=(0, 10))
        lbl_lang = ttk.Label(lang_row, font=("Arial", 8, "bold"))
        self.register_text(lbl_lang, "language")
        lbl_lang.pack(side="left")
        self.cb_language = ttk.Combobox(lang_row, values=["Espanol", "English"], state="readonly", width=10)
        self.cb_language.set("Espanol")
        self.cb_language.pack(side="right")
        self.cb_language.bind("<<ComboboxSelected>>", self.on_language_change)

        lbl_drug = ttk.Label(panel_left, font=("Arial", 9, "bold"))
        self.register_text(lbl_drug, "drug")
        lbl_drug.pack(anchor="w")
        sorted_drugs = sorted(list(self.drug_db.keys()))
        self.cb_drug = ttk.Combobox(panel_left, values=sorted_drugs, state="readonly")
        if sorted_drugs:
            self.cb_drug.current(0)
        self.cb_drug.pack(fill="x", pady=(0, 10))
        self.cb_drug.bind("<<ComboboxSelected>>", self.fill_drug_props)

        # Propiedades del fármaco
        self.drug_entries = {}
        f_props = ttk.Frame(panel_left)
        f_props.pack(fill="x")
        for k in ["Drug_MW_g_mol", "Drug_LogP", "Drug_HBD", "Drug_HBA", "Drug_pKa"]:
            row = ttk.Frame(f_props)
            row.pack(fill="x", pady=1)
            lbl = ttk.Label(
                row,
                width=18,
                anchor="w",
                font=("Arial", 8)
            )
            self.register_field_label(lbl, k)
            lbl.pack(side="left")
            e = ttk.Entry(row, state="readonly")
            e.pack(side="right", expand=True, fill="x")
            self.drug_entries[k] = e

        ttk.Separator(panel_left, orient="horizontal").pack(fill="x", pady=15)
        lbl_polymer = ttk.Label(panel_left, font=("Arial", 9, "bold"))
        self.register_text(lbl_polymer, "polymer")
        lbl_polymer.pack(anchor="w")

        self.polymer_entries = {}
        for col in ["CS_MW_kDa", "CS_DD_percent", "CS_Conc_wv_percent"]:
            lbl = ttk.Label(panel_left, font=("Arial", 8))
            self.register_field_label(lbl, col)
            lbl.pack(anchor="w")
            e = ttk.Entry(panel_left)
            e.insert(0, str(self.polymer_defaults.get(col, 0)))
            e.pack(fill="x", pady=2)
            self.polymer_entries[col] = e

        self.fill_drug_props(None)

        
        # 5) NOTEBOOK
        # -------------------------------------------------------------
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.tab_opt = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_opt, text=self.t("tab_opt"))
        self.setup_opt_tab()

        self.tab_curve = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_curve, text=self.t("tab_curve"))
        self.setup_curve_tab()

        self.tab_ee = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_ee, text=self.t("tab_ee"))
        self.setup_ee_tab()

    
    # CARGA DE MODELOS
    # =========================================================================
    def load_curve_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el modelo de curva:\n{path}")

        ensure_pickle_compatibility()
        payload = joblib.load(path)

        if not isinstance(payload, dict):
            raise ValueError(
                "El modelo de curva debe ser el payload dict guardado por "
                "train_method_1_xgb_chain.py"
            )

        required = ["model", "preprocessor", "y_scaler", "features_base"]
        for k in required:
            if k not in payload:
                raise ValueError(f"Falta la clave '{k}' en el modelo de curva.")

        self.curve_payload = payload
        self.curve_model = payload["model"]
        self.curve_preprocessor = payload["preprocessor"]
        self.curve_y_scaler = payload["y_scaler"]
        self.curve_drug_encoder = payload.get("drug_encoder", None)
        self.curve_features_base = list(payload.get("features_base", []))
        self.curve_target_names = list(
            payload.get("target_names", ["Target_A", "Target_log_b", "Target_log_k"])
        )

    def load_ee_model(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe el modelo de EE%:\n{path}")

        ensure_pickle_compatibility()
        obj = joblib.load(path)

        # Caso 1: payload dict
        if isinstance(obj, dict) and "model" in obj:
            self.ee_payload = obj
            self.ee_model = obj["model"]
            self.ee_feature_names = list(obj.get("features_base", []))
            if not self.ee_feature_names and hasattr(self.ee_model, "feature_names_in_"):
                self.ee_feature_names = list(self.ee_model.feature_names_in_)
        else:
            # Caso 2: sklearn Pipeline directo
            self.ee_payload = None
            self.ee_model = obj
            if hasattr(self.ee_model, "feature_names_in_"):
                self.ee_feature_names = list(self.ee_model.feature_names_in_)
            else:
                raise ValueError(
                    "No pude inferir las columnas del modelo de EE%. "
                    "Reentrena guardando el pipeline ya ajustado sobre DataFrame."
                )

    
    # CARGA BASES 
    # =========================================================================
    def load_reference_data(self, path):
        if not os.path.exists(path):
            print(f"Advertencia: no existe la base auxiliar:\n{path}")
            return

        try:
            df = pd.read_excel(path)
            df.columns = df.columns.astype(str).str.strip()

            # Base de drogas
            if "Drug_Name" in df.columns:
                df["Drug_Name"] = df["Drug_Name"].astype(str).str.strip()

                prop_cols = ["Drug_MW_g_mol", "Drug_LogP", "Drug_HBD", "Drug_HBA", "Drug_pKa"]
                for c in prop_cols:
                    if c in df.columns:
                        df[c] = pd.to_numeric(
                            df[c].astype(str).str.replace(",", "."),
                            errors="coerce"
                        )

                cols_present = [c for c in prop_cols if c in df.columns]
                if cols_present:
                    unique_drugs = (
                        df.groupby("Drug_Name")[cols_present]
                        .mean(numeric_only=True)
                        .reset_index()
                    )

                    for _, row in unique_drugs.iterrows():
                        name = str(row["Drug_Name"]).strip()
                        if name and name.lower() not in ["nan", "none", "unknown"]:
                            if name not in self.drug_db:
                                self.drug_db[name] = {}

                            for col in cols_present:
                                val = row[col]
                                if pd.notna(val):
                                    self.drug_db[name][col] = val

            # Categorías
            if "Crosslinker_Name" in df.columns:
                vals = sorted(df["Crosslinker_Name"].dropna().astype(str).str.strip().unique().tolist())
                vals = [v for v in vals if v.lower() not in ["nan", "none", "unknown", ""]]
                if vals:
                    self.available_crosslinkers = sorted(list(set(self.available_crosslinkers + vals)))

            if "Prep_Method" in df.columns:
                vals = sorted(df["Prep_Method"].dropna().astype(str).str.strip().unique().tolist())
                vals = [v for v in vals if v.lower() not in ["nan", "none", "unknown", ""]]
                if vals:
                    self.available_methods = sorted(list(set(self.available_methods + vals)))

            if "Geometry" in df.columns:
                vals = sorted(df["Geometry"].dropna().astype(str).str.strip().unique().tolist())
                vals = [v for v in vals if v.lower() not in ["nan", "none", "unknown", ""]]
                if vals:
                    self.available_geometries = sorted(list(set(self.available_geometries + vals)))

        except Exception as e:
            print(f"Advertencia leyendo base auxiliar {path}: {e}")

    
    # HELPERS UI
    # =========================================================================
    def fill_drug_props(self, event):
        d_name = self.cb_drug.get()
        vals = self.drug_db.get(d_name, {})
        for k, e in self.drug_entries.items():
            e.configure(state="normal")
            e.delete(0, tk.END)
            v = vals.get(k, 0)
            if pd.isna(v):
                v = 0
            e.insert(0, str(v))
            e.configure(state="readonly")

    def get_common_inputs(self):
        data = {"Drug_Name": self.cb_drug.get()}
        for k, e in self.drug_entries.items():
            data[k] = to_float(e.get(), np.nan)
        for k, e in self.polymer_entries.items():
            data[k] = to_float(e.get(), np.nan)
        return data

    def create_formulation_widgets(self, parent, exclude=None):
        widgets = {}
        exclude = set(exclude or [])

        config = {
            "Crosslinker_Name": {"type": "cat", "values": self.available_crosslinkers},
            "Prep_Method": {"type": "cat", "values": self.available_methods},
            "Geometry": {"type": "cat", "values": self.available_geometries},
            "Medium_pH": {"type": "num", "default": self.form_defaults["Medium_pH"]},
            "Temp_C": {"type": "num", "default": self.form_defaults["Temp_C"]},
            "Crosslinker_Conc_percent": {
                "type": "num",
                "default": self.form_defaults["Crosslinker_Conc_percent"]
            },
            "CS_Crosslinker_Ratio": {
                "type": "num",
                "default": self.form_defaults["CS_Crosslinker_Ratio"]
            },
            "Drug_Mass_Initial_mg": {
                "type": "num",
                "default": self.form_defaults["Drug_Mass_Initial_mg"]
            },
            "Matrix_Mass": {"type": "num", "default": self.form_defaults["Matrix_Mass"]},
        }

        for col, cfg in config.items():
            if col in exclude:
                continue

            lbl = ttk.Label(parent, font=("Arial", 8))
            self.register_field_label(lbl, col)
            lbl.pack(anchor="w")
            if cfg["type"] == "cat":
                w = ttk.Combobox(parent, values=cfg["values"], state="readonly")
                if cfg["values"]:
                    default_val = self.form_defaults.get(col, cfg["values"][0])
                    if default_val in cfg["values"]:
                        w.set(default_val)
                    else:
                        w.current(0)
            else:
                w = ttk.Entry(parent)
                w.insert(0, str(cfg.get("default", 0.0)))
            w.pack(fill="x", pady=(0, 5))
            widgets[col] = w

        return widgets

    def get_formulation_values(self, widgets_dict):
        data = {}
        for col, w in widgets_dict.items():
            if isinstance(w, ttk.Combobox):
                data[col] = w.get()
            else:
                data[col] = to_float(w.get(), np.nan)
        return data

    
    # PREDICCIÓN CURVA
    # =========================================================================
    def build_curve_feature_row(self, raw_inputs):
        row = {}

        for col in self.curve_features_base:
            if col == "pH_LogP_Diff":
                ph = to_float(raw_inputs.get("Medium_pH", np.nan), np.nan)
                logp = to_float(raw_inputs.get("Drug_LogP", np.nan), np.nan)
                row[col] = abs(ph - logp) if np.isfinite(ph) and np.isfinite(logp) else np.nan

            elif col == "Crosslinker_Density":
                ccl = to_float(raw_inputs.get("Crosslinker_Conc_percent", np.nan), np.nan)
                ccs = to_float(raw_inputs.get("CS_Conc_wv_percent", np.nan), np.nan)
                row[col] = ccl / (ccs + 0.1) if np.isfinite(ccl) and np.isfinite(ccs) else np.nan

            elif col in ["Prep_Method", "Crosslinker_Name", "Geometry"]:
                row[col] = str(raw_inputs.get(col, "Unknown"))

            else:
                row[col] = raw_inputs.get(col, np.nan)

        X_base = pd.DataFrame([row])

        if self.curve_drug_encoder is not None:
            drug_df = pd.DataFrame({"Drug_Name": [str(raw_inputs.get("Drug_Name", "Unknown"))]})
            enc_df = self.curve_drug_encoder.transform(drug_df)
            X_full = pd.concat([X_base.reset_index(drop=True), enc_df.reset_index(drop=True)], axis=1)
        else:
            X_full = X_base

        return X_full

    def predict_curve_params(self, raw_inputs):
        X_full = self.build_curve_feature_row(raw_inputs)
        Xp = self.curve_preprocessor.transform(X_full)

        Y_pred_s = self.curve_model.predict(Xp)
        Y_pred = self.curve_y_scaler.inverse_transform(Y_pred_s)

        A = float(Y_pred[0, 0])
        b = float(np.expm1(Y_pred[0, 1]))
        k = float(np.expm1(Y_pred[0, 2]))

        A = max(0.0, min(110.0, A))
        b = max(0.05, min(15.0, b))
        k = max(1e-6, min(10000.0, k))

        return A, b, k

    def predict_release_curve(self, raw_inputs, times):
        A, b, k = self.predict_curve_params(raw_inputs)
        times = np.asarray(times, dtype=float)
        y = weibull(times, A, k, b)
        y = np.clip(y, 0, 110)
        y = np.maximum.accumulate(y)
        return y, A, b, k

    
    # PREDICCIÓN EE%
    # =========================================================================
    def build_ee_input_row(self, raw_inputs, theoretical_loading=None):
        packed = {}

        for col in self.ee_feature_names:
            if col in ["Prep_Method", "Geometry", "Crosslinker_Name", "Drug_Name"]:
                packed[col] = str(raw_inputs.get(col, "Unknown"))

            elif col == "Theorical_loading":
                packed[col] = to_float(
                    theoretical_loading if theoretical_loading is not None else raw_inputs.get(col, np.nan),
                    np.nan
                )

            elif col == "Theoretical_loading":
                packed[col] = to_float(
                    theoretical_loading if theoretical_loading is not None else raw_inputs.get(col, np.nan),
                    np.nan
                )

            else:
                packed[col] = to_float(raw_inputs.get(col, np.nan), np.nan)

        return pd.DataFrame([packed])

    def predict_ee(self, raw_inputs, theoretical_loading=None):
        X = self.build_ee_input_row(raw_inputs, theoretical_loading=theoretical_loading)
        y = self.ee_model.predict(X)
        pred = float(y[0])
        pred = max(0.0, min(100.0, pred))
        return pred

    
    # TAB 1: CURVA
    # =========================================================================
    def annotate_release_points(self, ax, times, preds, tmax):
        if len(times) == 0:
            return

        marker_times = np.linspace(0.0, float(tmax), 5)
        marker_values = np.interp(marker_times, times, preds)

        ax.scatter(
            marker_times,
            marker_values,
            s=38,
            color="#c62828",
            edgecolors="white",
            linewidths=0.8,
            zorder=5
        )

        offsets = [(8, 10), (8, 10), (8, -22), (8, 10), (-62, 10)]
        for idx, (tx, ty) in enumerate(zip(marker_times, marker_values)):
            ax.annotate(
                f"{ty:.1f}%\n{tx:.1f} h",
                xy=(tx, ty),
                xytext=offsets[idx],
                textcoords="offset points",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
                zorder=6
            )

    def render_curve_plot(self, times, preds, tmax, ymax):
        self.last_curve_plot = (times, preds, tmax, ymax)
        self.render_curve_plot(times, preds, tmax, ymax)

    def setup_curve_tab(self):
        f_in = ttk.Frame(self.tab_curve)
        f_in.pack(side="left", fill="y", padx=10, pady=10)

        lbl_vars = ttk.Label(f_in, font=("Arial", 10, "bold"))
        self.register_text(lbl_vars, "formulation_vars")
        lbl_vars.pack(anchor="w", pady=10)
        self.curve_widgets = self.create_formulation_widgets(f_in, exclude={"Matrix_Mass"})

        ttk.Separator(f_in, orient="horizontal").pack(fill="x", pady=10)

        f_ax = ttk.LabelFrame(f_in, padding=5)
        self.register_text(f_ax, "graph_config")
        f_ax.pack(fill="x")

        lbl_tmax = ttk.Label(f_ax)
        self.register_text(lbl_tmax, "time_max")
        self.add_tooltip(lbl_tmax, "entry_tmax")
        lbl_tmax.pack(anchor="w")
        self.entry_tmax = ttk.Entry(f_ax)
        self.entry_tmax.insert(0, "240")
        self.entry_tmax.pack(fill="x")

        lbl_ymax = ttk.Label(f_ax)
        self.register_text(lbl_ymax, "release_max")
        self.add_tooltip(lbl_ymax, "entry_ymax")
        lbl_ymax.pack(anchor="w")
        self.entry_ymax = ttk.Entry(f_ax)
        self.entry_ymax.insert(0, "100")
        self.entry_ymax.pack(fill="x")

        btn_curve = ttk.Button(f_in, command=self.run_curve)
        self.register_text(btn_curve, "generate_curve")
        btn_curve.pack(pady=15, fill="x")

        self.lbl_curve_params = ttk.Label(
            f_in,
            text=self.t("weibull_params", A="---", b="---", k="---"),
            font=("Arial", 10)
        )
        self.lbl_curve_params.pack(anchor="w", pady=10)

        self.f_curve_plot = ttk.Frame(self.tab_curve)
        self.f_curve_plot.pack(side="right", fill="both", expand=True)

    def run_curve(self):
        tmax = to_float(self.entry_tmax.get(), 240.0)
        ymax = to_float(self.entry_ymax.get(), 100.0)

        if not np.isfinite(tmax) or tmax <= 0:
            messagebox.showwarning(self.t("warning"), self.t("tmax_warning"))
            return

        raw = self.get_common_inputs()
        raw.update(self.get_formulation_values(self.curve_widgets))
        raw.setdefault("Matrix_Mass", self.form_defaults["Matrix_Mass"])

        base_grid = np.array([0, 0.5, 1, 2, 4, 8, 12, 24, 48, 72, 96, 120, 168, 240], dtype=float)
        dense = np.linspace(0, min(tmax, 24), 36)
        tail = np.linspace(24, tmax, 40) if tmax > 24 else np.array([])
        times = np.unique(np.concatenate([dense, tail, base_grid[base_grid <= tmax]]))
        times = times[(times >= 0) & (times <= tmax)]

        try:
            preds, A, b, k = self.predict_release_curve(raw, times)
        except Exception as e:
            messagebox.showerror(self.t("error_title"), self.t("curve_error", error=e))
            return

        self.current_curve_params = (A, b, k)
        self.lbl_curve_params.config(
            text=self.t("weibull_params", A=f"{A:.4f}", b=f"{b:.4f}", k=f"{k:.4f}")
        )

        for w in self.f_curve_plot.winfo_children():
            w.destroy()

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.plot(times, preds, "o-", linewidth=2, markersize=3)
        ax.fill_between(times, preds, alpha=0.12)

        ax.set_ylim(0, ymax)
        ax.set_xlim(0, tmax)
        ax.set_xlabel(self.t("x_time"))
        ax.set_ylabel(self.t("y_release"))
        ax.grid(True, linestyle="--", alpha=0.5)
        self.annotate_release_points(ax, times, preds, tmax)

        canvas = FigureCanvasTkAgg(fig, master=self.f_curve_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    
    # TAB 2: EE%
    # =========================================================================
    def setup_ee_tab(self):
        f_in = ttk.Frame(self.tab_ee)
        f_in.pack(side="left", fill="y", padx=10, pady=10)

        lbl_vars = ttk.Label(f_in, font=("Arial", 10, "bold"))
        self.register_text(lbl_vars, "formulation_vars")
        lbl_vars.pack(anchor="w", pady=10)
        self.ee_widgets = self.create_formulation_widgets(f_in)

        ttk.Separator(f_in, orient="horizontal").pack(fill="x", pady=10)

        btn_ee = ttk.Button(f_in, command=self.run_ee)
        self.register_text(btn_ee, "calc_ee")
        btn_ee.pack(pady=20, fill="x")

        self.f_ee_res = ttk.Frame(self.tab_ee)
        self.f_ee_res.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        self.lbl_ee_res = ttk.Label(self.f_ee_res, text="--- %", font=("Arial", 40, "bold"))
        self.lbl_ee_res.pack(pady=50)

        self.lbl_ee_info = ttk.Label(
            self.f_ee_res,
            text=self.t("ee_intro"),
            font=("Arial", 12)
        )
        self.lbl_ee_info.pack()

    def run_ee(self):
        raw = self.get_common_inputs()
        raw.update(self.get_formulation_values(self.ee_widgets))

        theoretical_loading = to_float(self.form_defaults["Theorical_loading"], np.nan)
        raw["Theorical_loading"] = theoretical_loading
        raw["Theoretical_loading"] = theoretical_loading

        try:
            pred = self.predict_ee(raw, theoretical_loading=theoretical_loading)
            self.lbl_ee_res.config(text=f"{pred:.2f} %")
            self.lbl_ee_info.config(text=self.t("ee_done"))
        except Exception as e:
            messagebox.showerror(self.t("error_title"), self.t("ee_error", error=e))

    
    # TAB 3: OPTIMIZADOR
    # =========================================================================
    def setup_opt_tab(self):
        self.opt_numeric_ranges = {
            "Medium_pH": (1.0, 8.0),
            "Temp_C": (20.0, 45.0),
            "Crosslinker_Conc_percent": (0.1, 5.0),
            "CS_Crosslinker_Ratio": (1.0, 10.0),
            "Drug_Mass_Initial_mg": (1.0, 30.0),
            "Matrix_Mass": (20.0, 300.0),
        }
        self.opt_category_sources = {
            "Crosslinker_Name": self.available_crosslinkers,
            "Prep_Method": self.available_methods,
            "Geometry": self.available_geometries,
        }

        f_top = ttk.LabelFrame(self.tab_opt, padding=10)
        self.register_text(f_top, "objective")
        f_top.pack(fill="x", padx=10, pady=10)

        lbl_target = ttk.Label(f_top)
        self.register_text(lbl_target, "target_release")
        self.add_tooltip(lbl_target, "entry_target")
        lbl_target.pack(side="left", padx=(0, 4))
        self.entry_target = ttk.Entry(f_top, width=8)
        self.entry_target.insert(0, "80")
        self.entry_target.pack(side="left", padx=(0, 10))

        lbl_time = ttk.Label(f_top)
        self.register_text(lbl_time, "target_time")
        self.add_tooltip(lbl_time, "entry_ttime")
        lbl_time.pack(side="left", padx=(0, 4))
        self.entry_ttime = ttk.Entry(f_top, width=8)
        self.entry_ttime.insert(0, "24")
        self.entry_ttime.pack(side="left", padx=(0, 10))

        lbl_iterations = ttk.Label(f_top)
        self.register_text(lbl_iterations, "iterations")
        self.add_tooltip(lbl_iterations, "entry_iterations")
        lbl_iterations.pack(side="left", padx=(0, 4))
        self.entry_iterations = ttk.Entry(f_top, width=8)
        self.entry_iterations.insert(0, "4000")
        self.entry_iterations.pack(side="left")

        f_vars = ttk.LabelFrame(self.tab_opt, padding=10)
        self.register_text(f_vars, "constraints")
        f_vars.pack(fill="x", padx=10, pady=(0, 8))

        self.opt_widgets = {}
        config = {
            "Crosslinker_Name": {"type": "cat", "values": [""] + self.available_crosslinkers},
            "Prep_Method": {"type": "cat", "values": [""] + self.available_methods},
            "Geometry": {"type": "cat", "values": [""] + self.available_geometries},
            "Medium_pH": {"type": "num"},
            "Temp_C": {"type": "num"},
            "Crosslinker_Conc_percent": {"type": "num"},
            "CS_Crosslinker_Ratio": {"type": "num"},
            "Drug_Mass_Initial_mg": {"type": "num"},
            "Matrix_Mass": {"type": "num"},
        }

        r = 0
        c = 0
        for col, cfg in config.items():
            lbl = ttk.Label(f_vars)
            self.register_field_label(lbl, col)
            lbl.grid(row=r, column=c, sticky="e", padx=(0, 4))
            if cfg["type"] == "cat":
                w = ttk.Combobox(f_vars, values=cfg["values"], state="readonly", width=18)
            else:
                w = ttk.Entry(f_vars, width=18)
            w.grid(row=r, column=c + 1, padx=5, pady=2)
            self.opt_widgets[col] = w

            c += 2
            if c >= 6:
                c = 0
                r += 1

        btn_opt = ttk.Button(self.tab_opt, command=self.run_opt)
        self.register_text(btn_opt, "start_search")
        btn_opt.pack(pady=(4, 2))
        self.lbl_opt_status = ttk.Label(self.tab_opt, text=self.t("ready"), font=("Arial", 9))
        self.lbl_opt_status.pack(pady=(0, 6))

        f_results = ttk.Frame(self.tab_opt)
        f_results.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.txt_res = tk.Text(f_results, height=18, wrap="word")
        self.txt_res.pack(side="left", fill="both", expand=True, padx=(0, 8))

        self.f_opt_plot = ttk.Frame(f_results)
        self.f_opt_plot.pack(side="right", fill="both", expand=True)

    def get_opt_label(self, col):
        if col in self.opt_numeric_ranges:
            lo, hi = self.opt_numeric_ranges[col]
            return f"{self.field_label(col)} [{lo:g}-{hi:g}]"
        return self.field_label(col)

    def read_opt_constraints(self):
        constraints = {}
        for col, w in self.opt_widgets.items():
            txt = w.get().strip()
            if not txt:
                continue

            if isinstance(w, ttk.Combobox):
                constraints[col] = txt
                continue

            val = to_float(txt, np.nan)
            if not np.isfinite(val):
                raise ValueError(self.t("numeric_required", field=self.field_label(col)))

            if col in self.opt_numeric_ranges:
                lo, hi = self.opt_numeric_ranges[col]
                if val < lo or val > hi:
                    raise ValueError(self.t("range_required", field=self.field_label(col), lo=lo, hi=hi))

            constraints[col] = val

        return constraints

    def build_opt_candidate(self, base, constraints, rng, use_defaults=False):
        cand = dict(base)

        for col, values in self.opt_category_sources.items():
            clean_values = [v for v in values if str(v).strip()]
            if col in constraints:
                cand[col] = constraints[col]
            elif use_defaults and self.form_defaults.get(col) in clean_values:
                cand[col] = self.form_defaults[col]
            elif clean_values:
                cand[col] = str(rng.choice(clean_values))
            else:
                cand[col] = self.form_defaults.get(col, "Unknown")

        for col, (lo, hi) in self.opt_numeric_ranges.items():
            if col in constraints:
                cand[col] = constraints[col]
            elif use_defaults and col in self.form_defaults:
                cand[col] = self.form_defaults[col]
            else:
                cand[col] = float(rng.uniform(lo, hi))

        for col, default in self.form_defaults.items():
            cand.setdefault(col, default)

        return cand

    def render_opt_curve(self, cand, target, t_obj, y_obj):
        for w in self.f_opt_plot.winfo_children():
            w.destroy()

        plot_tmax = max(24.0, float(t_obj) * 1.5)
        times = np.linspace(0, plot_tmax, 140)
        preds, _, _, _ = self.predict_release_curve(cand, times)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
        ax.plot(times, preds, "-", linewidth=2)
        ax.fill_between(times, preds, alpha=0.12)
        ax.axhline(target, linestyle="--", linewidth=1, color="#555555")
        ax.axvline(t_obj, linestyle="--", linewidth=1, color="#555555")
        ax.scatter([t_obj], [y_obj], s=48, color="#c62828", edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(
            f"{y_obj:.1f}%\n{t_obj:.1f} h",
            xy=(t_obj, y_obj),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#555555", alpha=0.9),
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
            zorder=6
        )

        y_top = max(100.0, float(target) + 10.0, float(np.nanmax(preds)) + 5.0)
        ax.set_xlim(0, plot_tmax)
        ax.set_ylim(0, y_top)
        ax.set_xlabel(self.t("x_time"))
        ax.set_ylabel(self.t("y_release"))
        ax.grid(True, linestyle="--", alpha=0.5)

        canvas = FigureCanvasTkAgg(fig, master=self.f_opt_plot)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def format_opt_result(self, best, target, t_obj, iterations, evaluated):
        cand = best["cand"]
        res = (
            f"{self.t('best_result')}\n"
            f"{'='*70}\n"
            f"{self.t('predicted_release', y=best['y'], t=t_obj)}\n"
            f"{self.t('goal', target=target)}\n"
            f"{self.t('error', err=best['err'])}\n\n"
            f"{self.t('requested_iterations', iterations=iterations)}\n"
            f"{self.t('valid_candidates', evaluated=evaluated)}\n\n"
            f"{self.t('suggested_formulation')}\n"
        )

        ordered_cols = [
            "Crosslinker_Name", "Prep_Method", "Geometry",
            "Medium_pH", "Temp_C", "Crosslinker_Conc_percent",
            "CS_Crosslinker_Ratio", "CS_Conc_wv_percent",
            "Drug_Mass_Initial_mg", "Matrix_Mass"
        ]

        for col in ordered_cols:
            if col in cand:
                v = cand[col]
                if is_finite_number(v):
                    res += f"{self.field_label(col)}: {float(v):.4f}\n"
                else:
                    res += f"{self.field_label(col)}: {v}\n"

        return res

    def run_opt(self):
        target = to_float(self.entry_target.get(), 80.0)
        t_obj = to_float(self.entry_ttime.get(), 24.0)
        iterations_raw = to_float(self.entry_iterations.get(), 4000)

        if not np.isfinite(target) or target < 0 or target > 110 or not np.isfinite(t_obj) or t_obj <= 0:
            messagebox.showwarning(self.t("warning"), self.t("check_target_time"))
            return

        if not np.isfinite(iterations_raw):
            messagebox.showwarning(self.t("warning"), self.t("check_iterations"))
            return

        iterations = int(iterations_raw)
        if iterations < 100 or iterations > 30000:
            messagebox.showwarning(self.t("warning"), self.t("iteration_range"))
            return

        try:
            constraints = self.read_opt_constraints()
        except ValueError as e:
            messagebox.showwarning(self.t("warning"), str(e))
            return

        self.txt_res.delete(1.0, tk.END)
        self.txt_res.insert(tk.END, self.t("optimizing") + "\n")
        self.lbl_opt_status.config(text=self.t("optimizing"))
        self.root.update()

        base = self.get_common_inputs()
        rng = np.random.default_rng()
        best = None
        evaluated = 0

        for i in range(iterations):
            cand = self.build_opt_candidate(base, constraints, rng, use_defaults=(i == 0))

            try:
                y, A, b, k = self.predict_release_curve(cand, [t_obj])
                y = float(y[0])
                err = abs(y - target)
                evaluated += 1

                if best is None or err < best["err"]:
                    best = {
                        "err": err,
                        "y": y,
                        "A": A,
                        "b": b,
                        "k": k,
                        "cand": cand
                    }
            except Exception:
                continue

            if (i + 1) % 500 == 0 and best is not None:
                self.lbl_opt_status.config(
                    text=self.t("evaluated", done=i + 1, total=iterations, err=best["err"])
                )
                self.root.update()

        if best is None:
            self.lbl_opt_status.config(text=self.t("invalid_candidates"))
            self.txt_res.insert(tk.END, self.t("no_candidates") + "\n")
            return

        cand = best["cand"]
        res = self.format_opt_result(best, target, t_obj, iterations, evaluated)
        self.last_opt_result = (best, target, t_obj, iterations, evaluated)
        self.txt_res.delete(1.0, tk.END)
        self.txt_res.insert(tk.END, res)
        self.lbl_opt_status.config(text=self.t("finished", err=best["err"]))
        self.render_opt_curve(cand, target, t_obj, best["y"])

# MAIN
# =========================================================================
if __name__ == "__main__":
    root = tk.Tk()

    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    style = ttk.Style()
    try:
        style.theme_use("clam")
    except:
        pass

    app = HydrogelApp(root)
    root.mainloop()
