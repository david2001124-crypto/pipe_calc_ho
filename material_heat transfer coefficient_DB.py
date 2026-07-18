# material_db.py
# Thermal Conductivity Data Source: ASME BPVC Section II Part D & NIST
# Absolute Roughness Source: Engineering ToolBox

MATERIAL_DB = {
    "Carbon Steel": {
        "Average (Default)": {
            "roughness_m": 0.000056, # 0.000045 ~ 0.0000675 average
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [47.5, 51.0, 51.5, 50.5, 47.7, 44.2, 40.8, 37.5] # Mean of typical CS grades
            }
        },
        "SA-106 Grade B (Seamless)": {
            "roughness_m": 0.000045,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [48.0, 51.5, 51.9, 51.1, 48.6, 45.3, 41.9, 38.6]
            }
        },
        "SA-53 Grade B (Welded)": {
            "roughness_m": 0.0000675,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [47.0, 50.5, 51.1, 49.9, 46.8, 43.1, 39.7, 36.4]
            }
        }
    },
    "Stainless Steel": {
        "Average (Default)": {
            "roughness_m": 0.0000035,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [11.0, 13.9, 14.5, 15.8, 17.4, 18.9, 20.4, 21.7] # Mean of 304 and 316
            }
        },
        "AISI 304 (Austenitic)": {
            "roughness_m": 0.0000035,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [11.2, 14.3, 14.8, 16.3, 18.0, 19.6, 21.2, 22.5]
            }
        },
        "AISI 316 (Austenitic)": {
            "roughness_m": 0.0000035,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [10.8, 13.5, 14.2, 15.3, 16.8, 18.2, 19.6, 20.9]
            }
        }
    },
    "Copper-Nickel": {
        "Average (Default)": {
            "roughness_m": 0.0000015,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [37.5, 42.5, 45.0, 50.0, 54.5, 59.0, 63.5, 68.0]
            }
        },
        "90/10 Cu-Ni": {
            "roughness_m": 0.0000015,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [40.0, 45.0, 50.0, 54.0, 58.0, 63.0, 68.0, 72.0]
            }
        },
        "70/30 Cu-Ni": {
            "roughness_m": 0.0000015,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [35.0, 40.0, 40.0, 46.0, 51.0, 55.0, 59.0, 64.0]
            }
        }
    },
    "GRE / Plastic": {
        "Average (Default)": {
            "roughness_m": 0.00000425,
            "k_table": {
                "T_C": [-50, 0, 20, 50, 100],
                "k_W_mK": [0.28, 0.30, 0.32, 0.34, 0.36]
            }
        }
    },
    "Galvanized Steel": {
        "Average (Default)": {
            "roughness_m": 0.00015,
            "k_table": {
                "T_C": [-100, 0, 20, 100, 200, 300, 400, 500],
                "k_W_mK": [48.0, 51.5, 51.9, 51.1, 48.6, 45.3, 41.9, 38.6] # Follows CS base
            }
        }
    }
}