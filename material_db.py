import numpy as np

# ---------------------------------------------------------
# RAW DATABASE (ASME BPVC Section II Part D - Table TCD)
# ---------------------------------------------------------
RAW_DB = {
    "Carbon and Low Alloy Steels": {
        "Material Group A [Note (1)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750],
            "k_W_mK": [60.4, 59.8, 58.9, 58.0, 57.0, 55.9, 54.7, 53.6, 52.5, 51.4, 50.3, 49.2, 48.1, 47.0, 45.9, 44.9, 43.8, 42.7, 41.6, 40.5, 39.3, 38.2, 37.0, 35.8, 34.7, 33.5, 32.3, 31.2, 30.1, 29.1]
        },
        "Material Group B [Note (2)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800],
            "k_W_mK": [47.3, 47.9, 48.0, 47.9, 47.6, 47.2, 46.7, 46.1, 45.5, 44.8, 44.2, 43.5, 42.9, 42.2, 41.5, 40.9, 40.2, 39.4, 38.6, 37.8, 36.9, 36.0, 35.0, 34.0, 33.0, 31.9, 30.8, 29.8, 28.8, 27.8, 26.9, 26.1]
        },
        "Material Group C [Note (3)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750],
            "k_W_mK": [41.0, 40.8, 40.7, 40.6, 40.5, 40.4, 40.3, 40.1, 39.8, 39.5, 39.1, 38.7, 38.3, 37.8, 37.3, 36.8, 36.3, 35.8, 35.3, 34.8, 34.4, 33.9, 33.4, 32.8, 32.2, 31.6, 30.7, 29.1, 27.6, 26.7]
        },
        "Material Group D [Note (4)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800],
            "k_W_mK": [36.3, 36.5, 36.7, 36.9, 37.0, 37.1, 37.2, 37.2, 37.2, 37.1, 36.9, 36.7, 36.5, 36.2, 35.8, 35.4, 35.0, 34.6, 34.2, 33.7, 33.3, 32.8, 32.4, 32.0, 31.5, 31.1, 30.6, 30.1, 28.7, 27.4, 26.8, 26.7]
        }
    },
    "High Alloy Steels": {
        "Material Group J [Note (10)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750],
            "k_W_mK": [14.8, 15.3, 15.8, 16.2, 16.6, 17.0, 17.5, 17.9, 18.3, 18.6, 19.0, 19.4, 19.8, 20.1, 20.5, 20.8, 21.2, 21.5, 21.9, 22.2, 22.6, 22.9, 23.3, 23.6, 24.0, 24.3, 24.7, 25.0, 25.4, 25.7]
        },
        "Material Group K [Note (11)]": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750],
            "k_W_mK": [14.1, 14.6, 15.0, 15.4, 15.7, 16.1, 16.5, 16.8, 17.2, 17.6, 17.9, 18.3, 18.7, 19.0, 19.4, 19.7, 20.1, 20.5, 20.8, 21.2, 21.5, 21.9, 22.2, 22.6, 22.9, 23.2, 23.6, 23.9, 24.2, 24.6]
        }
    },
    "Aluminum Alloys": {
        "A95083": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200],
            "k_W_mK": [116.1, 120.6, 123.8, 126.7, 129.5, 132.1, 134.5, 136.7]
        }
    },
    "Titanium Alloys": {
        "Titanium Gr. 1, 2, 2H, 3, 7, 7H, 11, 12, 16, 16H, 17, 26, 26H, and 27": {
            "T_C": [20, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650, 675, 700, 725, 750, 775, 800, 825, 850, 875, 900],
            "k_W_mK": [22.0, 21.4, 21.1, 20.7, 20.5, 20.2, 20.0, 19.9, 19.7, 19.6, 19.5, 19.4, 19.4, 19.3, 19.3, 19.3, 19.4, 19.4, 19.5, 19.6, 19.7, 19.8, 19.9, 20.4, 20.9, 21.4, 21.9, 22.4, 22.9, 23.3, 23.8, 24.3, 24.8, 25.2, 25.6, 25.9]
        }
    }
}

# ---------------------------------------------------------
# MATERIAL MAPPING (ASME Standard & Engineering ToolBox)
# ---------------------------------------------------------
MATERIAL_MAP = {
    "탄소강 / ASME Material Group A": {
        "roughness_m": 0.000045,
        "asme_category": "Carbon and Low Alloy Steels",
        "asme_grade": "Material Group A [Note (1)]",
        "desc": "일반 상업용 탄소강"
    },
    "저온 탄소강 / ASME Material Group B": {
        "roughness_m": 0.000045,
        "asme_category": "Carbon and Low Alloy Steels",
        "asme_grade": "Material Group B [Note (2)]",
        "desc": "LPG 및 한랭지 운항용"
    },
    "스테인리스강 (SUS 316L) / ASME Material Group K": {
        "roughness_m": 0.0000035,
        "asme_category": "High Alloy Steels",
        "asme_grade": "Material Group K [Note (11)]",
        "desc": "LNG 및 암모니아 라인용"
    },
    "스테인리스강 (SUS 304) / ASME Material Group J": {
        "roughness_m": 0.0000035,
        "asme_category": "High Alloy Steels",
        "asme_grade": "Material Group J [Note (10)]",
        "desc": "일반 부식 방지용"
    },
    "듀플렉스강 (Duplex SS) / ASME Material Group K": {
        "roughness_m": 0.0000035,
        "asme_category": "High Alloy Steels",
        "asme_grade": "Material Group K [Note (11)]",
        "desc": "스크러버 및 화학 제품창"
    },
    "9% 니켈강 (9% Ni) / ASME Material Group D": {
        "roughness_m": 0.000045,
        "asme_category": "Carbon and Low Alloy Steels",
        "asme_grade": "Material Group D [Note (4)]",
        "desc": "LNG 독립형 탱크 부재"
    },
    "알루미늄 합금 (Al 5083) / ASME A95083": {
        "roughness_m": 0.0000015,
        "asme_category": "Aluminum Alloys",
        "asme_grade": "A95083",
        "desc": "모스형 LNG 화물창 부재"
    },
    "티타늄 (Titanium Gr. 1/2) / ASME Titanium Gr. 1~27": {
        "roughness_m": 0.0000015,
        "asme_category": "Titanium Alloys",
        "asme_grade": "Titanium Gr. 1, 2, 2H, 3, 7, 7H, 11, 12, 16, 16H, 17, 26, 26H, and 27",
        "desc": "해수 냉각 배관 및 열교환기 플레이트"
    }
}

# ---------------------------------------------------------
# FITTING & VALVE DATABASE (HYSYS / Crane TP-410 & Chisholm B-parameter)
# ---------------------------------------------------------
# HYSYS 기반 K-Factor 연산: K = A(Velocity Head Factor) + B(FT factor) * f_T
# Chisholm B: 2상 유동(Two-phase) 압력 강하 보정을 위한 파라미터
HYSYS_FITTING_DB = {
    "180 Degree Close Return": {"A": 0.0, "B": 50, "Chisholm_B": 2.2},
    "Angle Valve, 45 deg: Open": {"A": 0.0, "B": 55, "Chisholm_B": 1.5},
    "Angle Valve, 90 deg: Open": {"A": 0.0, "B": 150, "Chisholm_B": 1.5},
    "Angle Valve: Open": {"A": 2.0, "B": 0, "Chisholm_B": 1.5},
    "Ball Valve: Open": {"A": 0.0, "B": 3, "Chisholm_B": 1.5},
    "Bend: 90, r/d 1": {"A": 0.0, "B": 20, "Chisholm_B": 2.2},
    "Bend: 90, r/d 1.5": {"A": 0.0, "B": 14, "Chisholm_B": 2.2},
    "Bend: 90, r/d 10": {"A": 0.0, "B": 30, "Chisholm_B": 2.2},
    "Bend: 90, r/d 12": {"A": 0.0, "B": 34, "Chisholm_B": 2.2},
    "Bend: 90, r/d 14": {"A": 0.0, "B": 38, "Chisholm_B": 2.2},
    "Bend: 90, r/d 16": {"A": 0.0, "B": 42, "Chisholm_B": 2.2},
    "Bend: 90, r/d 2": {"A": 0.0, "B": 12, "Chisholm_B": 2.2},
    "Bend: 90, r/d 20": {"A": 0.0, "B": 50, "Chisholm_B": 2.2},
    "Bend: 90, r/d 3": {"A": 0.0, "B": 12, "Chisholm_B": 2.2},
    "Bend: 90, r/d 4": {"A": 0.0, "B": 14, "Chisholm_B": 2.2},
    "Bend: 90, r/d 6": {"A": 0.0, "B": 17, "Chisholm_B": 2.2},
    "Bend: 90, r/d 8": {"A": 0.0, "B": 24, "Chisholm_B": 2.2},
    "Blowoff Valve: Open": {"A": 3.0, "B": 0, "Chisholm_B": 1.5},
    "Butterfly Valve: 10-14in, Open": {"A": 0.0, "B": 35, "Chisholm_B": 1.5},
    "Butterfly Valve: 16-24in, Open": {"A": 0.0, "B": 25, "Chisholm_B": 1.5},
    "Butterfly Valve: 2-8in, Open": {"A": 0.0, "B": 45, "Chisholm_B": 1.5},
    "Butterfly Valve: Angle 10": {"A": 0.52, "B": 0, "Chisholm_B": 1.5},
    "Butterfly Valve: Angle 20": {"A": 1.54, "B": 0, "Chisholm_B": 1.5},
    "Butterfly Valve: Angle 40": {"A": 10.8, "B": 0, "Chisholm_B": 1.5},
    "Butterfly Valve: Angle 5": {"A": 0.24, "B": 0, "Chisholm_B": 1.5},
    "Butterfly Valve: Angle 60": {"A": 118.0, "B": 0, "Chisholm_B": 1.5},
    "Check Valve: 45 deg Lift": {"A": 0.0, "B": 55, "Chisholm_B": 1.5},
    "Check Valve: Ball": {"A": 70.0, "B": 0, "Chisholm_B": 1.5},
    "Check Valve: Disk": {"A": 10.0, "B": 0, "Chisholm_B": 1.5},
    "Check Valve: Lift": {"A": 0.0, "B": 600, "Chisholm_B": 1.5},
    "Check Valve: Swing": {"A": 2.0, "B": 0, "Chisholm_B": 1.5},
    "Coupling/Union": {"A": 0.04, "B": 0, "Chisholm_B": 1.5},
    "Diaphram Valve: Half": {"A": 4.3, "B": 0, "Chisholm_B": 1.5},
    "Diaphram Valve: One Quarter": {"A": 21.0, "B": 0, "Chisholm_B": 1.5},
    "Diaphram Valve: Open": {"A": 2.3, "B": 0, "Chisholm_B": 1.5},
    "Diaphram Valve: Three Quarter": {"A": 2.6, "B": 0, "Chisholm_B": 1.5},
    "Elbow: 45 Long": {"A": 0.2, "B": 0, "Chisholm_B": 2.0},
    "Elbow: 45 Mitre": {"A": 0.0, "B": 15, "Chisholm_B": 2.0},
    "Elbow: 45 Std": {"A": 0.0, "B": 16, "Chisholm_B": 2.0},
    "Elbow: 90 Long": {"A": 0.45, "B": 0, "Chisholm_B": 2.2},
    "Elbow: 90 Mitre": {"A": 0.0, "B": 60, "Chisholm_B": 2.2},
    "Elbow: 90 Std": {"A": 0.0, "B": 30, "Chisholm_B": 2.2},
    "Foot Valve": {"A": 15.0, "B": 0, "Chisholm_B": 1.5},
    "Foot Valve: Hinged disk": {"A": 0.0, "B": 75, "Chisholm_B": 1.5},
    "Foot Valve: Poppet disk": {"A": 0.0, "B": 420, "Chisholm_B": 1.5},
    "Gate Valve, Crane: Open": {"A": 0.0, "B": 8, "Chisholm_B": 1.5},
    "Gate Valve: Half": {"A": 4.5, "B": 0, "Chisholm_B": 1.5},
    "Gate Valve: One Quarter": {"A": 24.0, "B": 0, "Chisholm_B": 1.5},
    "Gate Valve: Open": {"A": 0.17, "B": 0, "Chisholm_B": 1.5},
    "Gate Valve: Three Quarter": {"A": 0.9, "B": 0, "Chisholm_B": 1.5},
    "Globe Valve, Crane: Open": {"A": 0.0, "B": 340, "Chisholm_B": 1.5},
    "Globe Valve: Half": {"A": 9.5, "B": 0, "Chisholm_B": 1.5},
    "Globe Valve: Open": {"A": 6.0, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Angle 10": {"A": 0.29, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Angle 20": {"A": 1.56, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Angle 40": {"A": 17.3, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Angle 5": {"A": 0.05, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Angle 60": {"A": 206.0, "B": 0, "Chisholm_B": 1.5},
    "Plug Cock: Open": {"A": 0.0, "B": 18, "Chisholm_B": 1.5},
    "Tee: As Elbow": {"A": 0.0, "B": 60, "Chisholm_B": 1.8},
    "Tee: Branch Blanked": {"A": 0.0, "B": 20, "Chisholm_B": 1.5},
    "Water Meter: Disk": {"A": 7.0, "B": 0, "Chisholm_B": 1.5},
    "Water Meter: Piston": {"A": 15.0, "B": 0, "Chisholm_B": 1.5},
    "Water Meter: Rotary": {"A": 10.0, "B": 0, "Chisholm_B": 1.5},
    "Water Meter: Turbine": {"A": 6.0, "B": 0, "Chisholm_B": 1.5}
}
