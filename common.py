"""Funções compartilhadas entre o treinamento (train.py) e a aplicação (app.py).

Manter o pré-processamento em UM único lugar garante que a aplicação trate
as entradas exatamente como o treinamento tratou (requisito do enunciado).
"""
import re
import numpy as np
import pandas as pd

# Colunas "brutas" usadas como entrada do modelo (estatísticas médias por jogo
# da temporada de calouro). Os percentuais NÃO são entradas: são recalculados
# a partir dos arremessos feitos/tentados (ver add_derived_features).
RAW_FEATURES = [
    "gp", "min", "pts",
    "fgm", "fga",
    "tpm", "tpa",
    "ftm", "fta",
    "oreb", "dreb",
    "ast", "stl", "blk", "tov",
]
DERIVED_FEATURES = ["fg_pct", "tp_pct", "ft_pct"]
FEATURES = RAW_FEATURES + DERIVED_FEATURES
TARGET = "target"

# Descrição amigável para a interface (rótulo, unidade, mínimo, máximo, padrão)
FIELD_INFO = {
    "gp":   ("Jogos disputados", "jogos (0-82)", 0, 82, 60),
    "min":  ("Minutos por jogo", "min/jogo (0-48)", 0.0, 48.0, 18.0),
    "pts":  ("Pontos por jogo", "pts/jogo", 0.0, 40.0, 7.0),
    "fgm":  ("Arremessos de quadra convertidos", "por jogo", 0.0, 20.0, 2.8),
    "fga":  ("Arremessos de quadra tentados", "por jogo", 0.0, 40.0, 6.3),
    "tpm":  ("Bolas de 3 convertidas", "por jogo", 0.0, 6.0, 0.3),
    "tpa":  ("Bolas de 3 tentadas", "por jogo", 0.0, 15.0, 1.0),
    "ftm":  ("Lances livres convertidos", "por jogo", 0.0, 12.0, 1.1),
    "fta":  ("Lances livres tentados", "por jogo", 0.0, 15.0, 1.5),
    "oreb": ("Rebotes ofensivos", "por jogo", 0.0, 6.0, 1.0),
    "dreb": ("Rebotes defensivos", "por jogo", 0.0, 12.0, 2.0),
    "ast":  ("Assistências", "por jogo", 0.0, 12.0, 1.6),
    "stl":  ("Roubos de bola", "por jogo", 0.0, 4.0, 0.6),
    "blk":  ("Bloqueios", "por jogo", 0.0, 4.0, 0.4),
    "tov":  ("Erros / turnovers", "por jogo", 0.0, 6.0, 1.2),
}

# Mapeia nomes de colunas das diferentes versões do dataset para nomes canônicos
_ALIASES = {
    "gp": "gp", "gamesplayed": "gp", "games_played": "gp",
    "min": "min", "minutes": "min", "minutesplayed": "min",
    "pts": "pts", "pointspergame": "pts", "points_per_game": "pts",
    "fgm": "fgm", "fieldgoalsmade": "fgm",
    "fga": "fga", "fieldgoalsattempt": "fga",
    "fg": "fg_pct", "fg_pct": "fg_pct", "fieldgoalpercent": "fg_pct",
    "3p_made": "tpm", "3pmade": "tpm", "3pointmade": "tpm",
    "3pa": "tpa", "3pointattempt": "tpa",
    "3p": "tp_pct", "3p_pct": "tp_pct", "3pointpercent": "tp_pct",
    "ftm": "ftm", "freethrowmade": "ftm",
    "fta": "fta", "freethrowattempt": "fta",
    "ft": "ft_pct", "ft_pct": "ft_pct", "freethrowpercent": "ft_pct",
    "oreb": "oreb", "offensiverebounds": "oreb",
    "dreb": "dreb", "defensiverebounds": "dreb",
    "reb": "reb", "rebounds": "reb",
    "ast": "ast", "assists": "ast",
    "stl": "stl", "steals": "stl",
    "blk": "blk", "blocks": "blk",
    "tov": "tov", "turnovers": "tov",
    "target_5yrs": TARGET, "target": TARGET, "target5yrs": TARGET,
    "name": "name",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os nomes das colunas (funciona para as versões Kaggle/data.world)."""
    def clean(c):
        c = str(c).strip().lower().replace("%", "_pct")
        c = re.sub(r"[\s\-]+", "_", c)
        c = c.replace("_pct_pct", "_pct")
        return c
    cleaned = [clean(c) for c in df.columns]
    # versões onde "fg%" virou "fg_pct" ou "fg" são tratadas pelo dicionário
    new_cols = [_ALIASES.get(c, _ALIASES.get(c.replace("_", ""), c)) for c in cleaned]
    out = df.copy()
    out.columns = new_cols
    return out


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcula os percentuais (%) a partir de feitos/tentados.

    Quando não há tentativas (ex.: jogador que nunca arremessou de 3),
    o percentual é definido como 0. Isso evita NaN e mantém o treinamento
    e a aplicação consistentes.
    """
    out = df.copy()
    for pct, made, att in [("fg_pct", "fgm", "fga"),
                           ("tp_pct", "tpm", "tpa"),
                           ("ft_pct", "ftm", "fta")]:
        att_vals = out[att].astype(float)
        out[pct] = np.where(att_vals > 0, out[made].astype(float) / att_vals.where(att_vals > 0, 1) * 100.0, 0.0)
    return out


def validate_inputs(values: dict):
    """Valida os dados digitados pelo usuário.

    Retorna (erros, avisos). 'erros' impedem a previsão; 'avisos' apenas alertam.
    """
    errors, warnings = [], []
    for k in RAW_FEATURES:
        v = values.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            errors.append(f"O campo '{FIELD_INFO[k][0]}' não foi preenchido.")
            continue
        _, _, lo, hi, _ = FIELD_INFO[k]
        if v < lo or v > hi:
            errors.append(f"'{FIELD_INFO[k][0]}' deve estar entre {lo} e {hi} (recebido: {v}).")
    if errors:
        return errors, warnings

    if values["fgm"] > values["fga"]:
        errors.append("Arremessos convertidos não podem ser maiores que os tentados (quadra).")
    if values["tpm"] > values["tpa"]:
        errors.append("Bolas de 3 convertidas não podem ser maiores que as tentadas.")
    if values["ftm"] > values["fta"]:
        errors.append("Lances livres convertidos não podem ser maiores que os tentados.")
    if values["tpm"] > values["fgm"]:
        errors.append("Bolas de 3 convertidas não podem exceder o total de arremessos de quadra convertidos.")
    if values["tpa"] > values["fga"]:
        errors.append("Bolas de 3 tentadas não podem exceder o total de arremessos de quadra tentados.")

    # Avisos de consistência (não bloqueiam)
    pts_calc = 2 * values["fgm"] + values["tpm"] + values["ftm"]
    if abs(pts_calc - values["pts"]) > 1.5:
        warnings.append(
            f"Os pontos informados ({values['pts']:.1f}) diferem do esperado pelos arremessos "
            f"({pts_calc:.1f} = 2×FGM + 3PM + FTM). Confira os valores."
        )
    if values["gp"] == 0 and values["min"] > 0:
        warnings.append("Jogos = 0 com minutos > 0 é incoerente.")
    if values["gp"] < 10:
        warnings.append("Poucos jogos na temporada: as médias são pouco confiáveis e o modelo tende a errar mais.")
    return errors, warnings


def to_frame(values: dict) -> pd.DataFrame:
    """Converte o dicionário de entradas em DataFrame com as colunas do modelo."""
    df = pd.DataFrame([{k: float(values[k]) for k in RAW_FEATURES}])
    df = add_derived_features(df)
    return df[FEATURES]
