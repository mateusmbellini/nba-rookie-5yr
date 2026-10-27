"""Verifica a inferência fora da interface (compare com o resultado do app publicado)."""
import joblib
from common import to_frame, validate_inputs

entrada = dict(gp=80, min=32.0, pts=15.0, fgm=5.6, fga=12.0, tpm=1.2, tpa=3.4,
               ftm=2.6, fta=3.3, oreb=1.2, dreb=4.0, ast=3.5, stl=1.0, blk=0.5, tov=2.3)
erros, avisos = validate_inputs(entrada)
assert not erros, erros
b = joblib.load("models/model.joblib")
p = b["pipeline"].predict_proba(to_frame(entrada))[0, 1]
print(f"Probabilidade de >= 5 anos: {p:.4f} | limiar {b['threshold']:.2f} | classe {int(p >= b['threshold'])}")
