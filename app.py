"""Aplicação Streamlit - Previsão de longevidade de calouros da NBA (>= 5 anos)."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from common import FEATURES, RAW_FEATURES, FIELD_INFO, validate_inputs, to_frame

st.set_page_config(page_title="NBA Rookie: 5 anos na liga?", page_icon="🏀", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load("models/model.joblib")


bundle = load_bundle()
pipe, threshold = bundle["pipeline"], bundle["threshold"]

st.title("🏀 Um calouro da NBA ainda estará na liga daqui a 5 anos?")
st.markdown(
    """
**Finalidade:** estimar a probabilidade de um jogador que acabou de terminar sua
temporada de calouro continuar na NBA por pelo menos 5 anos, a partir das suas
**médias por jogo** nessa temporada.
**Público:** estudantes, analistas e entusiastas de basquete. É uma ferramenta
**educacional**, não um instrumento de decisão profissional.
"""
)

with st.expander("Como usar e o que significa cada campo"):
    st.markdown("Todos os valores (exceto *Jogos*) são **médias por jogo** da temporada de calouro.")
    st.table(pd.DataFrame(
        [(FIELD_INFO[k][0], FIELD_INFO[k][1], f"{FIELD_INFO[k][2]} a {FIELD_INFO[k][3]}") for k in RAW_FEATURES],
        columns=["Campo", "Unidade", "Faixa aceita"]))
    st.caption("Os percentuais de arremesso são calculados automaticamente (convertidos ÷ tentados), "
               "exatamente como no treinamento.")

# ---- Exemplos prontos (úteis para testar rapidamente) ----
EXAMPLES = {
    "Personalizado (padrão)": {k: FIELD_INFO[k][4] for k in RAW_FEATURES},
    "Titular com muitos minutos": dict(gp=80, min=32.0, pts=15.0, fgm=5.6, fga=12.0, tpm=1.2, tpa=3.4,
                                       ftm=2.6, fta=3.3, oreb=1.2, dreb=4.0, ast=3.5, stl=1.0, blk=0.5, tov=2.3),
    "Reserva com poucos minutos": dict(gp=25, min=6.0, pts=2.0, fgm=0.8, fga=2.0, tpm=0.0, tpa=0.2,
                                       ftm=0.4, fta=0.7, oreb=0.4, dreb=0.8, ast=0.3, stl=0.2, blk=0.1, tov=0.4),
}
choice = st.selectbox("Carregar um exemplo", list(EXAMPLES))
ex = EXAMPLES[choice]

with st.form("entrada"):
    cols = st.columns(3)
    values = {}
    for i, k in enumerate(RAW_FEATURES):
        label, unit, lo, hi, _ = FIELD_INFO[k]
        is_int = k == "gp"
        with cols[i % 3]:
            values[k] = st.number_input(
                f"{label} ({unit})", min_value=None, max_value=None,  # validação feita por validate_inputs
                value=int(ex[k]) if is_int else float(ex[k]),
                step=1 if is_int else 0.1, format="%d" if is_int else "%.1f", key=f"{choice}_{k}")
    enviado = st.form_submit_button("Prever")

if enviado:
    errors, warnings = validate_inputs(values)
    if errors:
        st.error("Não foi possível calcular a previsão:")
        for e in errors:
            st.write(f"- {e}")
    else:
        for w in warnings:
            st.warning(w)
        X = to_frame(values)
        proba = float(pipe.predict_proba(X)[0, 1])  # apenas inferência, sem retreinar
        positive = proba >= threshold

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Probabilidade de durar ≥ 5 anos", f"{proba:.1%}")
            st.caption(f"Limiar de decisão do modelo: {threshold:.2f}")
        with c2:
            if positive:
                st.success("**Previsão: provável permanência de 5 anos ou mais na NBA.**")
            else:
                st.info("**Previsão: carreira provavelmente mais curta que 5 anos.**")
            margem = abs(proba - threshold)
            if margem < 0.10:
                st.warning("Caso **limítrofe**: a probabilidade está próxima do limiar, "
                           "então a previsão é pouco confiável.")

        # Quais variáveis mais pesaram (só para regressão logística)
        if bundle["model_name"] == "Regressão Logística":
            sc = pipe.named_steps["sc"]; clf = pipe.named_steps["clf"]
            z = sc.transform(pipe.named_steps["imp"].transform(X))[0]
            contrib = pd.Series(z * clf.coef_[0], index=FEATURES).sort_values(key=np.abs, ascending=False).head(5)
            st.subheader("O que mais influenciou esta previsão")
            nomes = {**{k: FIELD_INFO[k][0] for k in RAW_FEATURES},
                     "fg_pct": "% arremessos de quadra", "tp_pct": "% bolas de 3", "ft_pct": "% lances livres"}
            for f, v in contrib.items():
                seta = "⬆️ aumenta" if v > 0 else "⬇️ reduz"
                st.write(f"- **{nomes[f]}**: {seta} a chance (contribuição {v:+.2f})")
        else:
            imp = pd.Series(bundle["permutation_importance"]).sort_values(ascending=False).head(5)
            st.subheader("Variáveis mais importantes no modelo (global)")
            st.write(", ".join(imp.index))

        # Avisa se a entrada está fora do intervalo visto no treino
        fora = [FIELD_INFO[k][0] for k in RAW_FEATURES
                if not (bundle["train_ranges"][k][0] <= values[k] <= bundle["train_ranges"][k][1])]
        if fora:
            st.warning("Valores fora do intervalo observado no treinamento (extrapolação): " + ", ".join(fora))

st.divider()
st.subheader("Limitações")
st.markdown(
    """
- O modelo usa **apenas estatísticas da temporada de calouro**. Lesões, posição de draft, idade, contrato,
  mudanças de regras e acaso **não** estão nos dados.
- Os dados são históricos (até cerca de 2016) e o estilo de jogo mudou; a previsão pode ser menos precisa para jogadores atuais.
- "Permanecer 5 anos" é um critério simplificado de "carreira de sucesso".
- O resultado é uma **probabilidade estimada**, não uma garantia. Não use para decisões reais de contratação.
"""
)
with st.expander("Sobre o modelo"):
    st.write(f"Modelo: **{bundle['model_name']}** | Limiar: {threshold:.2f}")
    st.json(bundle["test_metrics"])
