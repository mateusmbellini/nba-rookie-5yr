"""Treinamento e avaliação - 5-Year NBA Rookie Classification.

Uso:
    python train.py --data data/nba_rookies.csv

Etapas: leitura -> análise exploratória -> limpeza -> split treino/teste ->
comparação de modelos (validação cruzada SOMENTE no treino) -> ajuste do limiar
(SOMENTE no treino) -> avaliação final no teste -> salvamento do modelo.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (confusion_matrix, classification_report, ConfusionMatrixDisplay,
                             roc_auc_score, roc_curve, f1_score, accuracy_score)
from sklearn.inspection import permutation_importance

from common import (normalize_columns, add_derived_features, FEATURES, RAW_FEATURES, TARGET)

SEED = 42
FIG = "reports/figures"


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=150)
    plt.close()


def main(path):
    os.makedirs(FIG, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    log = []
    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s); log.append(s)

    # ------------------------------------------------------------------ 1. Dados
    raw = pd.read_csv(path)
    df = normalize_columns(raw)
    say("== DADOS ==")
    say("Formato original:", raw.shape)
    say("Colunas após padronização:", list(df.columns))

    missing_cols = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Colunas ausentes no CSV: {missing_cols}. Ajuste o dicionário _ALIASES em common.py.")

    say("\nValores ausentes por coluna (original):")
    say(raw.isna().sum()[raw.isna().sum() > 0].to_string() or "nenhum")

    # Duplicatas: considera todas as colunas de estatística + alvo (o nome pode repetir legitimamente)
    key_cols = [c for c in df.columns if c != "name"] + (["name"] if "name" in df.columns else [])
    n_dup = df.duplicated(subset=key_cols).sum()
    say(f"\nLinhas duplicadas (todas as colunas iguais): {n_dup}")
    df = df.drop_duplicates(subset=key_cols).reset_index(drop=True)

    # Alvo
    df[TARGET] = df[TARGET].astype(int)
    say("\nDistribuição do alvo (1 = durou >= 5 anos):")
    say(df[TARGET].value_counts().to_string())
    say((df[TARGET].value_counts(normalize=True) * 100).round(1).to_string())

    # Recalcula percentuais a partir de feitos/tentados (mesma função usada no app)
    df = add_derived_features(df)
    say("\nEstatísticas descritivas:")
    say(df[FEATURES].describe().T.round(2).to_string())

    # Valores implausíveis (ex.: feitos > tentados)
    bad = ((df["fgm"] > df["fga"]) | (df["tpm"] > df["tpa"]) | (df["ftm"] > df["fta"])).sum()
    say(f"\nLinhas com feitos > tentados: {bad}")
    df = df[~((df["fgm"] > df["fga"]) | (df["tpm"] > df["tpa"]) | (df["ftm"] > df["fta"]))]

    # ---------------------------------------------------------- 2. Gráficos de EDA
    df[TARGET].value_counts().sort_index().plot(kind="bar", color=["#c0504d", "#4f81bd"])
    plt.xticks([0, 1], ["< 5 anos (0)", ">= 5 anos (1)"], rotation=0); plt.ylabel("Jogadores")
    plt.title("Distribuição da variável-alvo"); savefig("eda_target.png")

    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns)
    fig.colorbar(im); ax.set_title("Correlação entre variáveis"); savefig("eda_corr.png")

    df.groupby(TARGET)[["gp", "min", "pts", "oreb", "ast", "tov"]].mean().T.plot(kind="bar")
    plt.title("Média por classe (alvo)"); plt.xticks(rotation=0); savefig("eda_means_by_class.png")
    say("\nMédias por classe:")
    say(df.groupby(TARGET)[FEATURES].mean().T.round(2).to_string())

    # --------------------------------------------------------- 3. Split treino/teste
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    say(f"\nTreino: {X_tr.shape}, Teste: {X_te.shape} (estratificado, seed={SEED})")

    # --------------------------------------------------- 4. Modelos (CV só no treino)
    # Imputer e scaler dentro do Pipeline => ajustados apenas no treino de cada fold.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    pipe_lr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=SEED)),
    ])
    grid_lr = {"clf__C": [0.01, 0.1, 1, 10], "clf__class_weight": [None, "balanced"]}

    pipe_dt = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", DecisionTreeClassifier(random_state=SEED)),
    ])
    grid_dt = {"clf__max_depth": [2, 3, 4, 5, 7, 10, None],
               "clf__min_samples_leaf": [5, 10, 20]}

    searches = {}
    for name, pipe, grid in [("Regressão Logística", pipe_lr, grid_lr),
                             ("Árvore de Decisão", pipe_dt, grid_dt)]:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1, return_train_score=True)
        gs.fit(X_tr, y_tr)
        searches[name] = gs
        say(f"\n[{name}] melhores parâmetros: {gs.best_params_}")
        say(f"[{name}] F1-macro CV (treino): {gs.best_score_:.4f}")

    # Curva de profundidade da árvore (para o relatório)
    res = pd.DataFrame(searches["Árvore de Decisão"].cv_results_)
    res["depth"] = res["param_clf__max_depth"].apply(lambda d: 12 if d is None else d)
    g = res.groupby("depth")[["mean_train_score", "mean_test_score"]].max()
    g.plot(marker="o"); plt.xlabel("max_depth (12 = sem limite)"); plt.ylabel("F1-macro")
    plt.title("Árvore: profundidade x desempenho (CV)"); plt.legend(["Treino", "Validação (CV)"])
    savefig("tree_depth_curve.png")

    # ------------------------------------------- 5. Avaliação no teste (ambos modelos)
    say("\n== AVALIAÇÃO NO CONJUNTO DE TESTE (limiar 0,5) ==")
    comparison = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, gs) in zip(axes, searches.items()):
        proba = gs.best_estimator_.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(y_te, pred), display_labels=["<5 anos", ">=5 anos"]).plot(ax=ax, colorbar=False)
        ax.set_title(name)
        rep = classification_report(y_te, pred, target_names=["<5 anos", ">=5 anos"], output_dict=True)
        comparison[name] = {"cv_f1_macro": gs.best_score_, "test_accuracy": accuracy_score(y_te, pred),
                            "test_f1_macro": f1_score(y_te, pred, average="macro"),
                            "test_auc": roc_auc_score(y_te, proba), "best_params": gs.best_params_,
                            "report": rep}
        say(f"\n--- {name} ---")
        say(classification_report(y_te, pred, target_names=["<5 anos", ">=5 anos"]))
        say(f"AUC-ROC: {comparison[name]['test_auc']:.4f}")
    savefig("confusion_matrices.png")

    plt.figure(figsize=(5, 4))
    for name, gs in searches.items():
        p = gs.best_estimator_.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("Falso positivo"); plt.ylabel("Verdadeiro positivo")
    plt.legend(); plt.title("Curvas ROC (teste)"); savefig("roc_curves.png")

    # ----------------------------------------- 6. Escolha do modelo (usa só CV do treino)
    chosen_name = max(searches, key=lambda n: searches[n].best_score_)
    chosen = searches[chosen_name].best_estimator_
    say(f"\nModelo escolhido para deploy (maior F1-macro na CV do treino): {chosen_name}")

    # ----------------------------------------- 7. Ajuste do limiar (somente treino, OOF)
    oof = cross_val_predict(chosen, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.arange(0.30, 0.71, 0.01)
    f1s = [f1_score(y_tr, (oof >= t).astype(int), average="macro") for t in thresholds]
    best_t = float(thresholds[int(np.argmax(f1s))])
    plt.plot(thresholds, f1s); plt.axvline(best_t, color="r", ls="--")
    plt.xlabel("Limiar"); plt.ylabel("F1-macro (OOF, treino)"); plt.title(f"Escolha do limiar = {best_t:.2f}")
    savefig("threshold_curve.png")

    chosen.fit(X_tr, y_tr)  # já ajustado pelo GridSearch (refit), reforço explícito
    proba_te = chosen.predict_proba(X_te)[:, 1]
    final = {}
    for label, t in [("limiar 0,50", 0.5), (f"limiar ajustado {best_t:.2f}", best_t)]:
        pred = (proba_te >= t).astype(int)
        final[label] = {"accuracy": accuracy_score(y_te, pred),
                        "f1_macro": f1_score(y_te, pred, average="macro"),
                        "confusion_matrix": confusion_matrix(y_te, pred).tolist()}
        say(f"\nTeste com {label}: acc={final[label]['accuracy']:.3f} f1_macro={final[label]['f1_macro']:.3f}")
        say("Matriz de confusão [[TN FP],[FN TP]]:", final[label]["confusion_matrix"])
    # Decisão: usa o limiar ajustado apenas se ele foi melhor na validação (OOF) - já é por construção.
    threshold = best_t

    # ------------------------------------------------------ 8. Interpretação
    pi = permutation_importance(chosen, X_te, y_te, n_repeats=20, random_state=SEED, scoring="f1_macro")
    imp = pd.Series(pi.importances_mean, index=FEATURES).sort_values()
    imp.plot(kind="barh", figsize=(7, 6)); plt.title("Importância por permutação (teste)")
    savefig("permutation_importance.png")
    say("\nImportância por permutação (top 8):")
    say(imp.sort_values(ascending=False).head(8).round(4).to_string())

    if chosen_name == "Regressão Logística":
        coefs = pd.Series(chosen.named_steps["clf"].coef_[0], index=FEATURES).sort_values()
        coefs.plot(kind="barh", figsize=(7, 6)); plt.title("Coeficientes (variáveis padronizadas)")
        savefig("lr_coefficients.png")
    else:
        plt.figure(figsize=(16, 7))
        plot_tree(chosen.named_steps["clf"], feature_names=FEATURES, class_names=["<5", ">=5"],
                  filled=True, max_depth=3, fontsize=7)
        savefig("tree_structure.png")

    # Análise de erros: onde o modelo erra
    err = X_te.copy(); err["real"] = y_te.values; err["prob"] = proba_te
    err["pred"] = (proba_te >= threshold).astype(int)
    wrong = err[err.real != err.pred]
    say(f"\nErros no teste: {len(wrong)} de {len(err)}")
    say("Médias dos erros vs acertos (gp, min, pts):")
    say(err.assign(ok=(err.real == err.pred)).groupby("ok")[["gp", "min", "pts"]].mean().round(2).to_string())
    wrong.to_csv("reports/erros_teste.csv", index=False)

    # ----------------------------------------------------------- 9. Salvamento
    ranges = {c: [float(X_tr[c].min()), float(X_tr[c].max())] for c in FEATURES}
    bundle = {"pipeline": chosen, "threshold": threshold, "features": FEATURES,
              "model_name": chosen_name, "train_ranges": ranges,
              "permutation_importance": imp.to_dict(),
              "test_metrics": final, "seed": SEED}
    joblib.dump(bundle, "models/model.joblib")
    with open("reports/metrics.json", "w", encoding="utf-8") as f:
        json.dump({"comparison": comparison, "final": final, "threshold": threshold,
                   "chosen": chosen_name}, f, ensure_ascii=False, indent=2, default=str)
    with open("reports/treino_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    say("\nModelo salvo em models/model.joblib | logs em reports/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/nba_rookies.csv")
    main(ap.parse_args().data)
