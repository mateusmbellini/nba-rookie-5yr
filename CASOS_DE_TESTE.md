# Casos de teste (preencher com as saídas reais do app publicado + captura de tela)

| # | Tipo | Entrada | Esperado | Obtido | Comentário | Captura |
|---|---|---|---|---|---|---|
| 1 | Válida | Exemplo "Titular com muitos minutos" (gp=80, min=32, pts=15, fgm=5.6, fga=12, tpm=1.2, tpa=3.4, ftm=2.6, fta=3.3, oreb=1.2, dreb=4, ast=3.5, stl=1, blk=0.5, tov=2.3) | Alta prob. de ≥5 anos (muito jogo e minutos). Resultado real do jogador não conhecido. | [prob./classe] | | cap1.png |
| 2 | Válida | Exemplo "Reserva com poucos minutos" (gp=25, min=6, pts=2, ...) | Baixa prob. | [..] | | cap2.png |
| 3 | Válida | Perfil intermediário (gp=55, min=18, pts=7, ...) | Prob. intermediária | [..] | | cap3.png |
| 4 | Inválida | fgm=9 e fga=5 (convertidos > tentados) | Mensagem de erro, sem previsão | [..] | | cap4.png |
| 5 | Limítrofe | Perfil com poucos jogos mas muitos minutos (gp=12, min=28, pts=11, ...) | Prob. próxima do limiar; aviso de baixa confiança | [..] | | cap5.png |
| 6 (extra) | Consistência | Mesma entrada do caso 1 em `predict_check.py` | Probabilidade idêntica à do app | [..] | | |

Observação: quando o desfecho real do jogador não é conhecido, registre isso; a saída do modelo não é "verdade".
