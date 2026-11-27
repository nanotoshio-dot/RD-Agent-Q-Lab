#!/usr/bin/env bash
# Testa o auto_atualizar.sh numa area temporaria, com construtor, rede e MetaTrader falsos.
# Nada fora da area temporaria e tocado. Uso: tools/rd_mt5/tests/teste_auto.sh (cerca de 1 minuto)
set -uo pipefail
ORIG=$(dirname "$(dirname "$(readlink -f "$0")")")
S=$(mktemp -d /tmp/rd_teste.XXXXXX) || exit 1
[ -n "$S" ] && [ -d "$S" ] || exit 1
mkdir -p "$S/rd" "$S/q" "$S/p" "$S/bin" "$S/mt5"
export RD_PROJETO=$S/p RD_PYTHON=$S/bin/python RD_DADOS=$S/q RD_ESTADO=$S/estado RD_MT5_COMUM=$S/mt5
export FAKE_RAIZ=$S FAKE_MODE=$S/modo FAKE_CALLS=$S/chamadas FAKE_FIM_FILE=$S/fim
if [ -z "${REAL_PY:-}" ]; then
    REAL_PY=$ORIG/../../.venv/bin/python
    [ -x "$REAL_PY" ] || REAL_PY=$(command -v python3)
fi
export REAL_PY
export PATH="$S/bin:$PATH"
SINAIS=$S/mt5/rd_sinais.csv
cp "$ORIG/auto_atualizar.sh" "$ORIG/auditar_dados.py" "$ORIG/comum.sh" "$S/rd/"
printf '#!/bin/sh\nexit 0\n' > "$S/bin/curl"
# disco sempre com espaco, para o teste nao depender deste computador
printf '#!/bin/sh\necho cabecalho\necho "falso 100000000 0 90000000 1%% /"\n' > "$S/bin/df"
cat > "$S/rd/atualizar_sinais.sh" <<'FIM_SINAIS'
#!/usr/bin/env bash
printf 'data_sinal,mercado\n%s,us-sp500\n' "$(tail -n 1 "$RD_DADOS/us_data_2026/calendars/day.txt")" \
    > "$RD_MT5_COMUM/rd_sinais.csv"
FIM_SINAIS
cat > "$S/bin/python" <<'FIM_PYTHON'
#!/usr/bin/env bash
if [ "$1" = -c ]; then cat "$FAKE_FIM_FILE"; exit 0; fi
if [ "$1" = -I ]; then exec "$REAL_PY" "$@"; fi
while [ $# -gt 0 ]; do
    case "$1" in --source) S=$2; shift ;; --target) T=$2; shift ;; --end) E=$2; shift ;; esac
    shift
done
MODE=$(cat "$FAKE_MODE")
if [ -d "$T" ]; then echo "retomada $MODE $(basename "$T")" >> "$FAKE_CALLS"; else echo "nova $MODE $(basename "$T")" >> "$FAKE_CALLS"; fi
if [ -e "$T" ] && [ ! -d "$T/_sources" ]; then echo "exists and was not created by this tool"; exit 1; fi
[ -e "$T" ] || { cp -r "$S" "$T"; mkdir "$T/_sources"; }
CUT=$(tail -n 1 "$S/calendars/day.txt")
[ "$(tail -n 1 "$T/calendars/day.txt")" = "$CUT" ] || { echo "already extended"; exit 1; }
echo "$CUT" > "$T/_sources/cutover.txt"
mkdir -p "$T/_sources/yahoo_raw" "$T/_sources/yahoo_planned" "$T/_sources/yahoo_normalized" "$T/_sources/yahoo_final"
[ "$MODE" = falha_download ] && { echo "3 members without new data: X, Y, Z. Nothing was changed"; exit 1; }
echo "$E" >> "$T/calendars/day.txt"
[ "$MODE" = falha_depois ] && exit 1
BARS=2; [ "$MODE" = reprovado ] && BARS=50
SP=499; [ "$MODE" = poucas ] && SP=484
B="{\"^GSPC\": \"$E\", \"^NDX\": \"$E\", \"^DJI\": \"$E\"}"
cat > "$T/_sources/audit.json" <<J
{"calendar": ["2000-01-01", "$E"], "misaligned": [], "benchmark_missing_days_after_cutover": {"^GSPC": 0},
 "benchmark_last_day": $B, "non_positive_closes_after_cutover": 0,
 "ohlc_violations_after_cutover": {"bars": $BARS}, "foreign_closes_left": {},
 "sp500": {"$E": "$SP/$SP of 503 in the source"}, "nasdaq100": {"$E": "100/100 of 101 in the source"}}
J
echo "ABC" > "$T/_sources/unrefreshed.txt"
echo "2026-01-01T00:00:00Z_end=$E" > "$T/_sources/build_id.txt"
if [ "$MODE" = ok_com_pesquisa ]; then (exec -a "$FAKE_RAIZ/p/bin/qrun" sleep 6) > /dev/null 2>&1 9>&- 3>&- & fi
exit 0
FIM_PYTHON
# so enxerga os processos do proprio teste: pesquisas reais em andamento nao atrapalham
cat > "$S/bin/pgrep" <<'FIM_PGREP'
#!/usr/bin/env bash
/usr/bin/pgrep -a -f "$2" | grep -F -- "$FAKE_RAIZ"
FIM_PGREP
chmod +x "$S/bin/python" "$S/bin/curl" "$S/bin/df" "$S/bin/pgrep" "$S/rd/auto_atualizar.sh" "$S/rd/atualizar_sinais.sh"

dia() { mkdir -p "$1/calendars" && echo "$2" > "$1/calendars/day.txt"; }
dia "$S/q/us_data" 2020-11-10
dia "$S/q/us_data_2026" 2026-11-23
mkdir -p "$S/q/us_data_2026/_sources/yahoo_raw"

FALHAS=0
rodar() {  # $1 = FIM, $2 = modo
    echo "$1" > "$FAKE_FIM_FILE"; echo "$2" > "$FAKE_MODE"
    "$S/rd/auto_atualizar.sh" > "$S/janela" 2>&1; RC=$?
}
ver() {  # $1 = descricao, resto = comando que deve dar certo
    local d=$1; shift
    if "$@"; then echo "  ok    $d"; else echo "  FALHA $d"; FALHAS=$((FALHAS + 1)); fi
}
tem() { [ "$(tail -n 1 "$S/q/$1/calendars/day.txt" 2> /dev/null)" = "$2" ]; }
sinal() { [ "$(sed -n 2p "$SINAIS" | cut -d, -f1)" = "$1" ]; }
sem() { [ -z "$(ls -d "$S"/q/$1 2> /dev/null)" ]; }
chamadas() { [ "$(wc -l < "$FAKE_CALLS")" = "$1" ]; }
ultima() { [ "$(tail -n 1 "$FAKE_CALLS")" = "$1" ]; }
tentativas() { [ "$(cd "$S/q" && echo us_data_2026_auto_*)" = "$1" ]; }

echo "1. primeira atualizacao"; rodar 2026-11-25 ok
ver "rc 0" [ "$RC" = 0 ]
ver "ativo ate 25" tem us_data_2026 2026-11-25
ver "anterior = dados originais ate 23" tem us_data_2026_anterior 2026-11-23
ver "ativo marcado" [ -e "$S/q/us_data_2026/.criado_pelo_atualizador" ]
ver "pastas de trabalho apagadas" [ ! -e "$S/q/us_data_2026/_sources/yahoo_final" ]
ver "sem tentativas sobrando" sem 'us_data_2026_auto_*'
ver "sinais de 25" sinal 2026-11-25

echo "2. nada a fazer"; rodar 2026-11-25 ok
ver "rc 0" [ "$RC" = 0 ]
ver "sem construir" chamadas 1

echo "3. rotacao: anterior nao marcado e guardado"; rodar 2026-11-27 ok
ver "ativo ate 27" tem us_data_2026 2026-11-27
ver "anterior ate 25" tem us_data_2026_anterior 2026-11-25
ver "original guardado" tem "$(basename "$(ls -d "$S"/q/us_data_2026_anterior_guardado_* | head -n 1)")" 2026-11-23

echo "4. rotacao: anterior marcado e apagado"; sleep 1; rodar 2026-11-30 ok
ver "anterior ate 27" tem us_data_2026_anterior 2026-11-27
ver "um so guardado" [ "$(ls -d "$S"/q/us_data_2026_anterior_guardado_* | wc -l)" = 1 ]
ver "nada .apagar" sem '*.apagar'

echo "5. falha no download"; rodar 2026-12-01 falha_download
ver "rc != 0" [ "$RC" != 0 ]
ver "ativo intacto" tem us_data_2026 2026-11-30
ver "tentativa guardada com .falhou" [ -e "$S/q/us_data_2026_auto_2026-12-01/.falhou" ]
ver "sinais continuam de 30" sinal 2026-11-30
ver "janela avisa" grep -q "nao deu certo" "$S/janela"
ver "janela mostra o motivo" grep -q "members without new data" "$S/janela"

echo "6. segunda falha apaga a tentativa"; rodar 2026-12-01 falha_download
ver "segunda chamada retomou" ultima "retomada falha_download us_data_2026_auto_2026-12-01"
ver "tentativa apagada" sem 'us_data_2026_auto_*'

echo "7. falha uma vez e depois retoma"; rodar 2026-12-01 falha_download; rodar 2026-12-01 ok
ver "retomou" ultima "retomada ok us_data_2026_auto_2026-12-01"
ver "ativo ate 01" tem us_data_2026 2026-12-01
ver "sem .falhou no ativo" [ ! -e "$S/q/us_data_2026/.falhou" ]

echo "8. falha depois de mexer nos precos: refaz do zero"; rodar 2026-12-02 falha_depois; rodar 2026-12-02 ok
ver "refez do zero" ultima "nova ok us_data_2026_auto_2026-12-02"
ver "ativo ate 02" tem us_data_2026 2026-12-02

echo "9. auditoria reprova"; rodar 2026-12-03 reprovado
ver "rc != 0" [ "$RC" != 0 ]
ver "ativo intacto" tem us_data_2026 2026-12-02
ver "relatorio guardado" [ -s "$S/estado/auto_ultima_reprovada.json" ]
ver "janela mostra o motivo" grep -q "barras inconsistentes" "$S/janela"
rodar 2026-12-04 ok
ver "tentativa reprovada apagada" sem 'us_data_2026_auto_*'
ver "ativo ate 04" tem us_data_2026 2026-12-04

echo "10. pesquisa comeca durante a construcao"; rodar 2026-12-07 ok_com_pesquisa
ver "rc 0" [ "$RC" = 0 ]
ver "troca adiada" tem us_data_2026 2026-12-04
ver "tentativa pronta guardada" [ -e "$S/q/us_data_2026_auto_2026-12-07/.criado_pelo_atualizador" ]
N=$(wc -l < "$FAKE_CALLS")
echo "11. pesquisa ainda rodando no inicio"; rodar 2026-12-07 ok
ver "nao construiu" chamadas "$N"
ver "avisou" grep -q "nada foi atualizado" "$S/janela"
sleep 7
rodar 2026-12-07 ok
ver "trocou sem construir de novo" chamadas "$N"
ver "ativo ate 07" tem us_data_2026 2026-12-07

echo "12. troca interrompida"; mv "$S/q/us_data_2026_anterior" "$S/q/de_lado"; mv "$S/q/us_data_2026" "$S/q/us_data_2026_anterior"
rodar 2026-12-07 ok
ver "restaurou" tem us_data_2026 2026-12-07
ver "rc 0" [ "$RC" = 0 ]

echo "13. outra atualizacao em andamento"
( flock 9; sleep 5 ) 9> "$S/estado/auto_atualizar.lock" & sleep 1
rodar 2026-12-08 ok
ver "nao fez nada" tem us_data_2026 2026-12-07
ver "avisou" grep -q "outra atualizacao" "$S/janela"
wait

echo "14. sinais atrasados em relacao aos dados"; printf 'data_sinal,mercado\n2020-01-01,us-sp500\n' > "$SINAIS"
rodar 2026-12-07 ok
ver "regerou" sinal 2026-12-07

echo "15. troca adiada e dia novo: entra a pronta e depois constroi a nova"; rodar 2026-12-08 ok_com_pesquisa; sleep 7
rodar 2026-12-09 ok
ver "rc 0" [ "$RC" = 0 ]
ver "ativo ate 09" tem us_data_2026 2026-12-09
ver "anterior ate 08" tem us_data_2026_anterior 2026-12-08
ver "sem tentativas" sem 'us_data_2026_auto_*'

echo "16. troca adiada e a construcao nova falha: a pronta entra mesmo assim"; rodar 2026-12-10 ok_com_pesquisa; sleep 7
rodar 2026-12-11 falha_download
ver "rc != 0" [ "$RC" != 0 ]
ver "ativo ate 10" tem us_data_2026 2026-12-10
ver "sinais de 10" sinal 2026-12-10

echo "17. pesquisa em duas construcoes seguidas: nao acumula pastas"; rodar 2026-12-14 ok_com_pesquisa; sleep 7
rodar 2026-12-15 ok_com_pesquisa
ver "ativo ate 14" tem us_data_2026 2026-12-14
ver "so a tentativa de 15" tentativas us_data_2026_auto_2026-12-15
sleep 7; N=$(wc -l < "$FAKE_CALLS"); rodar 2026-12-15 ok
ver "trocou sem construir" chamadas "$N"
ver "ativo ate 15" tem us_data_2026 2026-12-15

echo "18. pasta com o mesmo nome que nao foi criada pelo script"
mkdir -p "$S/q/us_data_2026_auto_2026-12-16" && echo meu > "$S/q/us_data_2026_auto_2026-12-16/arquivo"
N=$(wc -l < "$FAKE_CALLS"); rodar 2026-12-16 ok
ver "rc != 0" [ "$RC" != 0 ]
ver "pasta intacta" [ "$(cat "$S/q/us_data_2026_auto_2026-12-16/arquivo")" = meu ]
ver "nao construiu" chamadas "$N"

echo "19. perdeu muitos membros em relacao aos dados em uso"; rodar 2026-12-17 poucas
ver "rc != 0" [ "$RC" != 0 ]
ver "ativo intacto" tem us_data_2026 2026-12-15
ver "janela mostra a queda" grep -q "caiu de 499" "$S/janela"
rodar 2026-12-18 ok
ver "ativo ate 18" tem us_data_2026 2026-12-18

echo "20. quem conta como pesquisa rodando"
(exec -a "python -I $S/gerar_sinais.py" sleep 5) & sleep 0.5; rodar 2026-12-18 ok
ver "script python em uso adia" grep -q "nada foi atualizado" "$S/janela"
wait
(exec -a "nano $S/gerar_sinais.py" sleep 5) & sleep 0.5; rodar 2026-12-18 ok
ver "editor aberto nao adia" grep -q "sinais ja sao" "$S/janela"
wait

echo "21. sobras de exclusao interrompida"
mkdir -p "$S/q/us_data_2026_anterior.apagar" "$S/q/us_data_2026_auto_2026-01-02.apagar"
rodar 2026-12-18 ok
ver "sobras apagadas" sem '*.apagar'

if [ "$FALHAS" = 0 ]; then
    echo "TODOS OS TESTES PASSARAM"
    rm -rf -- "$S"
else
    echo "$FALHAS FALHA(S); area mantida em $S"
fi
exit "$FALHAS"
