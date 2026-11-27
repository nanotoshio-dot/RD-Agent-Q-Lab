#!/usr/bin/env bash
# Gera o ranking do RD com os dados ativos e copia para a pasta comum do MetaTrader, onde os robos o leem.
# So copia um arquivo novo, gerado sem erro nesta execucao.
# Uso: tools/rd_mt5/atualizar_sinais.sh
set -euo pipefail
D=$(dirname "$(readlink -f "$0")")
. "$D/comum.sh"
DEST=$(pasta_mt5) || { echo "FALHOU: pasta comum do MetaTrader nao encontrada (defina RD_MT5_COMUM)"; exit 1; }
mkdir -p "$E"
NOVO=$(mktemp -u "$E/sinais.XXXXXX")
LOG="$E/atualizar_sinais.log"
if ! "$PY" -I "$D/gerar_sinais.py" us-sp500 20 "$NOVO" > "$LOG" 2>&1; then
    grep -E "Sem previsao|Error|Traceback" "$LOG" | tail -n 5 || true
    echo "FALHOU: nada foi copiado (veja $LOG)"
    exit 1
fi
grep -E "sinais de|modelo:" "$LOG" || true
test -s "$NOVO"
cp "$NOVO" "$E/rd_sinais.csv"
mkdir -p "$DEST"
cp "$NOVO" "$DEST/rd_sinais.tmp"
mv -f "$DEST/rd_sinais.tmp" "$DEST/rd_sinais.csv"
rm -f "$NOVO"
echo "sinais copiados para o MetaTrader"
