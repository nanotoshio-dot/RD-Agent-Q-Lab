#!/usr/bin/env bash
# Atualiza os dados dos EUA ate o ultimo pregao, gera os sinais do RD e copia para o MetaTrader.
# Feito para o Agendador de Tarefas do Windows (uma vez por dia). Registro: $RD_ESTADO/auto_atualizar.log
#
# So troca os dados em uso quando a construcao termina e a auditoria passa. A versao anterior fica em
# us_data_2026_anterior. So apaga pastas que este script criou: tentativas us_data_2026_auto_<dia> listadas
# em auto_tentativas.txt e versoes antigas marcadas com .criado_pelo_atualizador.
set -euo pipefail
D=$(dirname "$(readlink -f "$0")")
. "$D/comum.sh"
ATIVO=$Q/us_data_2026
ANTERIOR=$Q/us_data_2026_anterior
MARCA=.criado_pelo_atualizador
LISTA=$E/auto_tentativas.txt
LOG=$E/auto_atualizar.log

ultimo_dia() { tail -n 1 "$1/calendars/day.txt" | tr -d '[:space:]'; }

aviso() {  # no registro e na janela
    echo "$*"
    echo "RD: $*" >&3 || true
}

ocupado() {  # algo que le ou constroi os dados esta rodando
    pgrep -f "[r]dagent\.app\.(market_launcher|qlib_rd_loop)|[r]dagent fin_|[b]in/qrun|[u]s_market_data (build|audit)|[p]ython[^ ]* (.*[ /])?(gerar_sinais|run_baseline|run_model_baseline|run_template|select_lgb|holdout_compare)\.py" > /dev/null
}

retomavel() {  # a construcao parou antes de mexer nos precos: pode continuar com o que ja baixou
    [ -d "$1/_sources" ] || return 1
    [ -f "$1/_sources/cutover.txt" ] || return 0
    [ "$(tr -d '[:space:]' < "$1/_sources/cutover.txt")" = "$(ultimo_dia "$1")" ]
}

apagar() {  # renomeia antes: uma exclusao interrompida nunca parece uma pasta inteira
    mv -T -- "$1" "$1.apagar"
    rm -rf -- "$1.apagar"
}

tentativas() {  # as pastas criadas por este script que ainda existem
    local pasta
    [ -f "$LISTA" ] || return 0
    while read -r pasta; do
        if [[ "$pasta" == "$Q"/us_data_2026_auto_????-??-?? && -d "$pasta" ]]; then echo "$pasta"; fi
    done < "$LISTA"
}

descartar_tentativas() {  # $1: "incompletas" ou "todas"; $2: pasta a preservar
    local velha
    while IFS= read -r velha; do
        [ "$velha" != "$2" ] || continue
        if [ "$1" = todas ] || [ ! -e "$velha/$MARCA" ]; then
            echo "apagando tentativa antiga: $velha"
            apagar "$velha"
        fi
    done < <(tentativas)
}

pronta_mais_nova() {  # a mais nova aprovada que ainda nao entrou em uso, se for mais nova que os dados em uso
    local pasta atual melhor=""
    atual=$(ultimo_dia "$ATIVO")
    while IFS= read -r pasta; do
        if [ -e "$pasta/$MARCA" ] && [[ "$(ultimo_dia "$pasta")" > "$atual" && "$pasta" > "$melhor" ]]; then
            melhor=$pasta
        fi
    done < <(tentativas)
    echo "$melhor"
}

trocar() {  # $1: pasta aprovada que passa a ser a usada
    if [ -e "$ANTERIOR" ]; then
        if [ -e "$ANTERIOR/$MARCA" ]; then
            apagar "$ANTERIOR"
        else
            mv -T -- "$ANTERIOR" "${ANTERIOR}_guardado_$(date +%Y%m%d%H%M%S)"
        fi
    fi
    mv -T -- "$ATIVO" "$ANTERIOR"
    if ! mv -T -- "$1" "$ATIVO"; then
        mv -T -- "$ANTERIOR" "$ATIVO"
        aviso "a troca falhou; os dados anteriores foram restaurados"
        return 1
    fi
    aviso "dados em uso agora vao ate $(ultimo_dia "$ATIVO")"
}

atualizar_dados() {
    local novo=$Q/us_data_2026_auto_$FIM pronta pasta livre modelo parecer
    # uma versao aprovada que ficou esperando (havia pesquisa rodando) entra primeiro
    pronta=$(pronta_mais_nova)
    if [ -n "$pronta" ] && ! ocupado; then
        trocar "$pronta"
        if [[ ! "$(ultimo_dia "$ATIVO")" < "$FIM" ]]; then
            descartar_tentativas todas ""
            return 0
        fi
    fi
    if ! grep -qxF "$novo" "$LISTA" 2> /dev/null; then
        if [ -e "$novo" ]; then
            aviso "$novo ja existe e nao foi criada por este script; nada foi feito"
            return 1
        fi
        echo "$novo" >> "$LISTA"
    fi
    descartar_tentativas incompletas "$novo"
    if [ ! -e "$novo/$MARCA" ]; then
        if [ -e "$novo" ] && ! retomavel "$novo"; then
            echo "sobra de uma tentativa anterior: comecando do zero"
            apagar "$novo"
        fi
        for pasta in "$Q" /mnt/c; do
            [ -d "$pasta" ] || continue
            livre=$(df -Pk "$pasta" | awk 'NR==2 {print int($4 / 1048576)}')
            if [ "$livre" -lt 6 ]; then
                aviso "pouco espaco em $pasta: $livre GB livres (preciso de 6)"
                return 1
            fi
        done
        # a rede pode demorar logo depois de ligar o computador
        for _ in $(seq 30); do
            curl -s -m 10 -o /dev/null https://query1.finance.yahoo.com/ && break
            sleep 10
        done
        # libera a memoria do modelo local, se estiver carregado
        for modelo in $(curl -s -m 5 http://127.0.0.1:11434/api/ps | grep -oE '"name":"[^"]+"' | cut -d'"' -f4 || true); do
            curl -s -m 30 http://127.0.0.1:11434/api/generate -d "{\"model\":\"$modelo\",\"keep_alive\":0}" > /dev/null || true
        done
        aviso "baixando os dados ate $FIM (a parte demorada)"
        if ! (cd "$P" && timeout -k 60 10800 "$PY" -u -m rdagent.app.us_market_data build --source "$Q/us_data" \
            --target "$novo" --end "$FIM" --reuse-raw "$ATIVO/_sources/yahoo_raw") > "$E/auto_build.log" 2>&1; then
            aviso "a construcao falhou e os dados em uso nao mudaram. Ultima linha de auto_build.log:"
            aviso "$(grep -v '^[[:space:]]*$' "$E/auto_build.log" | tail -n 1 | cut -c1-300)"
            # outra execucao para o mesmo dia continua de onde parou; se falhar de novo, a tentativa e apagada
            if [ -e "$novo/.falhou" ]; then apagar "$novo"; elif [ -d "$novo" ]; then touch "$novo/.falhou"; fi
            return 1
        fi
        if ! parecer=$("$PY" -I "$D/auditar_dados.py" "$novo" "$FIM" "$ATIVO" 2>&1); then
            cp -- "$novo/_sources/audit.json" "$E/auto_ultima_reprovada.json" 2> /dev/null || true
            aviso "os dados em uso nao mudaram: $parecer"
            return 1
        fi
        echo "$parecer"
        # as pastas de trabalho da construcao (~370 MB) nao sao usadas depois
        rm -rf -- "$novo/_sources/yahoo_planned" "$novo/_sources/yahoo_normalized" "$novo/_sources/yahoo_final"
        rm -f -- "$novo/.falhou"
        touch "$novo/$MARCA"
    fi
    descartar_tentativas todas "$novo"
    if ocupado; then
        aviso "ha pesquisa rodando: a troca dos dados fica para a proxima execucao"
        return 0
    fi
    trocar "$novo"
}

exec 3>&1
mkdir -p "$E"
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 1000000 ]; then mv -f -- "$LOG" "$LOG.antigo"; fi
echo "RD: atualizando os sinais. Pode levar ate 2 horas: minimize esta janela, mas nao feche." >&3
exec >> "$LOG" 2>&1
trap 'rc=$?; echo "=== $(date "+%F %T") fim rc=$rc"; if [ "$rc" -eq 0 ]; then echo "RD: fim" >&3; else echo "RD: nao deu certo (veja $LOG)" >&3; fi' EXIT
echo "=== $(date '+%F %T') inicio"
exec 9> "$E/auto_atualizar.lock"
flock -n 9 || { aviso "outra atualizacao em andamento"; exit 0; }
if ocupado; then
    aviso "ha pesquisa ou construcao rodando; nada foi atualizado agora"
    exit 0
fi
# uma troca interrompida (computador desligado no meio) deixa so a versao anterior
if [ ! -e "$ATIVO" ] && [ -d "$ANTERIOR" ]; then
    mv -T -- "$ANTERIOR" "$ATIVO"
    aviso "troca interrompida: versao anterior restaurada"
fi
test -s "$ATIVO/calendars/day.txt" || { aviso "dados em uso nao encontrados em $ATIVO"; exit 1; }
# sobras de uma exclusao interrompida
rm -rf -- "$ANTERIOR.apagar" "$Q"/us_data_2026_auto_????-??-??.apagar
# ultimo pregao antes de hoje, pelo calendario de feriados da bolsa de Nova York
FIM=$(cd "$P" && "$PY" -c "from datetime import date, timedelta; from rdagent.app.us_market_data import last_business_day; print(last_business_day((date.today() - timedelta(days=1)).isoformat()))" | tail -n 1)
[[ "$FIM" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || { aviso "nao consegui calcular o ultimo pregao"; exit 1; }
FALHA=0
if [[ "$(ultimo_dia "$ATIVO")" < "$FIM" ]]; then
    set +e
    (set -e; atualizar_dados)
    FALHA=$?
    set -e
else
    echo "dados ja vao ate $(ultimo_dia "$ATIVO")"
fi
# os sinais acompanham sempre os dados em uso, mesmo quando a atualizacao dos dados falhou
TEM=$(ultimo_dia "$ATIVO")
DEST=$(pasta_mt5) || { aviso "pasta comum do MetaTrader nao encontrada (defina RD_MT5_COMUM)"; exit 1; }
DATA_SINAL=$(sed -n '2p' "$DEST/rd_sinais.csv" 2> /dev/null | cut -d, -f1 || true)
if [ "$DATA_SINAL" != "$TEM" ]; then
    "$D/atualizar_sinais.sh"
    aviso "sinais de $TEM copiados para o MetaTrader"
else
    aviso "sinais ja sao de $TEM"
fi
exit "$FALHA"
