#!/usr/bin/env bash
# Caminhos usados pelos scripts desta pasta. D (a pasta dos scripts) vem de quem inclui este arquivo.
# Cada caminho pode ser trocado pela variavel de ambiente indicada.
P=${RD_PROJETO:-$(cd "$D/../.." && pwd)}      # raiz do projeto
PY=${RD_PYTHON:-$P/.venv/bin/python}          # Python do projeto
Q=${RD_DADOS:-$HOME/.qlib/qlib_data}          # dados do Qlib
E=${RD_ESTADO:-$HOME/.rdagent/rd_mt5}         # registros, trava e ultimo arquivo de sinais
export RD_PROJETO="$P" RD_DADOS="$Q" RD_ESTADO="$E"

pasta_mt5() {  # pasta comum do MetaTrader onde os robos leem os sinais (RD_MT5_COMUM)
    local appdata
    if [ -n "${RD_MT5_COMUM:-}" ]; then
        printf '%s\n' "$RD_MT5_COMUM"
        return
    fi
    # chcp 65001: um nome de usuario com acento chega em UTF-8
    appdata=$(cd /mnt/c && /mnt/c/Windows/System32/cmd.exe /d /c 'chcp 65001 >nul & echo %APPDATA%' 2> /dev/null | tr -d '\r') || return 1
    [[ -n "$appdata" && "$appdata" != %* ]] || return 1
    appdata=$(wslpath -u "$appdata") && [ -d "$appdata" ] || return 1
    printf '%s/MetaQuotes/Terminal/Common/Files/RD\n' "$appdata"
}
