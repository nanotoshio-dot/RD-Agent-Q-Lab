//+------------------------------------------------------------------+
//| RD_Demo.mq5                                                      |
//| Executa o ranking do RD-Agent SOMENTE em conta DEMO hedging.     |
//| Compra o lote minimo, com stop, e so mexe nas proprias posicoes. |
//| Rode em um unico computador.                                     |
//+------------------------------------------------------------------+
#property copyright "RD-Agent-Q-Lab"
#property version   "1.10"
#property description "Somente conta DEMO. Lote minimo, stop obrigatorio e limites de posicoes, valor e ordens."

#include <Trade\Trade.mqh>

input bool   InpOperarDemo       = false;               // Enviar ordens (somente conta DEMO)
input string InpArquivoSinais    = "RD\\rd_sinais.csv"; // Arquivo de sinais (pasta comum)
input string InpArquivoDiario    = "RD\\rd_ordens.csv"; // Diario de ordens (pasta comum)
input string InpArquivoParar     = "RD\\PARAR";         // Se este arquivo existir, nao envia ordens
input string InpMercado          = "us-sp500";          // Mercado aceito no arquivo de sinais
input string InpGrupoAcoes       = "share;stock;equit"; // Grupos de ativos aceitos (acoes)
input int    InpMaxPosicoes      = 5;                   // Maximo de posicoes do robo (1 a 10)
input int    InpMaxOrdensDia     = 10;                  // Maximo de compras por dia (1 a 20)
input double InpStopPercent      = 10.0;                // Stop abaixo da compra, em % (1 a 50)
input int    InpMaxDiasPosicao   = 10;                  // Fecha a posicao depois de N dias (1 a 60)
input int    InpMaxIdadeSinal    = 7;                   // Ignora sinais com mais de N dias (1 a 10)
input double InpMaxSpreadPercent = 1.0;                 // Nao compra com spread acima de N% do preco
input double InpMaxPosicaoPercent = 50.0;               // Valor maximo por posicao, % do patrimonio
input double InpMaxExposicaoPercent = 100.0;            // Valor maximo somado, % do patrimonio
input int    InpSegundos         = 30;                  // Ciclo a cada N segundos (10 a 60)
input long   InpMagic            = 7720261;             // Identificador das posicoes deste robo

#define MAX_SINAIS    20
#define ARQUIVO_TRAVA "RD\\rd_demo.lock"

struct Sinal
  {
   string            ticker;
   string            nome;    // nome na corretora; vazio se nao achado
   string            grupo;   // caminho do ativo na corretora
   bool              avisado; // o caminho recusado ja foi escrito no registro
  };

Sinal    g_sinais[];
int      g_total = 0;
string   g_data = "";
datetime g_quando = 0;
string   g_conteudo = "";
string   g_erro_arquivo = "";
int      g_linhas_painel = 0;
int      g_trava = INVALID_HANDLE;
long     g_login = 0;        // conta vista na primeira leitura, nunca impressa
string   g_servidor = "";
datetime g_conectado_desde = 0;
int      g_recusas = 0;      // ordens recusadas em seguida
string   g_dia_recusas = "";
uint     g_ultimo_codigo = 0;
CTrade   g_trade;

//+------------------------------------------------------------------+
//| Conta: -1 desconhecida (sem conexao), 0 demo, 1 nao e demo       |
//+------------------------------------------------------------------+
int EstadoConta()
  {
   if(!TerminalInfoInteger(TERMINAL_CONNECTED) || AccountInfoInteger(ACCOUNT_LOGIN) <= 0)
      return -1;
   return AccountInfoInteger(ACCOUNT_TRADE_MODE) == ACCOUNT_TRADE_MODE_DEMO ? 0 : 1;
  }

bool ContaDemo()
  {
   if(EstadoConta() != 0)
      return false;
   return g_login == 0 || (AccountInfoInteger(ACCOUNT_LOGIN) == g_login &&
                           AccountInfoString(ACCOUNT_SERVER) == g_servidor);
  }

bool Hedging()
  {
   return AccountInfoInteger(ACCOUNT_MARGIN_MODE) == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING;
  }

void AtualizarConexao()
  {
   if(EstadoConta() == 0 && g_login == 0)
     {
      g_login = AccountInfoInteger(ACCOUNT_LOGIN);
      g_servidor = AccountInfoString(ACCOUNT_SERVER);
     }
   if(!ContaDemo())
      g_conectado_desde = 0;
   else if(g_conectado_desde == 0)
      g_conectado_desde = TimeLocal();
  }

// conectado ha pelo menos dois ciclos, para as posicoes ja estarem sincronizadas
bool Sincronizado()
  {
   return g_conectado_desde > 0 && TimeLocal() - g_conectado_desde >= 2 * InpSegundos;
  }

//+------------------------------------------------------------------+
void Candidatos(const string ticker, string &lista[])
  {
   ArrayResize(lista, 7);
   lista[0] = ticker;
   lista[1] = ticker + ".US";
   lista[2] = ticker + ".us";
   lista[3] = ticker + ".NYSE";
   lista[4] = ticker + ".NAS";
   lista[5] = "#" + ticker;
   lista[6] = ticker + "-US";
  }

bool EhCandidato(const string ticker, const string simbolo)
  {
   string lista[];
   Candidatos(ticker, lista);
   for(int i = 0; i < ArraySize(lista); i++)
      if(lista[i] == simbolo)
         return true;
   return false;
  }

bool GrupoDeAcoes(string caminho)
  {
   StringToLower(caminho);
   string grupos[];
   int n = StringSplit(InpGrupoAcoes, ';', grupos);
   for(int i = 0; i < n; i++)
     {
      string g = grupos[i];
      StringToLower(g);
      if(StringLen(g) > 0 && StringFind(caminho, g) >= 0)
         return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Acao na corretora: grupo de acoes, cotada em USD, nunca moedas   |
//+------------------------------------------------------------------+
string AcharSimbolo(const string ticker, string &grupo)
  {
   grupo = "";
   string lista[];
   Candidatos(ticker, lista);
   for(int i = 0; i < ArraySize(lista); i++)
     {
      bool personalizado = false;
      if(!SymbolExist(lista[i], personalizado) || personalizado)
         continue;
      SymbolSelect(lista[i], true);
      ENUM_SYMBOL_CALC_MODE modo = (ENUM_SYMBOL_CALC_MODE)SymbolInfoInteger(lista[i], SYMBOL_TRADE_CALC_MODE);
      if(modo == SYMBOL_CALC_MODE_FOREX || modo == SYMBOL_CALC_MODE_FOREX_NO_LEVERAGE)
         continue;
      if(SymbolInfoString(lista[i], SYMBOL_CURRENCY_PROFIT) != "USD")
         continue;
      grupo = SymbolInfoString(lista[i], SYMBOL_PATH);
      if(!GrupoDeAcoes(grupo))
         continue;
      return lista[i];
     }
   return "";
  }

void ResolverFaltantes()
  {
   if(EstadoConta() != 0)
      return;
   for(int j = 0; j < g_total; j++)
      if(g_sinais[j].nome == "")
        {
         g_sinais[j].nome = AcharSimbolo(g_sinais[j].ticker, g_sinais[j].grupo);
         if(g_sinais[j].nome != "")
            Print("RD DEMO: ", g_sinais[j].ticker, " -> ", g_sinais[j].nome, " (", g_sinais[j].grupo, ")");
         else if(g_sinais[j].grupo != "" && !g_sinais[j].avisado)
           {
            Print("RD DEMO: ", g_sinais[j].ticker, " recusado, grupo fora das acoes: ", g_sinais[j].grupo);
            g_sinais[j].avisado = true;
           }
        }
  }

//+------------------------------------------------------------------+
//| Le o arquivo de sinais. So um arquivo completo, do mercado certo |
//| e nao mais antigo substitui o anterior.                          |
//+------------------------------------------------------------------+
void CarregarSinais()
  {
   int arquivo = FileOpen(InpArquivoSinais, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(arquivo == INVALID_HANDLE)
     {
      g_erro_arquivo = "arquivo de sinais nao encontrado";
      return;
     }
   string linhas[];
   int n = 0;
   string conteudo = "";
   while(!FileIsEnding(arquivo))
     {
      string linha = FileReadString(arquivo);
      if(StringLen(linha) == 0)
         continue;
      ArrayResize(linhas, n + 1);
      linhas[n++] = linha;
      conteudo += linha + ";";
     }
   FileClose(arquivo);
   if(conteudo == g_conteudo)
     {
      g_erro_arquivo = "";
      return;
     }
   int total = n - 1;
   if(total < 1 || total > MAX_SINAIS || StringFind(linhas[0], "data_sinal,mercado,") != 0)
     {
      g_erro_arquivo = "arquivo vazio, longo ou sem cabecalho";
      return;
     }
   Sinal novos[];
   ArrayResize(novos, total);
   string data = "";
   for(int i = 1; i <= total; i++)
     {
      string campo[];
      if(StringSplit(linhas[i], ',', campo) < 7 || StringLen(campo[0]) != 10 || StringLen(campo[3]) == 0 ||
         campo[1] != InpMercado || campo[2] != IntegerToString(i) || campo[6] != IntegerToString(total))
        {
         g_erro_arquivo = "arquivo incompleto ou invalido";
         return;
        }
      if(data == "")
         data = campo[0];
      for(int k = 0; k < i - 1; k++)
         if(novos[k].ticker == campo[3])
           {
            g_erro_arquivo = "ativo repetido no arquivo";
            return;
           }
      if(campo[0] != data)
        {
         g_erro_arquivo = "datas diferentes no arquivo";
         return;
        }
      novos[i - 1].ticker = campo[3];
      novos[i - 1].nome = "";
      novos[i - 1].grupo = "";
      novos[i - 1].avisado = false;
     }
   string com_pontos = data;
   StringReplace(com_pontos, "-", ".");
   datetime quando = StringToTime(com_pontos);
   if(TimeToString(quando, TIME_DATE) != com_pontos || quando > TimeLocal())
     {
      g_erro_arquivo = "data invalida no arquivo";
      return;
     }
   if(g_quando > 0 && quando < g_quando)
     {
      g_erro_arquivo = "arquivo mais antigo que o atual";
      return;
     }
   ArrayResize(g_sinais, total);
   for(int i = 0; i < total; i++)
      g_sinais[i] = novos[i];
   g_total = total;
   g_data = data;
   g_quando = quando;
   g_conteudo = conteudo;
   g_erro_arquivo = "";
   Print("RD DEMO: ", total, " sinais de ", data, " carregados");
  }

//+------------------------------------------------------------------+
int IdadeSinal()
  {
   return g_quando <= 0 ? 9999 : (int)((TimeLocal() - g_quando) / 86400);
  }

bool Pausado()
  {
   return FileIsExist(InpArquivoParar, FILE_COMMON) || FileIsExist(InpArquivoParar + ".txt", FILE_COMMON);
  }

//+------------------------------------------------------------------+
//| O que precisa estar certo para qualquer ordem                    |
//+------------------------------------------------------------------+
bool PodeNegociar(string &motivo)
  {
   if(!InpOperarDemo)
      motivo = "desligadas nos parametros";
   else if(!ContaDemo())
      motivo = "conta nao confirmada como demo";
   else if(!Hedging())
      motivo = "somente conta hedging";
   else if(AccountInfoString(ACCOUNT_CURRENCY) != "USD")
      motivo = "conta nao esta em USD";
   else if(!Sincronizado())
      motivo = "aguardando sincronizacao";
   else if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED))
      motivo = "botao Algo Trading desligado";
   else if(!MQLInfoInteger(MQL_TRADE_ALLOWED))
      motivo = "algo trading negado no robo";
   else if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED) || !AccountInfoInteger(ACCOUNT_TRADE_EXPERT))
      motivo = "conta nao permite robos";
   else if(Pausado())
      motivo = "arquivo PARAR presente";
   else
     {
      motivo = "";
      return true;
     }
   return false;
  }

bool SinaisValidos(string &motivo)
  {
   if(g_erro_arquivo != "")
      motivo = "problema no arquivo de sinais";
   else if(g_total == 0)
      motivo = "sem sinais";
   else if(IdadeSinal() > InpMaxIdadeSinal)
      motivo = "sinais vencidos";
   else
     {
      motivo = "";
      return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
bool SessaoAberta(const string nome)
  {
   MqlDateTime agora;
   TimeToStruct(TimeTradeServer(), agora);
   long segundos = agora.hour * 3600 + agora.min * 60 + agora.sec;
   for(uint i = 0; i < 16; i++)
     {
      datetime de, ate;
      if(!SymbolInfoSessionTrade(nome, (ENUM_DAY_OF_WEEK)agora.day_of_week, i, de, ate))
         break;
      if(segundos >= (long)de && segundos < (long)ate)
         return true;
     }
   return false;
  }

// preco recente e sessao de negociacao aberta
bool Cotacao(const string nome, MqlTick &tick)
  {
   if(!SymbolInfoTick(nome, tick) || tick.ask <= 0 || tick.bid <= 0)
      return false;
   return TimeTradeServer() - tick.time <= 600 && SessaoAberta(nome);
  }

//+------------------------------------------------------------------+
int MinhasPosicoes()
  {
   int n = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetTicket(i) > 0 && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         n++;
   return n;
  }

double ValorExposto()
  {
   double soma = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetTicket(i) > 0 && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         soma += PositionGetDouble(POSITION_VOLUME) * PositionGetDouble(POSITION_PRICE_CURRENT) *
                 SymbolInfoDouble(PositionGetString(POSITION_SYMBOL), SYMBOL_TRADE_CONTRACT_SIZE);
   return soma;
  }

// minha posicao no ativo (retorno) ou de outro (outro)
bool TemPosicao(const string nome, bool &outro, double &resultado)
  {
   bool minha = false;
   outro = false;
   resultado = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      if(PositionGetTicket(i) == 0 || PositionGetString(POSITION_SYMBOL) != nome)
         continue;
      if(PositionGetInteger(POSITION_MAGIC) == InpMagic)
        {
         minha = true;
         resultado += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
        }
      else
         outro = true;
     }
   return minha;
  }

//+------------------------------------------------------------------+
void Diario(const string acao, const string ativo, const double volume, const double preco, const uint codigo,
            const string detalhe)
  {
   string linha = StringFormat("%s,%s,%s,%s,%.2f,%.5f,%u,%s", TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS),
                               g_data, acao, ativo, volume, preco, codigo, detalhe);
   Print("RD DEMO ", linha);
   int arquivo = FileOpen(InpArquivoDiario, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(arquivo == INVALID_HANDLE)
     {
      Print("RD DEMO: nao consegui abrir o diario, erro ", GetLastError());
      return;
     }
   FileSeek(arquivo, 0, SEEK_END);
   if(FileTell(arquivo) == 0)
      FileWriteString(arquivo, "hora,data_sinal,acao,ativo,volume,preco,codigo,detalhe\r\n");
   FileWriteString(arquivo, linha + "\r\n");
   FileClose(arquivo);
  }

//+------------------------------------------------------------------+
string ChaveDia()
  {
   return "RD_DIA_" + TimeToString(TimeTradeServer(), TIME_DATE);
  }

string ChaveSinal(const string ticker)
  {
   return "RD_S_" + g_data + "_" + ticker;
  }

// compras de hoje: o maior entre o contador e o historico da conta
int ComprasHoje()
  {
   int contador = GlobalVariableCheck(ChaveDia()) ? (int)GlobalVariableGet(ChaveDia()) : 0;
   datetime inicio = StringToTime(TimeToString(TimeTradeServer(), TIME_DATE));
   int historico = 0;
   if(HistorySelect(inicio, TimeTradeServer() + 86400))
      for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
        {
         ulong negocio = HistoryDealGetTicket(i);
         if(negocio > 0 && HistoryDealGetInteger(negocio, DEAL_MAGIC) == InpMagic &&
            HistoryDealGetInteger(negocio, DEAL_ENTRY) == DEAL_ENTRY_IN)
            historico++;
        }
   return MathMax(contador, historico);
  }

// ja comprado ou vendido com estes sinais (inclusive em outro reinicio ou computador)
bool JaUsado(const string ticker, const string nome)
  {
   if(GlobalVariableCheck(ChaveSinal(ticker)))
      return true;
   if(!HistorySelect(g_quando, TimeTradeServer() + 86400))
      return true; // sem historico, nao compra
   for(int i = HistoryDealsTotal() - 1; i >= 0; i--)
     {
      ulong negocio = HistoryDealGetTicket(i);
      if(negocio > 0 && HistoryDealGetInteger(negocio, DEAL_MAGIC) == InpMagic &&
         HistoryDealGetString(negocio, DEAL_SYMBOL) == nome)
         return true;
     }
   return false;
  }

// quem ja esta em carteira conta como usado para os sinais atuais
void MarcarUsados()
  {
   for(int j = 0; j < g_total; j++)
     {
      bool outro;
      double resultado;
      if(g_sinais[j].nome != "" && TemPosicao(g_sinais[j].nome, outro, resultado) &&
         !GlobalVariableCheck(ChaveSinal(g_sinais[j].ticker)))
         GlobalVariableSet(ChaveSinal(g_sinais[j].ticker), 1.0);
     }
  }

// no maximo uma tentativa a cada N segundos por chave
bool PodeTentar(const string chave, const int espera)
  {
   if(GlobalVariableCheck(chave) && TimeLocal() - (datetime)GlobalVariableGet(chave) < espera)
      return false;
   return GlobalVariableSet(chave, (double)TimeLocal()) > 0;
  }

//+------------------------------------------------------------------+
//| Nenhuma posicao do robo fica sem stop                            |
//+------------------------------------------------------------------+
void GarantirStops()
  {
   string motivo;
   if(!PodeNegociar(motivo))
      return;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetInteger(POSITION_MAGIC) != InpMagic || PositionGetDouble(POSITION_SL) > 0)
         continue;
      string nome = PositionGetString(POSITION_SYMBOL);
      MqlTick tick;
      if(!Cotacao(nome, tick) || !PodeTentar("RD_P_" + IntegerToString((long)ticket), 300))
         continue;
      if(!ContaDemo())
         return;
      int casas = (int)SymbolInfoInteger(nome, SYMBOL_DIGITS);
      double volume = PositionGetDouble(POSITION_VOLUME);
      double stop = NormalizeDouble(PositionGetDouble(POSITION_PRICE_OPEN) * (1.0 - InpStopPercent / 100.0), casas);
      if(stop < tick.bid && g_trade.PositionModify(ticket, stop, 0.0) &&
         g_trade.ResultRetcode() == TRADE_RETCODE_DONE)
         Diario("STOP", nome, volume, stop, g_trade.ResultRetcode(), "stop colocado");
      else
        {
         g_trade.SetTypeFillingBySymbol(nome);
         g_trade.PositionClose(ticket);
         Diario("FECHAR", nome, volume, tick.bid, g_trade.ResultRetcode(), "sem stop");
        }
     }
  }

//+------------------------------------------------------------------+
//| Fecha por prazo sempre; por sair da lista so com sinais validos  |
//+------------------------------------------------------------------+
void Fechar()
  {
   string motivo;
   if(!PodeNegociar(motivo))
      return;
   string porque;
   bool sinais = SinaisValidos(porque);
   for(int i = PositionsTotal() - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || PositionGetInteger(POSITION_MAGIC) != InpMagic)
         continue;
      string nome = PositionGetString(POSITION_SYMBOL);
      long dias = (long)(TimeTradeServer() - (datetime)PositionGetInteger(POSITION_TIME)) / 86400;
      bool venceu = dias >= InpMaxDiasPosicao;
      // a lista e comparada pelo ticker, mesmo que o ativo ainda nao tenha sido achado
      bool saiu = sinais;
      for(int j = 0; j < g_total; j++)
         if(EhCandidato(g_sinais[j].ticker, nome))
           {
            saiu = false;
            if(venceu)
               GlobalVariableSet(ChaveSinal(g_sinais[j].ticker), 1.0);
           }
      if(!venceu && !saiu)
         continue;
      MqlTick tick;
      if(!Cotacao(nome, tick) || !PodeTentar("RD_F_" + IntegerToString((long)ticket), 300))
         continue;
      if(!ContaDemo())
         return;
      double volume = PositionGetDouble(POSITION_VOLUME);
      g_trade.SetTypeFillingBySymbol(nome);
      g_trade.PositionClose(ticket);
      Diario("FECHAR", nome, volume, tick.bid, g_trade.ResultRetcode(), venceu ? "prazo" : "saiu da lista");
     }
  }

//+------------------------------------------------------------------+
//| Por que o ativo do ranking nao pode ser comprado agora; vazio se |
//| pode. O painel mostra o mesmo texto que decide a compra.         |
//+------------------------------------------------------------------+
string MotivoPular(const int j, MqlTick &tick, double &volume, double &stop, int &casas)
  {
   string nome = g_sinais[j].nome;
   if(nome == "")
      return g_sinais[j].grupo == "" ? "indisponivel" : "fora das acoes";
   bool outro;
   double resultado;
   if(TemPosicao(nome, outro, resultado))
      return "aberta";
   if(outro)
      return "posicao de outro";
   if(JaUsado(g_sinais[j].ticker, nome))
      return "ja usado";
   long modo = SymbolInfoInteger(nome, SYMBOL_TRADE_MODE);
   if(modo != SYMBOL_TRADE_MODE_FULL && modo != SYMBOL_TRADE_MODE_LONGONLY)
      return "compra bloqueada";
   if((SymbolInfoInteger(nome, SYMBOL_ORDER_MODE) & SYMBOL_ORDER_SL) == 0)
      return "sem stop"; // sem stop na ordem, nao compra
   if(!Cotacao(nome, tick))
      return "mercado fechado";
   if(GlobalVariableCheck("RD_E_" + nome) && TimeLocal() - (datetime)GlobalVariableGet("RD_E_" + nome) < 900)
      return "em espera";
   if((tick.ask - tick.bid) / tick.ask * 100.0 > InpMaxSpreadPercent)
      return "spread alto";
   volume = SymbolInfoDouble(nome, SYMBOL_VOLUME_MIN);
   casas = (int)SymbolInfoInteger(nome, SYMBOL_DIGITS);
   stop = NormalizeDouble(tick.ask * (1.0 - InpStopPercent / 100.0), casas);
   if(volume <= 0 || stop <= 0 ||
      tick.bid - stop < SymbolInfoInteger(nome, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(nome, SYMBOL_POINT))
      return "stop invalido";
   double patrimonio = AccountInfoDouble(ACCOUNT_EQUITY);
   double valor = volume * tick.ask * SymbolInfoDouble(nome, SYMBOL_TRADE_CONTRACT_SIZE);
   if(valor <= 0 || valor > patrimonio * InpMaxPosicaoPercent / 100.0)
      return "caro demais";
   if(ValorExposto() + valor > patrimonio * InpMaxExposicaoPercent / 100.0)
      return "limite de valor";
   double margem = 0.0;
   if(!OrderCalcMargin(ORDER_TYPE_BUY, nome, volume, tick.ask, margem) ||
      margem > AccountInfoDouble(ACCOUNT_MARGIN_FREE) * 0.5)
      return "sem margem";
   return "";
  }

// tres recusas seguidas param as compras ate o dia seguinte
bool Recusado()
  {
   if(g_dia_recusas != TimeToString(TimeTradeServer(), TIME_DATE))
      g_recusas = 0;
   return g_recusas >= 3;
  }

//+------------------------------------------------------------------+
//| Compra no maximo uma posicao por ciclo, na ordem do ranking      |
//+------------------------------------------------------------------+
void Abrir()
  {
   string motivo;
   if(!PodeNegociar(motivo) || !SinaisValidos(motivo) || Recusado())
      return;
   if(MinhasPosicoes() >= InpMaxPosicoes || ComprasHoje() >= InpMaxOrdensDia)
      return;
   for(int j = 0; j < g_total; j++)
     {
      MqlTick tick;
      double volume = 0.0, stop = 0.0;
      int casas = 0;
      if(MotivoPular(j, tick, volume, stop, casas) != "")
         continue;
      string nome = g_sinais[j].nome;
      if(!ContaDemo())
         return;
      // marcado antes de enviar: uma queda ou reinicio nunca envia duas vezes
      int antes = ComprasHoje();
      if(GlobalVariableSet(ChaveSinal(g_sinais[j].ticker), 1.0) == 0 || GlobalVariableSet(ChaveDia(), antes + 1) == 0)
         return;
      GlobalVariablesFlush();
      g_trade.SetTypeFillingBySymbol(nome);
      g_trade.Buy(volume, nome, 0.0, stop, 0.0, "RD " + g_data);
      uint codigo = g_trade.ResultRetcode();
      if(codigo == TRADE_RETCODE_MARKET_CLOSED || codigo == TRADE_RETCODE_CONNECTION)
        {
         // nada foi enviado: devolve a vez e espera 15 minutos
         GlobalVariableDel(ChaveSinal(g_sinais[j].ticker));
         GlobalVariableSet(ChaveDia(), antes);
         GlobalVariableSet("RD_E_" + nome, (double)TimeLocal());
        }
      g_ultimo_codigo = codigo;
      if(codigo == TRADE_RETCODE_DONE || codigo == TRADE_RETCODE_DONE_PARTIAL || codigo == TRADE_RETCODE_PLACED)
         g_recusas = 0;
      else if(codigo != TRADE_RETCODE_MARKET_CLOSED && codigo != TRADE_RETCODE_CONNECTION &&
              codigo != TRADE_RETCODE_TIMEOUT)
        {
         g_recusas++;
         g_dia_recusas = TimeToString(TimeTradeServer(), TIME_DATE);
        }
      Diario("COMPRAR", nome, volume, tick.ask, codigo, "stop " + DoubleToString(stop, casas));
      return;
     }
  }

//+------------------------------------------------------------------+
void Linha(const int i, const string texto, const color cor)
  {
   string nome = "RDPAINEL_L" + IntegerToString(i);
   if(ObjectFind(0, nome) < 0)
     {
      ObjectCreate(0, nome, OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, nome, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, nome, OBJPROP_XDISTANCE, 14);
      ObjectSetInteger(0, nome, OBJPROP_YDISTANCE, 24 + i * 16);
      ObjectSetString(0, nome, OBJPROP_FONT, "Consolas");
      ObjectSetInteger(0, nome, OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, nome, OBJPROP_SELECTABLE, false);
     }
   ObjectSetString(0, nome, OBJPROP_TEXT, texto); // ate 63 caracteres
   ObjectSetInteger(0, nome, OBJPROP_COLOR, cor);
  }

void Painel()
  {
   if(ObjectFind(0, "RDPAINEL_FUNDO") < 0)
     {
      ObjectCreate(0, "RDPAINEL_FUNDO", OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_XDISTANCE, 6);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_YDISTANCE, 18);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_XSIZE, 460);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_BGCOLOR, C'18,22,34');
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_BORDER_TYPE, BORDER_FLAT);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_COLOR, clrGoldenrod);
      ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_SELECTABLE, false);
     }
   string motivo;
   bool envia = PodeNegociar(motivo) && SinaisValidos(motivo);
   if(envia && Recusado())
     {
      envia = false;
      motivo = "3 ordens recusadas hoje";
     }
   if(envia && (MinhasPosicoes() >= InpMaxPosicoes || ComprasHoje() >= InpMaxOrdensDia))
      motivo = "limite atingido";
   int l = 0;
   int estado = EstadoConta();
   Linha(l++, "RD DEMO 1.10 | conta " + (estado == 0 ? "DEMO" : estado < 0 ? "sem conexao" : "NAO DEMO") +
         " | sinais sem vantagem", clrGoldenrod);
   Linha(l++, envia ? "ORDENS: LIGADAS" + (motivo == "" ? "" : " (" + motivo + ")") : "ORDENS PARADAS: " + motivo,
         envia ? clrLimeGreen : clrOrange);
   if(g_ultimo_codigo != 0)
      Linha(l++, "Ultima ordem: codigo " + IntegerToString(g_ultimo_codigo) + (g_ultimo_codigo ==
            TRADE_RETCODE_DONE ? " (executada)" : ""), clrSilver);
   Linha(l++, StringFormat("Sinais %s (%dd) | posicoes %d/%d | hoje %d/%d", g_data == "" ? "-" : g_data,
                           g_data == "" ? 0 : IdadeSinal(), MinhasPosicoes(), InpMaxPosicoes, ComprasHoje(),
                           InpMaxOrdensDia), clrWhite);
   if(g_erro_arquivo != "")
      Linha(l++, "Atencao: " + g_erro_arquivo, clrOrangeRed);
   Linha(l++, "Para parar: crie o arquivo Common\\Files\\" + InpArquivoParar, clrSilver);
   Linha(l++, StringFormat("%-6s %-9s %10s %-18s %9s", "ATIVO", "CORRETORA", "PRECO", "SITUACAO", "RESULTADO"),
         clrSilver);
   for(int j = 0; j < g_total; j++)
     {
      string nome = g_sinais[j].nome;
      string preco = "-", ganho = "";
      MqlTick tick;
      ZeroMemory(tick);
      double volume = 0.0, stop = 0.0;
      int casas = 0;
      string situacao = MotivoPular(j, tick, volume, stop, casas);
      color cor = nome == "" ? clrGray : clrSilver;
      if(situacao == "")
        {
         situacao = "aguardando";
         cor = clrWhite;
        }
      if(nome != "")
        {
         if(SymbolInfoTick(nome, tick) && tick.ask > 0)
            preco = DoubleToString(tick.ask, (int)SymbolInfoInteger(nome, SYMBOL_DIGITS));
         bool outro;
         double resultado;
         if(TemPosicao(nome, outro, resultado))
           {
            ganho = DoubleToString(resultado, 2);
            cor = resultado >= 0 ? clrLimeGreen : clrTomato;
           }
        }
      Linha(l++, StringFormat("%-6s %-9s %10s %-18s %9s", g_sinais[j].ticker, nome == "" ? "-" : nome, preco,
                              situacao, ganho), cor);
     }
   double total = 0.0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
      if(PositionGetTicket(i) > 0 && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         total += PositionGetDouble(POSITION_PROFIT) + PositionGetDouble(POSITION_SWAP);
   Linha(l++, "Resultado aberto do robo: " + DoubleToString(total, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY),
         total >= 0 ? clrLimeGreen : clrTomato);
   for(int i = l; i < g_linhas_painel; i++)
      ObjectDelete(0, "RDPAINEL_L" + IntegerToString(i));
   g_linhas_painel = l;
   ObjectSetInteger(0, "RDPAINEL_FUNDO", OBJPROP_YSIZE, 12 + l * 16);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   int estado = EstadoConta();
   if(estado == 1)
     {
      Alert("RD_Demo so roda em conta DEMO.");
      return INIT_FAILED;
     }
   if(estado == 0 && !Hedging())
     {
      Alert("RD_Demo so roda em conta demo do tipo hedging.");
      return INIT_FAILED;
     }
   if(InpMaxPosicoes < 1 || InpMaxPosicoes > 10 || InpMaxOrdensDia < 1 || InpMaxOrdensDia > 20 ||
      InpStopPercent < 1.0 || InpStopPercent > 50.0 || InpMaxDiasPosicao < 1 || InpMaxDiasPosicao > 60 ||
      InpMaxIdadeSinal < 1 || InpMaxIdadeSinal > 10 || InpMaxSpreadPercent <= 0.0 || InpMaxSpreadPercent > 5.0 ||
      InpMaxPosicaoPercent < 5.0 || InpMaxPosicaoPercent > 100.0 || InpMaxExposicaoPercent < 5.0 ||
      InpMaxExposicaoPercent > 100.0 || InpSegundos < 10 || InpSegundos > 60 || InpMagic <= 0)
     {
      Alert("RD_Demo: parametro fora do limite.");
      return INIT_PARAMETERS_INCORRECT;
     }
   // um unico RD_Demo neste computador: o arquivo fica aberto sem compartilhar
   g_trava = FileOpen(ARQUIVO_TRAVA, FILE_WRITE | FILE_BIN | FILE_COMMON);
   if(g_trava == INVALID_HANDLE)
     {
      Alert("Ja existe um RD_Demo rodando neste computador.");
      return INIT_FAILED;
     }
   ArrayFree(g_sinais);
   g_total = 0;
   g_data = "";
   g_quando = 0;
   g_conteudo = "";
   g_erro_arquivo = "";
   g_linhas_painel = 0;
   g_login = 0;
   g_servidor = "";
   g_conectado_desde = 0;
   g_recusas = 0;
   g_ultimo_codigo = 0;
   g_trade.SetExpertMagicNumber((ulong)InpMagic);
   g_trade.SetDeviationInPoints(50);
   EventSetTimer(InpSegundos);
   AtualizarConexao();
   CarregarSinais();
   ResolverFaltantes();
   Painel();
   Print("RD DEMO iniciado. Ordens ", InpOperarDemo ? "ligadas" : "desligadas", " nos parametros.");
   return INIT_SUCCEEDED;
  }

void OnTimer()
  {
   if(EstadoConta() == 1)
     {
      Print("RD DEMO: a conta nao e DEMO; removendo o robo.");
      ExpertRemove();
      return;
     }
   AtualizarConexao();
   CarregarSinais();
   ResolverFaltantes();
   MarcarUsados();
   GarantirStops();
   Fechar();
   Abrir();
   Painel();
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   ObjectsDeleteAll(0, "RDPAINEL_");
   if(g_trava != INVALID_HANDLE)
     {
      FileClose(g_trava);
      g_trava = INVALID_HANDLE;
     }
   ChartRedraw();
  }
//+------------------------------------------------------------------+
