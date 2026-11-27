//+------------------------------------------------------------------+
//| RD_Sombra.mq5                                                    |
//| Le os sinais do RD-Agent e registra o que faria.                 |
//| NAO TEM CODIGO DE ORDEM: nao compra, nao vende, nao altera nada. |
//+------------------------------------------------------------------+
#property copyright "RD-Agent-Q-Lab"
#property version   "1.00"

input string InpArquivoSinais = "RD\\rd_sinais.csv"; // Arquivo de sinais (pasta comum)
input string InpArquivoDiario = "RD\\rd_diario.csv"; // Diario (pasta comum)
input int    InpSegundos      = 60;                  // Ler o arquivo a cada N segundos

string g_ultimo = ""; // conteudo ja registrado, para nao repetir o diario

//+------------------------------------------------------------------+
//| Nome do ativo na corretora, com os sufixos mais comuns           |
//+------------------------------------------------------------------+
string AcharSimbolo(const string ticker)
  {
   string candidatos[7];
   candidatos[0] = ticker;
   candidatos[1] = ticker + ".US";
   candidatos[2] = ticker + ".us";
   candidatos[3] = ticker + ".NYSE";
   candidatos[4] = ticker + ".NAS";
   candidatos[5] = "#" + ticker;
   candidatos[6] = ticker + "-US";
   for(int i = 0; i < 7; i++)
     {
      bool personalizado = false;
      if(SymbolExist(candidatos[i], personalizado))
         return candidatos[i];
     }
   return "";
  }

//+------------------------------------------------------------------+
//| Le o arquivo de sinais e escreve o que faria                     |
//+------------------------------------------------------------------+
void Processar()
  {
   int entrada = FileOpen(InpArquivoSinais, FILE_READ | FILE_TXT | FILE_ANSI | FILE_COMMON | FILE_SHARE_READ);
   if(entrada == INVALID_HANDLE)
     {
      Comment("RD SOMBRA - sem ordens\nArquivo de sinais nao encontrado: ", InpArquivoSinais);
      return;
     }
   string linhas[];
   int total = 0;
   string conteudo = "";
   while(!FileIsEnding(entrada))
     {
      string linha = FileReadString(entrada);
      if(StringLen(linha) == 0)
         continue;
      ArrayResize(linhas, total + 1);
      linhas[total] = linha;
      total++;
      conteudo += linha + ";";
     }
   FileClose(entrada);
   if(total < 2 || conteudo == g_ultimo)
      return;

   int diario = FileOpen(InpArquivoDiario, FILE_READ | FILE_WRITE | FILE_TXT | FILE_ANSI | FILE_COMMON);
   if(diario == INVALID_HANDLE)
     {
      Print("RD SOMBRA: nao consegui abrir o diario, erro ", GetLastError());
      return;
     }
   FileSeek(diario, 0, SEEK_END);
   if(FileTell(diario) == 0)
      FileWriteString(diario, "hora,data_sinal,posicao,simbolo,na_corretora,preco_venda,spread_pontos,empatados,acao\r\n");

   int disponiveis = 0;
   string data = "";
   for(int i = 1; i < total; i++) // a linha 0 e o cabecalho
     {
      string campo[];
      if(StringSplit(linhas[i], ',', campo) < 6)
         continue;
      data = campo[0];
      string ticker = campo[3];
      string nome = AcharSimbolo(ticker);
      double preco = 0.0;
      long spread = 0;
      string acao = "ATIVO_INDISPONIVEL";
      if(nome != "")
        {
         SymbolSelect(nome, true); // so mostra o ativo na Observacao do Mercado
         preco = SymbolInfoDouble(nome, SYMBOL_ASK);
         spread = SymbolInfoInteger(nome, SYMBOL_SPREAD);
         acao = "SOMBRA_COMPRARIA";
         disponiveis++;
        }
      string registro = StringFormat("%s,%s,%s,%s,%s,%.5f,%I64d,%s,%s",
                                     TimeToString(TimeLocal(), TIME_DATE | TIME_SECONDS), data, campo[2], ticker,
                                     nome == "" ? "-" : nome, preco, spread, campo[5], acao);
      FileWriteString(diario, registro + "\r\n");
      Print("RD SOMBRA ", registro);
     }
   FileClose(diario);
   g_ultimo = conteudo;
   Comment("RD SOMBRA - nenhuma ordem e enviada\n",
           "Sinais de ", data, ": ", total - 1, " ativos, ", disponiveis, " disponiveis nesta corretora\n",
           "Diario: ", InpArquivoDiario);
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   if((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
     {
      Alert("RD SOMBRA so roda em conta DEMO.");
      return INIT_FAILED;
     }
   Print("RD SOMBRA iniciado em conta DEMO. Este robo nao tem funcao de ordem.");
   EventSetTimer(MathMax(10, InpSegundos));
   Processar();
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment("");
  }

void OnTimer()
  {
   Processar();
  }
//+------------------------------------------------------------------+
