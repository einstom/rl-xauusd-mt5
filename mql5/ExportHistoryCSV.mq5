//+------------------------------------------------------------------+
//| ExportHistoryCSV.mq5                                             |
//| One-click M1 history dump for the RL pipeline.                   |
//|                                                                  |
//| Writes MQL5\Files\XAUUSD_M1.csv in exactly the format            |
//| data_loader.load_mt_ohlcv_csv expects:                           |
//|   Time (EET),Open,High,Low,Close,Volume                          |
//| Timestamps are server time at bar OPEN (IC Markets = EET/EEST,   |
//| matching CFG.source_tz = Europe/Helsinki).                       |
//|                                                                  |
//| Usage: compile in MetaEditor (F7), open any XAUUSD chart, drag   |
//| the script onto it, set StartYear, OK.  Set Tools > Options >    |
//| Charts > "Max bars in chart" to Unlimited first so the terminal  |
//| pulls the full server history.                                   |
//+------------------------------------------------------------------+
#property version   "1.00"
#property script_show_inputs

input int    InpStartYear = 2015;             // First year to export
input string InpFileName  = "XAUUSD_M1.csv"; // Output file (in MQL5\Files)

//+------------------------------------------------------------------+
int CopyWithRetry(const string symbol, ENUM_TIMEFRAMES tf,
                  datetime from, datetime to, MqlRates &rates[])
  {
   // First requests for old history return -1 while the terminal is still
   // downloading it from the broker; retry with a pause instead of skipping.
   for(int attempt = 0; attempt < 10; attempt++)
     {
      ResetLastError();
      int n = CopyRates(symbol, tf, from, to, rates);
      if(n >= 0)
         return n;
      Sleep(1000);
     }
   return -1;
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   const string symbol = _Symbol;

   int handle = FileOpen(InpFileName, FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(handle == INVALID_HANDLE)
     {
      Print("FileOpen failed, error ", GetLastError());
      return;
     }
   FileWriteString(handle, "Time (EET),Open,High,Low,Close,Volume\r\n");

   datetime from = StringToTime(IntegerToString(InpStartYear) + ".01.01 00:00");
   datetime now  = TimeCurrent();
   long     total = 0;
   MqlRates rates[];

   // Month-sized windows: small enough to never hit array limits, aligned to
   // whole minutes so consecutive windows cannot duplicate a bar.
   for(datetime t0 = from; t0 < now && !IsStopped(); )
     {
      datetime t1 = t0 + 32 * 24 * 60 * 60;
      if(t1 > now) t1 = now;

      int n = CopyWithRetry(symbol, PERIOD_M1, t0, t1 - 60, rates);
      if(n > 0)
        {
         for(int i = 0; i < n; i++)
            FileWriteString(handle, StringFormat("%s,%s,%s,%s,%s,%d\r\n",
               TimeToString(rates[i].time, TIME_DATE|TIME_SECONDS),
               DoubleToString(rates[i].open,  _Digits),
               DoubleToString(rates[i].high,  _Digits),
               DoubleToString(rates[i].low,   _Digits),
               DoubleToString(rates[i].close, _Digits),
               (int)rates[i].tick_volume));
         total += n;
        }
      else if(n < 0)
         Print("CopyRates failed for ", TimeToString(t0, TIME_DATE), " .. ",
               TimeToString(t1, TIME_DATE), ", error ", GetLastError(),
               " (history may simply not reach this far back)");

      Comment(StringFormat("Exporting %s M1: %I64d bars, at %s",
              symbol, total, TimeToString(t0, TIME_DATE)));
      t0 = t1;
     }

   FileClose(handle);
   Comment("");
   PrintFormat("DONE: %I64d M1 bars of %s -> MQL5\\Files\\%s "
               "(File > Open Data Folder to find it)",
               total, symbol, InpFileName);
  }
//+------------------------------------------------------------------+
