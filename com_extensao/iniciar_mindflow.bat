@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [MindFlow] O ambiente Python ainda nao foi criado.
  echo Execute primeiro os comandos de instalacao do GUIA_MINDFLOW_INTEGRACAO.md.
  echo.
  pause
  exit /b 1
)

set "MINDFLOW_SUPABASE_URL=https://xeykjfytdvqtgcewvoii.supabase.co"
set "MINDFLOW_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhleWtqZnl0ZHZxdGdjZXd2b2lpIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODg4MTA2MzMsImV4cCI6MjEwNDM4NjYzM30.DHrm3C3Zl7ZnckpEv1FsngbkkUWC1a94q8577l3aLc0"
set "MINDFLOW_CONSENT_VERSION=v1"

echo.
echo [MindFlow] Iniciando Python com historico do Supabase...
echo [MindFlow] Mantenha esta janela aberta durante a reuniao.
echo.

".venv\Scripts\python.exe" -m mindflow_extractor.runners.extension_server

echo.
echo [MindFlow] O servidor foi encerrado.
pause
endlocal
