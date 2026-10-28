@echo off
rem HPSEC - Visor de cromatogrames (llancador per a tecnics)
rem Doble clic per obrir la finestra. Genera un HTML a la carpeta de la seq.
cd /d "%~dp0"

where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "%~dp0hpsec_visor.pyw"
  goto :eof
)

where pyw >nul 2>nul
if %errorlevel%==0 (
  start "" pyw "%~dp0hpsec_visor.pyw"
  goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
  start "" py -3 "%~dp0hpsec_visor.pyw"
  goto :eof
)

start "" python "%~dp0hpsec_visor.pyw"
