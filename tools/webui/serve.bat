@echo off
REM Launch the Anot ASR Console on http://localhost:3000
REM (served from localhost:3000 so it matches the service's allowed CORS origin)
cd /d "%~dp0"
start "" "http://localhost:3000/"
echo Serving Anot ASR Console at http://localhost:3000/  (press Ctrl+C to stop)
python -m http.server 3000
