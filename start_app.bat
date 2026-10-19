@echo off
rem Fallback to TranslateVideo.exe: starts the app from the project-local venv.
cd /d "%~dp0"
start "" ".venv\Scripts\pythonw.exe" -m translate_video.app
