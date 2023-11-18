@echo off

REM Nombre del entorno virtual
set "ENV_NAME=venv"
REM Ruta donde se creará el entorno virtual
set "ENV_PATH=%cd%\%ENV_NAME%"


REM Comprobar si el entorno virtual ya existe
if exist "%ENV_PATH%" (
    echo El entorno virtual %ENV_NAME% ya existe.
    call :activar
    call :iniciar_script
    goto :eof
)

echo Creando el entorno virtual %ENV_NAME%...
call :crear_venv
call :instalar_reqs
call :activar
call :iniciar_script

:crear_venv
REM Crear el entorno virtual
python -m venv "%ENV_PATH%"
call "%ENV_PATH%\Scripts\activate.bat"
goto :eof

:instalar_reqs
REM Install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
goto :eof

:activar
REM Activar el entorno virtual
echo Activando el entorno virtual %ENV_NAME%...
call "%ENV_PATH%\Scripts\activate.bat"
goto :eof

:iniciar_script
REM Iniciar script principal
echo Ejecutando script
cmd /k streamlit run interface.py
goto :eof

pause
