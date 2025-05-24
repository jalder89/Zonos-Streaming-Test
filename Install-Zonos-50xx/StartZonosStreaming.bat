@echo off
echo Starting Zonos Streaming Service for HerikaServer...
echo.
echo This will start Zonos in streaming mode for HerikaServer integration.
echo The service will run on port 8765 and stay open until you close it.
echo.

echo Copying startup script to WSL...
copy /y start_zonos_streaming \\wsl.localhost\DwemerAI4Skyrim3\home\dwemer\Zonos
if errorlevel 1 (
    echo Error: Could not copy startup script to WSL
    echo Please ensure the DwemerAI4Skyrim3 WSL distribution is running
    pause
    exit /b 1
)

echo Setting permissions...
wsl -d DwemerAI4Skyrim3 -- sed -i 's/\r$//' /home/dwemer/Zonos/start_zonos_streaming
wsl -d DwemerAI4Skyrim3 -- chown dwemer:dwemer /home/dwemer/Zonos/start_zonos_streaming
wsl -d DwemerAI4Skyrim3 -- chmod +x /home/dwemer/Zonos/start_zonos_streaming

echo.
echo Starting Zonos Streaming Service...
echo Service will be available at: http://localhost:8765
echo.
wsl -d DwemerAI4Skyrim3 -u dwemer -- /home/dwemer/Zonos/start_zonos_streaming

echo.
echo Service stopped. Press any key to exit.
pause