@echo off
echo Starting Zonos Streaming Service for HerikaServer...
echo.
echo Copying streaming service to WSL...
copy /y ..\HerikaServer-Streaming-Test\zonos_streaming_service.py \\wsl.localhost\DwemerAI4Skyrim3\home\dwemer\Zonos

echo Starting streaming service...
copy /y start_zonos_streaming \\wsl.localhost\DwemerAI4Skyrim3\home\dwemer\Zonos
wsl -d DwemerAI4Skyrim3 -- chown dwemer:dwemer /home/dwemer/Zonos/start_zonos_streaming
wsl -d DwemerAI4Skyrim3 -- chmod +x /home/dwemer/Zonos/start_zonos_streaming
wsl -d DwemerAI4Skyrim3 -u dwemer -- /home/dwemer/Zonos/start_zonos_streaming