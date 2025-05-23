@echo off
echo Downloading apt packages...
wsl -d DwemerAI4Skyrim3 -- rm -rf /home/dwemer/Zonos
wsl -d DwemerAI4Skyrim3 -- apt-get update
wsl -d DwemerAI4Skyrim3 -- apt install espeak-ng

echo Running Zonos installer...
copy /y zonos_install \\wsl.localhost\DwemerAI4Skyrim3\usr\local\bin
wsl -d DwemerAI4Skyrim3 -- chmod 557 /usr/local/bin/zonos_install
wsl -d DwemerAI4Skyrim3 -u dwemer -- /usr/local/bin/zonos_install

echo Downloading models...
copy /y zonos_download_models \\wsl.localhost\DwemerAI4Skyrim3\usr\local\bin
wsl -d DwemerAI4Skyrim3 -- chmod 557 /usr/local/bin/zonos_download_models
copy /y download_models.py \\wsl.localhost\DwemerAI4Skyrim3\home\dwemer\Zonos
wsl -d DwemerAI4Skyrim3 -- chown dwemer:dwemer /home/dwemer/Zonos/download_models.py
wsl -d DwemerAI4Skyrim3 -u dwemer -- /usr/local/bin/zonos_download_models

echo Copying start_zonos file...
copy start_zonos \\wsl.localhost\DwemerAI4Skyrim3\home\dwemer\Zonos
wsl -d DwemerAI4Skyrim3 -- chown dwemer:dwemer /home/dwemer/Zonos/start_zonos
wsl -d DwemerAI4Skyrim3 -- chmod +x /home/dwemer/Zonos/start_zonos

echo Install finished. Press any key to exit.
pause