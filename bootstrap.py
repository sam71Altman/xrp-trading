import subprocess
import sys

# Uninstall conflicting 'websocket' package
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "websocket"])

# Install/Upgrade 'websocket-client'
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "websocket-client"])

print("Dependencies fixed")