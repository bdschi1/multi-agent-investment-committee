#!/bin/bash
# Enable Ollama parallel inference (3 concurrent requests)
# Restart Ollama app after setting env vars via launchctl

echo "Setting OLLAMA_NUM_PARALLEL=3..."
launchctl setenv OLLAMA_NUM_PARALLEL 3

echo "Setting OLLAMA_KEEP_ALIVE=-1 (keep model loaded)..."
launchctl setenv OLLAMA_KEEP_ALIVE -1

echo "Restarting Ollama app..."
pkill -f "Ollama.app" 2>/dev/null
sleep 2
open -a Ollama
sleep 3

echo "Verifying..."
PARALLEL=$(launchctl getenv OLLAMA_NUM_PARALLEL)
KEEPALIVE=$(launchctl getenv OLLAMA_KEEP_ALIVE)
echo "  OLLAMA_NUM_PARALLEL = $PARALLEL"
echo "  OLLAMA_KEEP_ALIVE   = $KEEPALIVE"

# Quick health check
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is up and ready."
else
    echo "Waiting for Ollama to start..."
    sleep 5
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is up and ready."
    else
        echo "Ollama may still be starting. Check: curl http://localhost:11434/api/tags"
    fi
fi
