sudo iptables -t nat -A PREROUTING -p tcp --dport 80 -j REDIRECT --to-port 8000
sudo docker run -p 6333:6333 -p 6334:6334     -v $(pwd)/qdrant_storage:/qdrant/storage -d    qdrant/qdrant
LLM_NAME='stabilityai/stablelm-zephyr-3b' fastapi run ragclass/main.py