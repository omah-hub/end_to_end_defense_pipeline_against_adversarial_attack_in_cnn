import threading
import uvicorn

def run():
    uvicorn.run("deployment.api_server:app",
                host="0.0.0.0",
                port=8000,
                reload=False)

thread = threading.Thread(target=run)
thread.start()