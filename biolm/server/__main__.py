"""Worker entrypoint for detached ``biolm server start``."""
from biolm.server.runner import run_server_foreground
from biolm.server.settings import ServerSettings

if __name__ == "__main__":
    run_server_foreground(ServerSettings.from_env())
