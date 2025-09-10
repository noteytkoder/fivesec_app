from pathlib import Path
import subprocess, time, os
from logger import setup_logger

logger = setup_logger(log_dir=os.path.join(os.path.dirname(__file__), "logs"))
RESTART_FLAG = Path(os.path.join(os.path.dirname(__file__), "fivesec_restart.flag"))

def run():
    project_root = os.path.dirname(__file__)
    while True:
        if RESTART_FLAG.exists():
            logger.info("Removing existing fivesec_restart.flag")
            RESTART_FLAG.unlink()

        logger.info("Starting fivesec_app.main")
        process = subprocess.Popen(
            ["python", "-m", "main"],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        logger.debug(f"stdout: {stdout}")
        if stderr:
            filtered = "\n".join(
                line for line in stderr.splitlines()
                if not any(k in line for k in [
                    "dash-component-suites",
                    "GET /", "POST /", "Running on http",
                    "This is a development server"
                ])
            )
            if filtered.strip():
                logger.error(f"stderr: {filtered}")

        if RESTART_FLAG.exists():
            logger.info("Restart request detected")
            time.sleep(2)
            continue
        else:
            logger.info("Completed without restart. Exiting.")
            break

if __name__ == "__main__":
    run()
