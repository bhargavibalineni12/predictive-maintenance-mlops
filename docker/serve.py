import subprocess
import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        subprocess.run(
            [
                "uvicorn",
                "app.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "uvicorn",
                "app.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8080",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()