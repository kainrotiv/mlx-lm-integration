import argparse
import logging

from .server import create_server
from .settings import DEFAULT_PORT


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OpenAI-compatible model API.")
    parser.add_argument("--host", default="127.0.0.1", help="loopback address to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = create_server(args.host, args.port)
    logging.info("Local AI API listening on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Stopping Local AI API")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

