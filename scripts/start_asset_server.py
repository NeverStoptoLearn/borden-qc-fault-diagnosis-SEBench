import argparse
import http.server
import socket
import socketserver
from pathlib import Path


def is_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex(("127.0.0.1", int(port))) != 0


def choose(default, fallbacks):
    if is_free(default):
        return default
    for port in fallbacks:
        if is_free(port):
            return port
    raise RuntimeError("no free asset port")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="borden_qc_fault_diagnosis_package")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    port = choose(args.port, [8001, 8002, 8010, 18000, 18001])
    directory = Path(args.directory).resolve()
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(directory), **kw)
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving {directory} on http://0.0.0.0:{port}")
        print("Task JSON setup_cmds try ports: 8000 8001 8002 8010 18000 18001")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
