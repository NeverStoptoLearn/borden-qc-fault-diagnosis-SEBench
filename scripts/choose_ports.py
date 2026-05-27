import argparse
import json
import socket


def free(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, int(port))) != 0


def choose(default, fallbacks):
    if free(default):
        return default
    for port in fallbacks:
        if free(port):
            return port
    raise SystemExit("No free port found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset-default", type=int, default=8000)
    p.add_argument("--judge-default", type=int, default=8080)
    args = p.parse_args()
    asset = choose(args.asset_default, [8001, 8002, 8010, 18000, 18001])
    judge = choose(args.judge_default, [8081, 8082, 8090, 18080, 18081])
    print(json.dumps({"asset_port": asset, "judge_port": judge}, indent=2))
    print(f"ASSET_PORT={asset}")
    print(f"JUDGE_PORT={judge}")


if __name__ == "__main__":
    main()
