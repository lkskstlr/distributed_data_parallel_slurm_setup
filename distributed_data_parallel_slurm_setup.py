import socket
import argparse

parser = argparse.ArgumentParser(description="Setup Pytorch Distributed Data Parallel (DDP) on a cluster managed by SLURM.")
parser.add_argument("--filename", default="test.json", type=str, help='')


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    s=socket.socket();
    s.bind(("", 0));
    print(s.getsockname()[1]);
    s.close()