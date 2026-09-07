from __future__ import annotations
import socket
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def port_free(port:int)->bool:
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.settimeout(.15)
        return sock.connect_ex(('127.0.0.1',port)) != 0


def find_port(start=8501,end=8525):
    for port in range(start,end+1):
        if port_free(port): return port
    raise RuntimeError(f'No free Streamlit port found between {start} and {end}.')


def main():
    port=find_port()
    print('='*72)
    print('ApexML Final — Simple Pro')
    print(f'Using free port: {port}')
    print(f'Open: http://localhost:{port}')
    print('='*72)
    cmd=[sys.executable,'-m','streamlit','run',str(ROOT/'app.py'),'--server.port',str(port)]
    raise SystemExit(subprocess.call(cmd,cwd=ROOT))


if __name__=='__main__': main()
