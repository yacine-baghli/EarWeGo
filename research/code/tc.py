"""
Thunder Compute driver — programmatic SSH/SFTP to instance 0 via paramiko.
Bypasses the deprecated interactive `tnr` CLI. Uses Thunder's own utils to
resolve token/ip/keyfile, then gives run()/put()/get() for non-interactive use.

Usage:
    python scratch/tc.py run "nvidia-smi -L"
    python scratch/tc.py put local.file /home/ubuntu/remote.file
    python scratch/tc.py get /home/ubuntu/remote.file local.file
"""
import os, sys, time
import thunder.utils as u
import paramiko

INSTANCE = "0"


def _conn():
    tok = os.environ["TNR_API_TOKEN"]
    ok, err, inst = u.get_instances(tok)
    if not ok:
        raise SystemExit(f"get_instances failed: {err}")
    m = inst[INSTANCE]
    kf = u.get_key_file(m["uuid"])
    if not os.path.exists(kf):
        if not u.add_key_to_instance(INSTANCE, tok):
            raise SystemExit("could not create/download key file")
    ip, port = m["ip"], int(m.get("port", 22))
    last = None
    for _ in range(20):
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(ip, port=port, username="ubuntu", key_filename=kf,
                      allow_agent=False, look_for_keys=False, timeout=12)
            return c
        except Exception as e:
            last = e; time.sleep(2)
    raise SystemExit(f"ssh connect failed: {last}")


def run(cmd, quiet=False):
    c = _conn()
    stdin, stdout, stderr = c.exec_command(cmd, timeout=None)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    c.close()
    if not quiet:
        if out: print(out, end="" if out.endswith("\n") else "\n")
        if err: print("[stderr]", err, end="" if err.endswith("\n") else "\n")
    return code, out, err


def _rel(remote):
    # SFTP is chrooted to /home/ubuntu -> use paths relative to home
    for pre in ("/home/ubuntu/", "/home/ubuntu"):
        if remote.startswith(pre):
            return remote[len(pre):].lstrip("/") or "."
    return remote.lstrip("/")


def put(local, remote):
    remote = _rel(remote)
    c = _conn(); sf = c.open_sftp()
    sf.listdir(".")  # warm up the chroot/path resolution (first-open can race)
    t0 = time.time()
    last = None
    for attempt in range(5):
        try:
            with open(local, "rb") as src, sf.open(remote, "wb") as dst:
                dst.set_pipelined(True)
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    dst.write(chunk)
            break
        except IOError as e:
            last = e; time.sleep(1.5)
    else:
        raise last
    sz = os.path.getsize(local) / 1e6
    sf.close(); c.close()
    print(f"put {local} -> {remote} ({sz:.1f} MB, {time.time()-t0:.1f}s)")


def get(remote, local):
    remote = _rel(remote)
    c = _conn(); sf = c.open_sftp()
    sf.get(remote, local)
    sf.close(); c.close()
    print(f"get {remote} -> {local} ({os.path.getsize(local)/1e6:.2f} MB)")


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "run":
        code, _, _ = run(sys.argv[2]); sys.exit(code)
    elif action == "put":
        put(sys.argv[2], sys.argv[3])
    elif action == "get":
        get(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown action {action}")
