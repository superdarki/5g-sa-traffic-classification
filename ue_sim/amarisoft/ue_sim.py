import os
import sys
from typing import Any, Dict, Optional
import yaml
import time
import signal
import asyncio
import subprocess
from datetime import datetime
import websockets
import json
from pathlib import Path
import traceback
from tqdm import tqdm

#############################################
# Config
#############################################
AMARI_HOST = os.environ.get("AMARI_HOST", "localhost")
AMARI_PORT = int(os.environ.get("AMARI_PORT", "9002"))  # Remote API port
LOGS_ROOT = Path(__file__).resolve().parent / "logs"


#############################################
# Helper functions
#############################################
def timestamp():
    return datetime.now().isoformat(timespec="seconds")


def ensure_netns(ue_id: int):
    """Check if ip netns ueX exists."""
    try:
        ns_list = subprocess.check_output(["ip", "netns", "list"], text=True)
        return f"ue{ue_id}" in [line.split()[0] for line in ns_list.splitlines()]
    except Exception as e:
        print(f"[ERROR] Failed to list netns: {e}")
        return False


def run_single(ue_id: int, command: str, outfile: Path):
    """Run a single command inside netns, return process handle."""
    f = open(outfile, "w")
    print(f"[INFO] {timestamp()} launching single ue{ue_id}: {command}")
    return subprocess.Popen(
        ["ip", "netns", "exec", f"ue{ue_id}", "bash", "-c", command],
        stdout=f,
        stderr=subprocess.STDOUT,
    )


def run_periodic(ue_id: int, command: str, interval: int, outfile: Path):
    """Run a periodic command inside netns, looping until SIGTERM."""
    loop_script = f"""
trap 'exit 0' SIGTERM
while true; do
  ip netns exec ue{ue_id} bash -c "{command}"
  sleep {interval}
done
"""
    f = open(outfile, "w")
    print(
        f"[INFO] {timestamp()} launching periodic ue{ue_id} every {interval}s: {command}"
    )
    return subprocess.Popen(
        ["bash", "-c", loop_script], stdout=f, stderr=subprocess.STDOUT
    )


async def stream_amarisoft_logs(dest_dir: Path, stop_event: asyncio.Event, mode: str):
    """Stream Amarisoft 'log_get' replies and write a plain-text log similar to ue.log.

    Behavior:
    - Sends periodic {"message":"log_get", ...} requests (min/max within server limits).
    - Deduplicates entries using (src, layer, idx).
    - Writes a small header once per run and plain text lines for each log entry.
    """
    uri = f"ws://{AMARI_HOST}:{AMARI_PORT}/"
    headers = [
        ("Origin", f"http://{AMARI_HOST}"),
        ("User-Agent", "Mozilla/5.0"),
    ]
    out_file = dest_dir / "ue_sim_logs.txt"  # plain text output (like ue.log)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Request parameters (server limits: max <= 4096)
    REQ_MIN = 64
    REQ_MAX = int(os.environ.get("AMARI_LOG_MAX", "2048"))
    layers = {
        "PHY": "debug",
        "MAC": "debug",
        "RLC": "error",
        "PDCP": "error",
        "RRC": "error",
        "NAS": "error",
        "IP": "none",
        "S72": "error",
        "GTPU": "error",
        "IKEV2": "error",
        "SWU": "error",
        "NWU": "error",
        "IPSEC": "error",
        "COM": "error",
        "TRX": "error",
        "PROD": "error",
    }

    written_header = False
    # track last seen idx per (src, layer) to avoid duplicates
    last_idx = {}

    message_id = 1
    get_interval = 0.5  # seconds

    while not stop_event.is_set():
        try:
            print(f"[DEBUG] Stream connecting to {uri} mode={mode}")
            async with websockets.connect(uri, extra_headers=headers) as ws:
                # initial reset/cut if requested
                if mode in ("reset", "cut"):
                    reset_msg: Dict[str, Any] = {"message": "log_reset"}
                    if mode == "cut":
                        reset_msg["cut"] = True
                    try:
                        await ws.send(json.dumps(reset_msg))
                    except Exception as e:
                        print(f"[WARN] Failed to send reset message: {e}")

                # small header writer helper
                def write_header(payload: Optional[dict[str, Any]] = None):
                    nonlocal written_header
                    if written_header:
                        return
                    with open(out_file, "a", encoding="utf-8") as f:
                        # ts_now = datetime.now().strftime("%a %b %d %Y %H:%M:%S %Z")
                        f.write(
                            f"# Logs streamed on {datetime.now().isoformat()} from {AMARI_HOST}:{AMARI_PORT}\n"
                        )
                        if payload is not None:
                            # include some metadata if available
                            v = payload.get("version")
                            src = payload.get("type") or payload.get("src")
                            if v or src:
                                f.write(
                                    f"# Source: {src or 'unknown'} version:{v or 'unknown'}\n"
                                )
                    written_header = True

                next_get = 0.0
                while not stop_event.is_set():
                    now = time.time()
                    if now >= next_get:
                        next_get = now + get_interval
                        req: Dict[str, Any] = {
                            "message": "log_get",
                            "timeout": 1,
                            "min": REQ_MIN,
                            "max": REQ_MAX,
                            "layers": layers,
                            "headers": False,
                            "message_id": message_id,
                        }
                        message_id += 1
                        try:
                            await ws.send(json.dumps(req))
                        except Exception as e:
                            print(f"[WARN] Failed to send log_get: {e}")
                            break

                        try:
                            resp = await asyncio.wait_for(ws.recv(), timeout=3.0)
                        except asyncio.TimeoutError:
                            await asyncio.sleep(0.01)
                            continue
                        except websockets.ConnectionClosed:
                            print("[WARN] WebSocket closed by server, reconnecting...")
                            break
                        except Exception as e:
                            print(f"[WARN] Error receiving get reply: {e}")
                            break

                        # parse response as JSON (server returns structured object with 'logs' list)
                        parsed = None
                        try:
                            parsed = json.loads(resp) if isinstance(resp, str) else resp
                        except Exception:
                            # not JSON - write raw once and continue
                            write_header()
                            with open(out_file, "a", encoding="utf-8") as f:
                                f.write(
                                    str(resp)
                                    + ("\n" if not str(resp).endswith("\n") else "")
                                )
                            continue

                        # if server announces ready or version info, write header
                        if (
                            isinstance(parsed, dict)
                            and parsed.get("message") == "ready"
                        ):
                            write_header(parsed.get("payload") or parsed)
                            # continue to next poll

                        # process logs array
                        logs = parsed.get("logs") if isinstance(parsed, dict) else None
                        if logs and isinstance(logs, list):
                            write_header(parsed)
                            with open(out_file, "a", encoding="utf-8") as f:
                                for ent in logs:
                                    try:
                                        src = ent.get("src", "")
                                        layer = (ent.get("layer") or "").upper()
                                        idx = int(ent.get("idx", 0))
                                        key = (src, layer)
                                        if key in last_idx and idx <= last_idx[key]:
                                            continue
                                        last_idx[key] = idx

                                        # timestamp: prefer ms epoch 'timestamp', then top-level 'time'
                                        ts_ms = ent.get("timestamp")
                                        if ts_ms:
                                            try:
                                                dt = datetime.fromtimestamp(
                                                    ts_ms / 1000.0
                                                )
                                            except Exception:
                                                dt = datetime.now()
                                        else:
                                            assert isinstance(parsed, dict)
                                            tt = parsed.get("time") or ent.get("time")
                                            if tt:
                                                try:
                                                    dt = datetime.fromtimestamp(
                                                        float(tt)
                                                    )
                                                except Exception:
                                                    dt = datetime.now()
                                            else:
                                                dt = datetime.now()
                                        time_str = dt.strftime("%H:%M:%S.%f")[:-3]

                                        # direction normalization
                                        raw_dir = (ent.get("dir") or "").strip()
                                        if raw_dir == "" or raw_dir == "-":
                                            direction = "-"
                                        elif raw_dir.upper().startswith("DL"):
                                            direction = "DL"
                                        elif raw_dir.upper().startswith("UL"):
                                            direction = "UL"
                                        else:
                                            direction = raw_dir

                                        # helper to try multiple keys
                                        def pick(e, keys):
                                            for k in keys:
                                                v = e.get(k)
                                                if v is not None and v != "":
                                                    return v
                                            return None

                                        # ue id (format 4 digits when numeric)
                                        ue_val = pick(
                                            ent, ("ue_id", "ue", "ueid", "ue_id_str")
                                        )
                                        ue_str = "----"
                                        if ue_val is not None:
                                            try:
                                                ue_num = int(str(ue_val), 0)
                                                ue_str = f"{ue_num:04d}"
                                            except Exception:
                                                ue_str = str(ue_val)

                                        # cell id and rnti (best-effort)
                                        cell_val = pick(
                                            ent, ("cell", "cell_id", "cellId")
                                        )
                                        try:
                                            cell_str = (
                                                f"{int(cell_val):02d}"
                                                if cell_val is not None
                                                else "00"
                                            )
                                        except Exception:
                                            cell_str = (
                                                str(cell_val)
                                                if cell_val is not None
                                                else "00"
                                            )

                                        rnti_val = pick(
                                            ent, ("rnti", "rnti_str", "rnti_hex")
                                        )
                                        rnti_str = (
                                            str(rnti_val)
                                            if rnti_val is not None
                                            else "----"
                                        )

                                        # channel/sfn: prefer info.{channel,sfn} for PHY, fall back to top-level
                                        info = ent.get("info") or {}
                                        chan = pick(
                                            info, ("channel", "sfn", "sfn_channel")
                                        ) or pick(ent, ("channel", "sfn"))
                                        chan_str = str(chan) if chan is not None else ""

                                        # message body: join data list or common fields
                                        data = ent.get("data")
                                        if isinstance(data, list):
                                            body = " ".join([str(x) for x in data])
                                        else:
                                            body = pick(
                                                ent, ("message", "msg", "text")
                                            ) or json.dumps(ent, default=str)

                                        # Build ue.log-like line:
                                        # "HH:MM:SS.mmm [LAYER] DIR UEID CELL RNTI   CHANNEL message"
                                        channel_part = f"{cell_str} {rnti_str}"
                                        if chan_str:
                                            channel_part = (
                                                f"{channel_part}   {chan_str}"
                                            )
                                        line = f"{time_str} [{layer}] {direction} {ue_str} {channel_part} {body}"
                                        if not line.endswith("\n"):
                                            line += "\n"
                                        f.write(line)
                                    except Exception as e:
                                        # on any per-entry error, write a debug line
                                        f.write(
                                            f"# ERROR formatting log entry: {e} raw:{json.dumps(ent, default=str)}\n"
                                        )
                        # continue polling loop
                        continue

                    await asyncio.sleep(0.01)

        except Exception as e:
            print(f"[WARN] Stream connection failed: {e}")
            print(traceback.format_exc())

        await asyncio.sleep(1.0)


#############################################
# Main runner
#############################################
async def run_tests(config_file: str):
    with open(config_file) as f:
        config = yaml.safe_load(f)

    tests = config.get("tests", [])
    if not tests:
        print("No tests found in YAML")
        return

    LOGS_ROOT.mkdir(exist_ok=True)

    # Ask user whether to reset/cut/skip Amarisoft logs before streaming
    choice = None
    try:
        choice = (
            input("Amarisoft logs: (r)eset / (c)ut / (n)one [r/c/n]: ").strip().lower()
        )
    except Exception:
        choice = "r"
    if choice.startswith("r"):
        mode = "reset"
    elif choice.startswith("c"):
        mode = "cut"
    else:
        mode = "none"

    for test in tests:
        name = test.get("name", "unnamed")
        duration = int(test.get("duration", 0))
        if duration <= 0:
            print(f"[ERROR] Test '{name}' missing valid duration")
            continue

        ts_start = datetime.now().strftime("%Y%m%d-%H%M%S")
        per_test_dir = LOGS_ROOT / f"{ts_start}_{name}"
        per_test_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"[TEST] {name} (duration {duration}s)")
        print("-" * 60)

        # Start streaming logs in background while test runs
        stop_event = asyncio.Event()
        stream_task = asyncio.create_task(
            stream_amarisoft_logs(per_test_dir, stop_event, mode)
        )

        host_process = None
        host_cmd = test.get("host_command")
        host_type = test.get("host_type")
        if host_cmd and host_type:
            host_log = per_test_dir / "host.log"
            print(f"[INFO] Launching host command: {host_cmd}")
            if host_type == "single":
                host_process = subprocess.Popen(
                    host_cmd,
                    shell=True,
                    stdout=open(host_log, "w"),
                    stderr=subprocess.STDOUT,
                )
            elif host_type == "periodic":
                print(
                    f"[WARN] 'periodic' host_type not implemented, skipping host command."
                )
            else:
                print(f"[WARN] Unknown host_type '{host_type}', skipping host command.")

        await asyncio.sleep(1.0)  # give stream a moment to connect to websocket

        processes = []
        for ue in test.get("ues", []):
            ue_id = ue.get("ue_id")
            cmd = ue.get("command")
            ctype = ue.get("type")
            interval = ue.get("interval", None)

            if not ue_id or not cmd or not ctype:
                print(f"[ERROR] Invalid UE entry in test '{name}' → {ue}")
                continue
            if not ensure_netns(ue_id):
                print(f"[ERROR] Missing netns ue{ue_id}, skipping.")
                continue

            ue_log = per_test_dir / f"ue{ue_id}.log"

            if ctype == "single":
                p = run_single(ue_id, cmd, ue_log)
            elif ctype == "periodic":
                if not interval:
                    print(f"[ERROR] interval required for periodic ue{ue_id}")
                    continue
                p = run_periodic(ue_id, cmd, interval, ue_log)
            else:
                print(f"[WARN] Unknown type {ctype} for ue{ue_id}, skipping")
                continue

            processes.append(p)

        # Print timestamp after launching all UEs
        print(f"[INFO] All UEs launched at: {timestamp()}")

        # Sleep for test duration with progress bar
        print(f"[INFO] Test running for {duration}s...")
        for _ in tqdm(range(duration), desc="Test Progress", unit=""):
            await asyncio.sleep(1)
        await asyncio.sleep(1.0)  # small buffer after progress

        # Stop processes
        print(f"[INFO] Stopping UEs for test '{name}'")
        for p in processes:
            try:
                p.send_signal(signal.SIGTERM)
            except Exception:
                pass
        t_end = time.time() + 5
        for p in processes:
            try:
                while p.poll() is None and time.time() < t_end:
                    time.sleep(0.2)
                if p.poll() is None:
                    print(f"[WARN] Forcing kill PID {p.pid}")
                    p.kill()
            except Exception:
                pass

        # Stop host process if started
        if host_process:
            print(f"[INFO] Stopping host command for test '{name}'")
            try:
                host_process.send_signal(signal.SIGTERM)
            except Exception:
                pass
            t_end = time.time() + 5
            while host_process.poll() is None and time.time() < t_end:
                time.sleep(0.2)
            if host_process.poll() is None:
                print(f"[WARN] Forcing kill host PID {host_process.pid}")
                host_process.kill()

        # Stop streaming and wait for task to finish
        stop_event.set()
        try:
            await asyncio.wait_for(stream_task, timeout=10.0)
        except asyncio.TimeoutError:
            print("[WARN] Stream task did not finish promptly, continuing")

        print(f"[DONE] {name} → logs saved in {per_test_dir}")
        print("=" * 60)


#############################################
# Entrypoint
#############################################
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)
    asyncio.run(run_tests(sys.argv[1]))
