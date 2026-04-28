"""Vast.ai orchestration: rent a GPU instance with our custom ComfyUI+Flux image,
wait for it to become ready, expose the public URL, and tear down on demand.

Usage:
    python vastai_pipeline.py up        # search offer + create instance + wait + write URL
    python vastai_pipeline.py down      # destroy the instance (irreversible, stops billing)
    python vastai_pipeline.py stop      # stop the instance (keeps disk, cheaper restart)
    python vastai_pipeline.py start     # restart a previously stopped instance
    python vastai_pipeline.py status    # show current state + url
    python vastai_pipeline.py ssh       # print the SSH command for the instance

Reads VAST_API_KEY from ai-sprites/.env. State persists in .vast_state.json.

Why Vast vs Salad:
  - Vast = stop/start preserves disk + image (cheaper for iteration sessions)
  - Vast = direct port forwarding works out of the box (no readiness probe dance)
  - Vast = offers are volatile (must search fresh on each `up`)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import vastai_sdk
from dotenv import load_dotenv

THIS_DIR = Path(__file__).resolve().parent
load_dotenv(THIS_DIR / ".env")

API_KEY = os.environ.get("VAST_API_KEY")
STATE_FILE = THIS_DIR / ".vast_state.json"

CONTAINER_IMAGE = "daigami/comfyui-sdxl:latest"
COMFY_PORT = 8188
DISK_GB = 40  # image is ~24 GB (SDXL Base+Refiner FP16 + ControlNet Union + IP-Adapter Plus + CLIP-H + LoRA) + outputs

# Hosts with confirmed Docker Hub registry connectivity issues (cannot pull our image).
# Add machine_id here when you hit `context deadline exceeded` or `EOF` on registry-1.docker.io.
BLACKLIST_MACHINES = {57777, 59295}

# Docker Hub auth — passed to Vast as `image_login` so the host's docker pull is authenticated,
# avoiding anonymous rate-limit (100 pulls / 6h / IP) which hits popular hosts.
DOCKERHUB_USER = os.environ.get("DOCKERHUB_USER", "")
DOCKERHUB_TOKEN = os.environ.get("DOCKERHUB_TOKEN", "")

# Search parameters: cheapest verified RTX 3090 with direct port + decent bandwidth.
# cuda_max_good>=13.0 is critical — our image uses CUDA 13.0.1, hosts on older
# drivers (530/545/550) cause "Error 803: unsupported display driver" on container start.
SEARCH_QUERY = (
    "gpu_name=RTX_3090 num_gpus=1 verified=true rentable=true "
    "direct_port_count>=1 inet_down>=500 disk_space>=30 cuda_max_good>=13.0 "
    + " ".join(f"machine_id!={m}" for m in sorted(BLACKLIST_MACHINES))
)


def _require_env() -> None:
    if not API_KEY:
        print("[vast] VAST_API_KEY missing in ai-sprites/.env")
        sys.exit(1)


def _client() -> vastai_sdk.VastAI:
    return vastai_sdk.VastAI(api_key=API_KEY)


def _save_state(data: dict) -> None:
    STATE_FILE.write_text(json.dumps(data, indent=2))


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _extract_url(inst: dict) -> str | None:
    """Vast exposes container ports via host_port:container_port mapping.
    The ports dict is keyed by '<container_port>/tcp' and contains a list of host bindings."""
    ports = inst.get("ports") or {}
    binding = ports.get(f"{COMFY_PORT}/tcp")
    if not binding:
        return None
    host_port = binding[0].get("HostPort") if isinstance(binding, list) and binding else None
    public_ip = inst.get("public_ipaddr")
    if not (public_ip and host_port):
        return None
    return f"http://{public_ip}:{host_port}"


# Strings appearing in `status_msg` that mean "this host can't pull the image, give up now".
PULL_FAILURE_PATTERNS = (
    "context deadline exceeded",
    "request canceled",
    "EOF",
    "registry-1.docker.io",
    "Error response from daemon",
    "no space left on device",
    "manifest unknown",
    "unauthorized",
)

# Strings indicating CUDA/driver problem on the host (image pulled but won't run).
RUNTIME_FAILURE_PATTERNS = (
    "Error 803",
    "unsupported display driver",
    "libcudart",
)


def _try_one_offer(sdk, offer: dict, attempt: int, total: int, per_attempt_seconds: int) -> tuple[int | None, str | None]:
    """Create an instance on this offer, poll briefly with fail-fast.
    Returns (instance_id, url) on success, (instance_id, None) if launched but never reached running
    (caller should destroy), or (None, None) if create call itself failed."""
    offer_id = offer["id"]
    machine_id = offer.get("machine_id", "?")
    print(
        f"[vast] [{attempt}/{total}] try offer={offer_id} machine={machine_id} "
        f"${offer['dph_total']:.3f}/h inet={offer.get('inet_down', 0):.0f}Mb geo={offer.get('geolocation', '?')}"
    )
    # No onstart_cmd: image now bakes LoRA umempart, flux_union_pro ControlNet,
    # ip-adapter.bin, siglip vision encoder, and the patched IPAdapter custom node.
    # (Previous attempt with onstart returned 400 Bad Request from Vast API —
    # don't reintroduce it without solid testing.)
    create_kwargs = dict(
        id=offer_id,
        image=CONTAINER_IMAGE,
        disk=DISK_GB,
        env={f"-p {COMFY_PORT}:{COMFY_PORT}": "1"},
        label="comfyui-flux",
        runtype="args",  # don't let Vast wrap our CMD with their SSH server
        cancel_unavail=True,
    )
    if DOCKERHUB_USER and DOCKERHUB_TOKEN:
        create_kwargs["login"] = f"{DOCKERHUB_USER} {DOCKERHUB_TOKEN}"
    try:
        result = sdk.create_instance(**create_kwargs)
    except Exception as exc:
        print(f"[vast]   create call failed: {exc}")
        return None, None
    if not result.get("success"):
        print(f"[vast]   create rejected: {result}")
        return None, None
    instance_id = result["new_contract"]
    print(f"[vast]   created instance_id={instance_id}, polling (max {per_attempt_seconds}s) ...")

    deadline = time.time() + per_attempt_seconds
    last_msg = None
    while time.time() < deadline:
        try:
            inst = sdk.show_instance(id=instance_id)
        except Exception as exc:
            print(f"[vast]   poll error (retry): {exc}")
            time.sleep(8); continue
        s = inst.get("actual_status") or "?"
        msg = (inst.get("status_msg") or "").strip()
        if msg and msg != last_msg:
            print(f"[vast]   status={s} msg={msg[:120]}")
            last_msg = msg
        # Fail fast on known pull failures
        if any(p in msg for p in PULL_FAILURE_PATTERNS):
            print(f"[vast]   FAIL: pull failure detected -> next offer")
            return instance_id, None
        # Fail fast on runtime failures
        if any(p in msg for p in RUNTIME_FAILURE_PATTERNS):
            print(f"[vast]   FAIL: runtime/driver failure -> next offer")
            return instance_id, None
        if s == "running":
            url = _extract_url(inst)
            if url:
                print(f"[vast]   running with URL {url}")
                return instance_id, url
        if s in ("exited", "offline", "unknown"):
            print(f"[vast]   FAIL: terminal state {s} -> next offer")
            return instance_id, None
        time.sleep(10)
    print(f"[vast]   FAIL: per-attempt timeout {per_attempt_seconds}s -> next offer")
    return instance_id, None


def cmd_up(max_attempts: int = 5, per_attempt_seconds: int = 1200) -> None:
    """Try multiple offers in series until one boots ComfyUI within per_attempt_seconds.
    Aggressively destroys failed instances so we never leave billed resources behind."""
    _require_env()
    sdk = _client()

    print(f"[vast] searching offers: {SEARCH_QUERY}")
    offers = sdk.search_offers(query=SEARCH_QUERY, order="dph_total", limit=max(20, max_attempts * 4))
    if not offers:
        print("[vast] no offers found — relax filters or try again later")
        sys.exit(2)
    print(f"[vast] {len(offers)} offers found, will try up to {max_attempts}")

    if DOCKERHUB_USER and DOCKERHUB_TOKEN:
        print(f"[vast] Docker Hub auth ON as user '{DOCKERHUB_USER}'")

    tried_machines: set[int] = set()
    final_instance_id: int | None = None
    final_url: str | None = None
    final_offer_id: int | None = None

    for attempt, offer in enumerate(offers, 1):
        if attempt > max_attempts:
            break
        machine_id = offer.get("machine_id")
        if machine_id in BLACKLIST_MACHINES:
            print(f"[vast] [{attempt}/{max_attempts}] skip offer {offer['id']} (machine {machine_id} blacklisted)")
            continue
        if machine_id in tried_machines:
            print(f"[vast] [{attempt}/{max_attempts}] skip offer {offer['id']} (machine {machine_id} already tried)")
            continue
        tried_machines.add(machine_id)

        inst_id, url = _try_one_offer(sdk, offer, attempt, max_attempts, per_attempt_seconds)
        if url:
            final_instance_id = inst_id
            final_url = url
            final_offer_id = offer["id"]
            break
        if inst_id is not None:
            try:
                sdk.destroy_instance(id=inst_id)
                print(f"[vast]   destroyed failed instance {inst_id}")
            except Exception as exc:
                print(f"[vast]   destroy after fail err: {exc}")

    if final_url is None or final_instance_id is None:
        print(f"\n[vast] [FAIL] all {max_attempts} attempts exhausted, no host could boot ComfyUI")
        sys.exit(5)

    _save_state({
        "instance_id": final_instance_id,
        "offer_id": final_offer_id,
        "url": final_url,
        "status": "running",
    })
    print(f"\n[vast] [OK] ready: {final_url}")
    print(f"[vast] Now run: python gen_sprite.py <prompt> --remote {final_url} --flux ...")
    print("[vast] When done iterating: python vastai_pipeline.py stop  (preserves disk)")
    print("[vast] When fully done   : python vastai_pipeline.py down  (destroy, irreversible)")


def cmd_down() -> None:
    _require_env()
    sdk = _client()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        print("[vast] no instance in local state")
        return
    print(f"[vast] destroying instance id={instance_id}...")
    try:
        sdk.destroy_instance(id=instance_id)
        print("[vast] [OK] destroyed, billing stopped")
    except Exception as exc:
        print(f"[vast] destroy failed (might already be gone): {exc}")
    if STATE_FILE.exists():
        STATE_FILE.unlink()


def cmd_stop() -> None:
    _require_env()
    sdk = _client()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        print("[vast] no instance in local state")
        return
    print(f"[vast] stopping instance id={instance_id} (disk preserved, GPU charges stop)...")
    sdk.stop_instance(id=instance_id)
    state["status"] = "stopped"
    _save_state(state)
    print("[vast] [OK] stopped — `python vastai_pipeline.py start` to resume")


def cmd_start() -> None:
    _require_env()
    sdk = _client()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        print("[vast] no instance in local state")
        return
    print(f"[vast] starting instance id={instance_id}...")
    sdk.start_instance(id=instance_id)
    # Re-poll for url
    print("[vast] waiting for running state...")
    deadline = time.time() + 300
    last_status = None
    while time.time() < deadline:
        inst = sdk.show_instance(id=instance_id)
        s = inst.get("actual_status")
        if s != last_status:
            print(f"[vast]   status={s}")
            last_status = s
        if s == "running":
            url = _extract_url(inst)
            if url:
                state.update({"url": url, "status": "running"})
                _save_state(state)
                print(f"[vast] [OK] ready: {url}")
                return
        time.sleep(10)
    print("[vast] timed out")


def cmd_status() -> None:
    _require_env()
    sdk = _client()
    state = _load_state()
    print(f"[vast] local state file: {state}")
    instance_id = state.get("instance_id")
    if not instance_id:
        print("[vast] (no instance tracked locally)")
        return
    inst = sdk.show_instance(id=instance_id)
    print(
        f"[vast] id={instance_id} status={inst.get('actual_status')} "
        f"gpu={inst.get('gpu_name')} dph={inst.get('dph_total')}/h"
    )
    print(f"[vast] public_ip={inst.get('public_ipaddr')} ssh={inst.get('ssh_host')}:{inst.get('ssh_port')}")
    print(f"[vast] url={_extract_url(inst)}")


def cmd_ssh() -> None:
    _require_env()
    sdk = _client()
    state = _load_state()
    instance_id = state.get("instance_id")
    if not instance_id:
        print("[vast] no instance")
        return
    inst = sdk.show_instance(id=instance_id)
    host, port = inst.get("ssh_host"), inst.get("ssh_port")
    print(f"ssh -p {port} root@{host}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["up", "down", "stop", "start", "status", "ssh"])
    args = parser.parse_args()
    if args.command == "up":
        def _handler(_sig: int, _frame) -> None:  # type: ignore
            print("\n[vast] interrupted — calling down()")
            cmd_down()
            sys.exit(130)
        signal.signal(signal.SIGINT, _handler)
        cmd_up()
    elif args.command == "down":
        cmd_down()
    elif args.command == "stop":
        cmd_stop()
    elif args.command == "start":
        cmd_start()
    elif args.command == "status":
        cmd_status()
    elif args.command == "ssh":
        cmd_ssh()


if __name__ == "__main__":
    main()
