#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Play the compiled Zork executable with an NLP interface.
Repo: https://github.com/devshane/zork
"""

import os
import re
import time

from zork_nlp_system_9 import EnhancedNLPParser

PROMPT_RES = [
    re.compile(r"^\s*>\s*$", re.M),             # ">"
    re.compile(r">\s*", re.M),                  # any '>' prompt
    re.compile(r"\bWhat (now|do you want)\??", re.I),
]

def resolve_executable() -> str:
    """Find zork(.exe) nearby, in PATH, or via ZORK_EXE."""
    from shutil import which

    candidates = [
        os.environ.get("ZORK_EXE"),
        r".\zork.exe", r".\zork",
        "./zork.exe", "./zork",
        "zork.exe", "zork",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p

    w = which("zork") or which("zork.exe")
    if w:
        return w

    raise FileNotFoundError(
        "Zork executable not found."
    )


def spawn_game(exe):
    """Cross-platform child process using pexpect where possible."""
    import pexpect
    c = pexpect.spawn(exe, encoding="utf-8", timeout=2)
    if hasattr(c, "setecho"):
        c.setecho(False)
    return c


def read_until_prompt_or_quiet(child, total_timeout=3.0, quiet_time=0.25, chunk_timeout=0.2):
    """
    Try to read until a Zork-ish prompt appears.
    """
    start = time.time()
    buf = ""

    while time.time() - start < total_timeout:
        try:
            # Attempt each prompt regex quickly
            for rx in PROMPT_RES:
                child.expect(rx, timeout=chunk_timeout)
                buf += child.before or ""
                return buf
            # If none matched (shouldn't happen due to immediate return), just continue
        except Exception:
            # No prompt yet; try nonblocking read
            try:
                chunk = child.read_nonblocking(size=4096, timeout=chunk_timeout)
                if chunk:
                    buf += chunk
                    last = time.time()
                    # keep slurping while we see output
                    while time.time() - last < quiet_time:
                        try:
                            more = child.read_nonblocking(size=4096, timeout=chunk_timeout)
                            if more:
                                buf += more
                                last = time.time()
                        except Exception:
                            break
                    return buf
            except Exception:
                pass

        # If the child died, exit
        if hasattr(child, "isalive") and not child.isalive():
            try:
                buf += child.before or ""
            except Exception:
                pass
            try:
                buf += child.after or ""
            except Exception:
                pass
            return buf

    try:
        buf += child.before or ""
    except Exception:
        pass
    return buf


def play_compiled_zork_with_nlp():
    parser = EnhancedNLPParser()
    exe = resolve_executable()

    print("Starting Zork with Natural Language Interface…")
    print(f"Executable: {exe}")

    child = spawn_game(exe)
    child.timeout = 2

    initial = read_until_prompt_or_quiet(child, total_timeout=4.0)
    if initial.strip():
        print(initial.strip())

    if hasattr(child, "isalive") and not child.isalive():
        print("\nThe Zork process exited immediately.")
        return

    while True:
        try:
            user = input("> ").strip()
            if not user:
                continue

            if user.lower() in {"quit", "q", "exit"}:
                try:
                    child.sendline("quit")
                    read_until_prompt_or_quiet(child, total_timeout=1.0)
                    child.sendline("y")
                    read_until_prompt_or_quiet(child, total_timeout=1.0)
                except Exception:
                    pass
                break

            if hasattr(child, "isalive") and not child.isalive():
                print("(Game has ended)")
                break

            parsed = parser.parse_command(user)
            zcmd = parser.to_zork_command(parsed) or user

            if user.lower() != zcmd.lower():
                print(f"[→ {zcmd}]")

            child.sendline(zcmd)
            out = read_until_prompt_or_quiet(child, total_timeout=3.0)
            if not out and hasattr(child, "isalive") and not child.isalive():
                print("(Game ended)")
                break

            if out:
                clean = []
                for line in out.splitlines():
                    s = line.strip()
                    if not s:
                        continue
                    if s.lower() == zcmd.lower() or s.lower() == user.lower():
                        continue
                    clean.append(s)
                if clean:
                    print("\n".join(clean))

        except KeyboardInterrupt:
            print("\n(ctl-c) quitting…")
            try:
                child.sendline("quit")
                read_until_prompt_or_quiet(child, total_timeout=1.0)
                child.sendline("y")
            except Exception:
                pass
            break
        except Exception as e:
            msg = str(e)
            if "End of file" in msg or "EOF" in msg:
                print("(Game closed its output)")
            else:
                print(f"Error: {e}")
            break

    try:
        child.close(force=True)
    except Exception:
        pass
    print("Thanks for playing!")


if __name__ == "__main__":
    play_compiled_zork_with_nlp()
