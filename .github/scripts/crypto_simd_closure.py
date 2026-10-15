"""Print `true` when silver_crypto_simd's locked dependency closure differs
between two Cargo.lock files, else `false`.

Usage: crypto_simd_closure.py BASE_LOCK HEAD_LOCK

Every resolved entry is compared whole, so a changed version, checksum or git
revision counts. Cargo.lock does not mark dev-dependencies, so they count too.
An edge that matches no entry, or more than one, also prints `true`: an
unparsed lockfile must never make the proof look irrelevant.
"""

import re
import sys
import tomllib

ROOT = "silver_crypto_simd"
# `name`, `name version` or `name version (source)`; the source omits any `#rev`.
EDGE = re.compile(r"(?P<name>\S+)(?: (?P<version>\S+))?(?: \((?P<source>[^)]+)\))?")


class Unresolved(Exception):
    pass


def resolve(edge, packages):
    m = EDGE.fullmatch(edge)
    if m is None:
        raise Unresolved(edge)
    found = [
        p
        for p in packages
        if p["name"] == m["name"]
        and (m["version"] is None or p["version"] == m["version"])
        and (m["source"] is None or p.get("source", "").split("#")[0] == m["source"])
    ]
    if len(found) != 1:
        raise Unresolved(edge)
    return found[0]


def closure(path):
    with open(path, "rb") as f:
        packages = tomllib.load(f).get("package", [])
    roots = [p for p in packages if p["name"] == ROOT]
    if len(roots) != 1:
        return None
    seen = {}
    todo = roots
    while todo:
        p = todo.pop()
        key = (p["name"], p["version"], p.get("source"))
        if key not in seen:
            seen[key] = p
            todo.extend(resolve(e, packages) for e in p.get("dependencies", []))
    return seen


def changed(base_path, head_path):
    try:
        base, head = closure(base_path), closure(head_path)
    except Unresolved as e:
        print(f"unresolved dependency edge: {e}", file=sys.stderr)
        return True
    return head is None or base != head


if __name__ == "__main__":
    print("true" if changed(sys.argv[1], sys.argv[2]) else "false")
