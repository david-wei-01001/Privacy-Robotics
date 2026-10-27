#!/usr/bin/env python3
"""Record plan changes automatically (PLAN §13.3).

  python3 tools/plan_log.py new --trigger "..." --why "..." --evidence "..." [--decisions 0007]
      Compares docs/plan/PLAN.md with the last snapshot, section by section, bumps the minor version in the
      plan header, prepends a CHANGELOG entry, and refreshes the snapshot.
  python3 tools/plan_log.py snapshot      Save the current plan as the comparison baseline.
  python3 tools/plan_log.py check         Used by the pre-commit hook: a staged PLAN.md change needs a
                                          CHANGELOG entry for the plan's current version.
Works without commits: the baseline is docs/plan/.snapshot.md, not git.
"""
import argparse, datetime as dt, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs/plan/PLAN.md"
LOG = ROOT / "docs/plan/CHANGELOG.md"
SNAP = ROOT / "docs/plan/.snapshot.md"
VER_RE = re.compile(r"\*\*Plan version (\d+)\.(\d+) \((\d{4}-\d{2}-\d{2})\)\.\*\*")


def sections(text: str) -> dict:
    out, key, buf, in_code = {}, "(preamble)", [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{2,3} ", line):
            out[key] = "\n".join(buf)
            key, buf = line.lstrip("#").strip(), []
        else:
            buf.append(line)
    out[key] = "\n".join(buf)
    return out


def diff_sections(old: str, new: str):
    a, b = sections(old), sections(new)
    added = [k for k in b if k not in a]
    removed = [k for k in a if k not in b]
    changed = [k for k in b if k in a and a[k] != b[k] and k != "(preamble)"]
    return added, removed, changed


def line_counts(old: str, new: str):
    import difflib
    plus = minus = 0
    for l in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
        if l.startswith("+") and not l.startswith("+++"):
            plus += 1
        elif l.startswith("-") and not l.startswith("---"):
            minus += 1
    return plus, minus


def cmd_new(args):
    new = PLAN.read_text(encoding="utf-8")
    old = SNAP.read_text(encoding="utf-8") if SNAP.exists() else ""
    m = VER_RE.search(new)
    if not m:
        sys.exit("plan header with '**Plan version X.Y (YYYY-MM-DD).**' not found")
    major, minor = int(m.group(1)), int(m.group(2)) + 1
    today = dt.date.today().isoformat()
    new = new[:m.start()] + f"**Plan version {major}.{minor} ({today}).**" + new[m.end():]
    PLAN.write_text(new, encoding="utf-8")
    added, removed, changed = diff_sections(old, new)
    plus, minus = line_counts(old, new)
    parts = [f"added: {', '.join(added)}" if added else "", f"removed: {', '.join(removed)}" if removed else "",
             f"changed: {', '.join(changed)}" if changed else ""]
    when = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = [f"## v{major}.{minor} ({when})",
             f"- Sections: {'; '.join(p for p in parts if p) or 'none'} (+{plus} / -{minus} lines)",
             f"- Trigger: {args.trigger}", f"- Why: {args.why}", f"- Evidence: {args.evidence or 'n/a'}"]
    if args.decisions:
        entry.append(f"- Decisions: {args.decisions}")
    log = LOG.read_text(encoding="utf-8")
    i = log.find("\n## ")
    log = log[:i + 1] + "\n".join(entry) + "\n\n" + log[i + 1:] if i >= 0 else log + "\n" + "\n".join(entry) + "\n"
    LOG.write_text(log, encoding="utf-8")
    SNAP.write_text(new, encoding="utf-8")
    print(f"plan -> v{major}.{minor}; CHANGELOG updated")


def cmd_check(_):
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if "docs/plan/PLAN.md" not in staged:
        return
    plan = subprocess.run(["git", "show", ":docs/plan/PLAN.md"], cwd=ROOT, capture_output=True, text=True).stdout
    log = subprocess.run(["git", "show", ":docs/plan/CHANGELOG.md"], cwd=ROOT, capture_output=True, text=True).stdout
    m = VER_RE.search(plan)
    if not m or f"## v{m.group(1)}.{m.group(2)} " not in log:
        sys.exit("PLAN.md changed without a CHANGELOG entry for its version: run tools/plan_log.py new ...")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--trigger", required=True)
    n.add_argument("--why", required=True)
    n.add_argument("--evidence", default="")
    n.add_argument("--decisions", default="")
    sub.add_parser("snapshot")
    sub.add_parser("check")
    a = ap.parse_args()
    if a.cmd == "new":
        cmd_new(a)
    elif a.cmd == "snapshot":
        SNAP.write_text(PLAN.read_text(encoding="utf-8"), encoding="utf-8")
        print("snapshot saved")
    else:
        cmd_check(a)


if __name__ == "__main__":
    main()
