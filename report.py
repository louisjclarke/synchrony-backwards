"""Save results as JSON and write them into README.md, so the README always matches the code."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def update_readme(tag, text, readme=ROOT / "README.md"):
    """Replace the text between <!-- tag:start --> and <!-- tag:end --> in the README."""
    readme = Path(readme)
    start, end = f"<!-- {tag}:start -->", f"<!-- {tag}:end -->"
    s = readme.read_text(encoding="utf-8")
    if start not in s or end not in s:
        return False
    head, rest = s.split(start, 1)
    _, tail = rest.split(end, 1)
    readme.write_text(f"{head}{start}\n{text.strip()}\n{end}{tail}", encoding="utf-8")
    return True


def sci(x, digits=3):
    """3.16e+13 written as 3.16 × 10<sup>13</sup> for the README."""
    m, e = f"{x:.{digits - 1}e}".split("e")
    return f"{m} × 10<sup>{int(e)}</sup>" if abs(int(e)) >= 4 else f"{x:.{digits}g}"
