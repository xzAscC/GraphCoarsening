"""Convert JSON result files to Markdown reports in results/ directory."""

import json
import os
import glob


def json_to_markdown(json_path):
    """Convert a JSON result file to a Markdown summary."""
    with open(json_path) as f:
        data = json.load(f)

    basename = os.path.splitext(os.path.basename(json_path))[0]
    lines = [f"# Experiment Results: {basename}", ""]

    if isinstance(data, dict):
        _dict_to_md(data, lines, depth=0)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.append(f"## Entry {i+1}")
            lines.append("")
            if isinstance(item, dict):
                _dict_to_md(item, lines, depth=1)
            else:
                lines.append(str(item))
                lines.append("")
    else:
        lines.append(str(data))
        lines.append("")

    return "\n".join(lines)


def _dict_to_md(d, lines, depth=0):
    prefix = "#" * (depth + 2)
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{prefix} {key}")
            lines.append("")
            _dict_to_md(value, lines, depth + 1)
        elif isinstance(value, list):
            if value and isinstance(value[0], (int, float)):
                lines.append(f"- **{key}**: mean={sum(value)/len(value):.4f}, std={_std(value):.4f}, n={len(value)}")
            elif value and isinstance(value[0], dict):
                lines.append(f"{prefix} {key}")
                lines.append("")
                for i, item in enumerate(value):
                    lines.append(f"{'#'*(depth+3)} Item {i+1}")
                    lines.append("")
                    _dict_to_md(item, lines, depth + 2)
            else:
                lines.append(f"- **{key}**: {value}")
        elif isinstance(value, float):
            lines.append(f"- **{key}**: {value:.4f}")
        else:
            lines.append(f"- **{key}**: {value}")
    lines.append("")


def _std(values):
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5


def main():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("No results directory found")
        return

    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    if not json_files:
        print("No JSON result files found")
        return

    os.makedirs(results_dir, exist_ok=True)

    for json_path in sorted(json_files):
        md_path = os.path.splitext(json_path)[0] + ".md"
        md_content = json_to_markdown(json_path)
        with open(md_path, "w") as f:
            f.write(md_content)
        print(f"Converted {json_path} -> {md_path}")


if __name__ == "__main__":
    main()
