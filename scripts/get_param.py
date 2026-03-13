import yaml
import sys
import re
import os


def resolve_variables(value, data):
    """
    Rudimentary variable resolution for strings like ${project.paths.storage_root}.
    """
    if isinstance(value, list):
        return [resolve_variables(v, data) for v in value]

    if not isinstance(value, dict):
        if not isinstance(value, str):
            return value

        matches = re.findall(r"\${([^}]+)}", value)
        for match in matches:
            keys = match.split(".")
            ref = data
            try:
                for k in keys:
                    ref = ref[k]
                # Recursively resolve if the reference itself has variables
                resolved_ref = resolve_variables(ref, data)
                value = value.replace(f"${{{match}}}", str(resolved_ref))
            except (KeyError, TypeError):
                continue
        return value
    else:
        # If it's a dict, resolve all its values
        return {k: resolve_variables(v, data) for k, v in value.items()}


def get_param(yaml_file, param_path, default=""):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Runtime site override via environment variable
    site_override = os.environ.get("PF_SITE")
    if site_override and "project" in data and "sites" in data["project"]:
        if site_override in data["project"]["sites"]:
            # Perform a shallow merge similar to YAML's '<<'
            data["project"].update(data["project"]["sites"][site_override])

    keys = param_path.split(".")
    val = data
    try:
        for k in keys:
            val = val[k]
        resolved = resolve_variables(val, data)
        if isinstance(resolved, list):
            return " ".join(map(str, resolved))
        return resolved
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: PF_SITE=tallinn python3 get_param.py <yaml_file> <param_path> [default]")
        sys.exit(1)

    yaml_file = sys.argv[1]
    param_path = sys.argv[2]
    default = sys.argv[3] if len(sys.argv) > 3 else ""

    print(get_param(yaml_file, param_path, default))
