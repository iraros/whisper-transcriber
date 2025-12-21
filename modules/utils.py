from pathlib import Path


def get_tmp_path(filename: str) -> str:
    """
    Finds the project root, locates the 'tmp' folder there,
    and returns the full string path for the given filename.
    """
    # 1. Start from this file's location
    current_path = Path(__file__).resolve()

    # 2. Walk up until we find the folder containing 'packages.txt' or 'app.py'
    # This identifies your 'transcriber' root folder regardless of where this script sits
    project_root = current_path
    for parent in current_path.parents:
        if (parent / "packages.txt").exists() or (parent / "app.py").exists():
            project_root = parent
            break

    # 3. Target the 'tmp' folder at that root
    tmp_dir = project_root / "tmp"

    # 4. Create it if it's missing (important for first-time cloud runs)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # 5. Return the full absolute path as a string
    return str(tmp_dir / filename)

print(get_tmp_path(''))