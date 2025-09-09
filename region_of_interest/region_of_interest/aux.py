from pathlib import Path
from typing import List

def find_parent_cwd(path: Path, experiment_name: str , data_path: List[Path]) -> Path:
    """
    Finds the parent working directory for running commands based on the given path, experiment name, and a list of data paths.

    Args:
        path (Path): The target path to search from.
        experiment_name (str): The name of the experiment to match within the data paths.
        data_path (List[Path]): A list of possible data paths to search for the experiment.

    Returns:
        Path: The parent directory path where the experiment is located, or None if not found.

    Example:
        If experiment_name is 'data' and data_path contains '/workspace/Desktop/FruitProposal/datasets/data',
        the function will return the parent directory up to 'datasets'.
    """

    path = str(path).split("/")
    experiment_path = None
    r = None

    # If name in p in data_path, keep
    for p in data_path:
        if experiment_name in str(p).split("/"):
            experiment_path = p

    for i, dir in enumerate(str(experiment_path).split("/")):
        if path[0] == dir:
            r = "/".join(str(experiment_path).split("/")[:i])
            return Path(r)