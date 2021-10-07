import json
from typing import List, Optional, Union

import fire
import pandas as pd

from jury import Jury
from jury import __version__ as jury_version


def file_extension(path: str) -> str:
    return ".".join(path.split("/")[-1].split(".")[1:])


def read_config(path: str):
    with open(path, "r") as jf:
        return json.load(jf)


def read_file(filepath: str) -> Union[List[str], List[List[str]]]:
    if file_extension(filepath) == "csv":
        df = pd.read_csv(filepath, header=None)
        content = df.to_numpy().tolist()
    elif file_extension(filepath) == "tsv":
        df = pd.read_csv(filepath, header=None)
        content = df.to_numpy().tolist()
    else:
        with open(filepath, "r") as in_file:
            content = in_file.readlines()
    return content


def from_file(predictions: str, references: str, reduce_fn: Optional[str] = None, config: Optional[str] = None):
    args = read_config(config) if config is not None else {}
    predictions = predictions if predictions is not None else args.get("predictions")
    references = references if references is not None else args.get("references")
    reduce_fn = reduce_fn if reduce_fn is not None else args.get("reduce_fn")
    metrics = args.get("metrics")

    predictions = read_file(predictions)
    references = read_file(references)
    jury = Jury(metrics=metrics)
    scores = jury(predictions=predictions, references=references, reduce_fn=reduce_fn)
    print(json.dumps(scores, default=str, indent=4))


def app() -> None:
    """Cli app."""
    fire.Fire({"version": jury_version, "eval": from_file})


if __name__ == "__main__":
    app()
