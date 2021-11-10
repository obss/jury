import glob
import json
import os.path
from typing import Any, Dict, List, Optional, Tuple, Union

import fire
import pandas as pd

from jury import Jury
from jury import __version__ as jury_version
from jury.utils.common import get_common_keys
from jury.utils.io import json_load, json_save


def file_extension(path: str) -> str:
    return ".".join(path.split("/")[-1].split(".")[1:])


def from_file(
    predictions: str,
    references: str,
    reduce_fn: Optional[str] = None,
    config: Optional[str] = "",
    export: Optional[str] = None,
):
    args = json_load(config) or {}
    predictions = predictions or args.get("predictions")
    references = references or args.get("references")
    reduce_fn = reduce_fn or args.get("reduce_fn")
    metrics = args.get("metrics")
    scorer = Jury(metrics=metrics)

    if os.path.isfile(predictions) and os.path.isfile(references):
        scores = score_from_file(scorer=scorer, predictions=predictions, references=references, reduce_fn=reduce_fn)
    elif os.path.isdir(predictions) and os.path.isdir(references):
        paths = read_folders(predictions, references)
        scores = {}
        for pred_file, ref_file in paths:
            common_name = os.path.basename(pred_file)
            scores[common_name] = score_from_file(
                scorer=scorer, predictions=pred_file, references=ref_file, reduce_fn=reduce_fn
            )
    else:
        raise ValueError("predictions and references either both must be files or both must be folders.")

    if export:
        json_save(scores, export)

    print(json.dumps(scores, default=str, indent=4))


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


def read_folders(predictions_path: str, references_path: str) -> List[Tuple[str, str]]:
    glob_predictions_path = os.path.join(predictions_path, "*")
    glob_references_path = os.path.join(references_path, "*")
    prediction_files = {os.path.basename(p): p for p in glob.glob(glob_predictions_path)}
    reference_files = {os.path.basename(p): p for p in glob.glob(glob_references_path)}

    common_files = get_common_keys(prediction_files, reference_files)

    files_to_read = []
    for common_file in common_files:
        common_pair = (prediction_files[common_file], reference_files[common_file])
        files_to_read.append(common_pair)

    return files_to_read


def score_from_file(scorer: Jury, predictions: str, references: str, reduce_fn: Optional[str] = None) -> Dict[str, Any]:
    predictions = read_file(predictions)
    references = read_file(references)
    return scorer(predictions=predictions, references=references, reduce_fn=reduce_fn)


def app() -> None:
    """Cli app."""
    fire.Fire({"version": jury_version, "eval": from_file})


if __name__ == "__main__":
    app()
