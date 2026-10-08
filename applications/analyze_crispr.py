"""Reproduce the two manuscript Achilles analyses using public inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import pandas as pd

from support.crispr_inputs import (
    infer_labels,
    prepare_matching_covariates,
    gene_effect_matrix,
    score_long_from_matrix,
    control_score_long_from_matrix,
    matched_shared_cbt_pvalues,
    across_conformal_pvalues,
    add_labels,
)
from support.crispr_metrics import task_curves, METHOD_LABELS

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Achilles.zip": "b122b36ab37658c728bc06f23c88cd70",
    "AvanaGuideMap.csv": "b694af3982d70117fd6214fdd8ce2e2e",
    "AchillesCommonEssentialControls.csv": "1cbfa612d5d4d16e287bb9f23964839c",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/crispr")
    parser.add_argument("--output", type=Path, default=ROOT / "results/crispr")
    args = parser.parse_args()
    for name, digest in EXPECTED.items():
        with (args.data_dir / name).open("rb") as handle:
            if hashlib.file_digest(handle, "md5").hexdigest() != digest:
                raise ValueError(
                    f"{name} does not match the manuscript input; see GET_CRISPR.txt"
                )
    args.output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.data_dir / "Achilles.zip") as archive:
        labels = infer_labels(
            archive, args.data_dir / "AchillesCommonEssentialControls.csv"
        )
        for task, stem in [
            ("unrestricted", "allgenes"),
            ("global_null", "no_positives"),
        ]:
            matrix = gene_effect_matrix(
                archive, f"Achilles/chronos_gene_effect_{stem}.csv"
            )
            controls_set = (
                labels.negative_controls_all
                if task == "unrestricted"
                else labels.negative_controls_train
            )
            scores = score_long_from_matrix(
                matrix,
                method="Chronos score",
                score_sign=-1,
                exclude_genes=controls_set,
            )
            controls = control_score_long_from_matrix(
                matrix, control_genes=controls_set, score_sign=-1
            )
            # The paper used eight guide-design covariates, without expression data.
            cov = prepare_matching_covariates(
                avana_guide_map=args.data_dir / "AvanaGuideMap.csv",
                expression_csv=args.output / "expression_not_used",
                blocks=matrix.index,
            )
            if cov is None or cov.expression is not None:
                raise RuntimeError("Expected guide-design covariates only.")
            matched, diagnostics = matched_shared_cbt_pvalues(
                scores=scores, controls=controls, covariates=cov, task=task
            )
            block_controls = {
                block: group.score.to_numpy()
                for block, group in controls.groupby("block")
            }
            baseline = across_conformal_pvalues(scores, block_controls)
            within_cell = baseline.assign(method="Within-cell conformal")
            pvalues = pd.concat([baseline, within_cell, *matched], ignore_index=True)
            pvalues = pvalues[pvalues.method.isin(METHOD_LABELS)].copy()
            pvalues = add_labels(pvalues, labels, task=f"achilles_{stem}")
            pvalues.to_csv(args.output / f"{task}_pvalues.csv.gz", index=False)
            diagnostics.to_csv(args.output / f"{task}_matching.csv", index=False)
            task_curves(pvalues, task).to_csv(
                args.output / f"{task}_curves.csv", index=False
            )
    (args.output / "inputs.json").write_text(json.dumps(EXPECTED, indent=2) + "\n")


if __name__ == "__main__":
    main()
