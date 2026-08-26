"""Utilities for generating the per-sample Quarto pipeline report."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from path_config import PathConfig
from pipeline_config import PipelineConfig


logger = logging.getLogger(__name__)

REPORT_TEMPLATE_NAME = "perSample_preprocess_report.qmd"


def generate_quarto_report(
    paths: PathConfig,
    config: PipelineConfig,
) -> Path | None:
    """Render the finished pipeline run as a self-contained Quarto HTML report.

    The report template is rendered in a temporary directory rather than
    directly into the analysis directory. This keeps Quarto's generated
    support files (including its HTML/JS resources) together while Quarto
    creates the standalone HTML. Only the final HTML file is copied into the
    pipeline working directory.
    """

    quarto_executable = shutil.which("quarto")
    if quarto_executable is None:
        logger.warning(
            "Quarto was not found on PATH. Pipeline completed, "
            "but the HTML report was not generated."
        )
        return None

    report_template = Path(__file__).resolve().parent / REPORT_TEMPLATE_NAME
    if not report_template.is_file():
        logger.error(
            "Quarto report template was not found at %s",
            report_template,
        )
        return None

    paths.working_dir.mkdir(parents=True, exist_ok=True)

    output_filename = (
        f"pipeline_report_{config.save_prefix}_{config.run_date}.html"
    )
    report_path = paths.working_dir / output_filename

    if report_path.exists():
        try:
            report_path.unlink()
        except OSError as exc:
            logger.warning(
                "Could not remove existing report %s: %s",
                report_path,
                exc,
            )

    # Render in an isolated directory. Quarto creates its intermediate
    # notebook and HTML resources next to the source document, which avoids
    # the missing quarto.js/resource-path problem caused by rendering a QMD
    # from one directory while placing its output in another.
    with tempfile.TemporaryDirectory(
        prefix="quarto_pipeline_report_",
        dir=str(paths.working_dir),
    ) as tmp_dir:
        tmp_path = Path(tmp_dir)
        tmp_qmd = tmp_path / REPORT_TEMPLATE_NAME
        tmp_output = tmp_path / output_filename

        shutil.copy2(report_template, tmp_qmd)

        command = [
            quarto_executable,
            "render",
            tmp_qmd.name,
            "--to",
            "html",
            "--output",
            output_filename,
            "-P",
            f"config_pickle={paths.pipeline_config_pickle}",
        ]

        # The saved PipelineConfig is a pickle containing the PipelineConfig
        # class. Quarto executes the QMD from the temporary render directory,
        # so explicitly expose the pipeline source directory to the Jupyter
        # kernel. This lets pickle import pipeline_config (and its dependency
        # gene_sets) without making the QMD depend on the current working
        # directory or on path_config.
        source_dir = Path(__file__).resolve().parent
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(source_dir)
            if not existing_pythonpath
            else str(source_dir) + os.pathsep + existing_pythonpath
        )

        logger.info("Generating Quarto pipeline report: %s", report_path)

        try:
            completed = subprocess.run(
                command,
                cwd=tmp_path,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            logger.error(
                "Unable to execute Quarto while generating the pipeline report: %s",
                exc,
            )
            return None

        if completed.stdout:
            logger.info("Quarto output:\n%s", completed.stdout.rstrip())

        if completed.returncode != 0:
            if completed.stderr:
                logger.error(
                    "Quarto error output:\n%s",
                    completed.stderr.rstrip(),
                )
            logger.error(
                "Quarto report generation failed with exit code %s. "
                "Pipeline outputs remain available.",
                completed.returncode,
            )
            return None

        if not tmp_output.is_file():
            if completed.stderr:
                logger.warning(
                    "Quarto warnings:\n%s",
                    completed.stderr.rstrip(),
                )
            logger.error(
                "Quarto reported success, but the expected temporary report "
                "was not found at %s",
                tmp_output,
            )
            return None

        if completed.stderr:
            logger.info(
                "Quarto stderr/warnings:\n%s",
                completed.stderr.rstrip(),
            )

        # The QMD uses embed-resources: true and standalone: true, so the
        # final HTML is intended to contain the figures and Quarto assets.
        # Copy only that final artifact into the project output directory.
        shutil.copy2(tmp_output, report_path)

    logger.info("Quarto pipeline report generated: %s", report_path)
    return report_path
