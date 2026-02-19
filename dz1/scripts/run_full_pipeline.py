from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RL + BC + report artifact pipeline.")
    parser.add_argument("--rl-config", type=str, required=True)
    parser.add_argument("--bc-config", type=str, required=True)
    parser.add_argument("--report-output", type=str, default="report/assets")
    args = parser.parse_args()

    _run([sys.executable, "-m", "scripts.run_rl_suite", "--config", args.rl_config])
    _run(
        [
            sys.executable,
            "-m",
            "scripts.run_bc_suite",
            "--config",
            args.bc_config,
            "--expert",
            "artifacts/rl/best_expert.pt",
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "scripts.make_report_artifacts",
            "--input",
            "artifacts",
            "--output",
            args.report_output,
        ]
    )


if __name__ == "__main__":
    main()
