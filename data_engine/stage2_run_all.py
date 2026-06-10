import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_script(name: str, argv: list[str]) -> int:
    """Run a Python script in the script directory; return exit code."""
    script_path = os.path.join(SCRIPT_DIR, name)
    cmd = [sys.executable, script_path] + argv
    return subprocess.run(cmd, cwd=SCRIPT_DIR).returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stage 2: s4 revise instances → s5 prompt generation → s6 rubric generation."
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Max number of tasks for s4 (for testing)",
    )
    parser.add_argument(
        "--filtered", default=None,
        help="Path to filtered_tasks.json for s4 (default: ../tasks/filtered_tasks.json)",
    )
    parser.add_argument(
        "--s4-output", default=None,
        help="Output path for s4 revised_instances.json (default: ../tasks/revised_instances.json)",
    )
    args, extra = parser.parse_known_args()

    s4_argv = []
    if args.limit is not None:
        s4_argv.extend(["--limit", str(args.limit)])
    if args.filtered is not None:
        s4_argv.extend(["--filtered", args.filtered])
    if args.s4_output is not None:
        s4_argv.extend(["--output", args.s4_output])
    s4_argv.extend(extra)

    print("=== Stage 2: s4 → s5 → s6 ===\n")

    print("--- Step 4: Revise eval instances ---")
    code = run_script("s4_revise_instances.py", s4_argv)
    if code != 0:
        sys.exit(code)

    print("\n--- Step 5: Prompt generation ---")
    code = run_script("s5_prompt_generation.py", [])
    if code != 0:
        sys.exit(code)

    print("\n--- Step 6: Rubric generation ---")
    code = run_script("s6_rubric_generation.py", [])
    if code != 0:
        sys.exit(code)

    print("\n=== Stage 2 complete ===")


if __name__ == "__main__":
    main()
