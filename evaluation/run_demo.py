import os
import subprocess
import sys


def main():

    if len(sys.argv) < 3:

        print(
            "Usage: python evaluation/run_demo.py "
            "<original_image> <generated_image>"
        )

        return 1

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    original_image = sys.argv[1]
    generated_image = sys.argv[2]

    # Convert relative paths to project paths
    if not os.path.isabs(original_image):

        original_image = os.path.join(
            project_root,
            original_image
        )

    if not os.path.isabs(generated_image):

        generated_image = os.path.join(
            project_root,
            generated_image
        )

    # Check files
    if not os.path.exists(original_image):

        print(
            f"[ERROR] Original image not found:\n"
            f"{original_image}"
        )

        return 1

    if not os.path.exists(generated_image):

        print(
            f"[ERROR] Generated image not found:\n"
            f"{generated_image}"
        )

        return 1

    # Output report
    output_report = os.path.join(
        project_root,
        "test_data",
        "qc_report.json"
    )

    # Main evaluation script
    test_script = os.path.join(
        project_root,
        "test_quality_report.py"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT pass --semantic-threshold
    # because project requirement says TBD.
    # --------------------------------------------------------

    command = [
        sys.executable,
        test_script,

        "--original",
        original_image,

        "--generated",
        generated_image,

        "--out",
        output_report,

        "--edge-threshold",
        "0.15",

        "--bbox-threshold",
        "0.60",

        "--count-drift",
        "2",

        "--fid-threshold",
        "15"
    ]

    print()

    result = subprocess.run(
        command,
        cwd=project_root,
        check=False
    )

    return result.returncode


if __name__ == "__main__":

    raise SystemExit(
        main()
    )