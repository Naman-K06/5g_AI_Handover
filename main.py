
import subprocess
import sys
import argparse
import os
import datetime
import logging

# -------------------
# 1. LOGGING SETUP
# -------------------
os.makedirs("outputs", exist_ok=True)
logging.basicConfig(
    filename="outputs/pipeline.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------
# 2. SCRIPT MAPPING
# -------------------
SCRIPTS = {
    1: "02_generate_routes",
    2: "03_run_multi_user",
    3: "04_train_ai",
    4: "05_final_comparison",
    5: "06_ablation",
}

SCRIPT_DESCRIPTIONS = {
    1: "Route Generation     - Create variable-speed mobility routes from OSM",
    2: "Baseline Simulation  - 3GPP A3 handover logic with physics engine",
    3: "AI Training          - Train Random Forest handover model",
    4: "Final Comparison     - Compare baseline vs AI with statistics and plots",
    5: "Ablation Study       - Vary stability margin and measure KPI sensitivity",
}

def main(steps):
    """
    Execute pipeline stages in order.
    """
    # --- SHARED SESSION PATH ---
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    base_folder  = os.path.join("outputs", "new_routes")
    os.makedirs(base_folder, exist_ok=True)

    existing_runs  = [d for d in os.listdir(base_folder) if d.startswith("Run_")]
    run_num        = len(existing_runs) + 1
    shared_run_path = os.path.join(base_folder, f"Run_{run_num}_{timestamp}")

    # Set environment variable so all sub-scripts inherit the same output path
    os.environ["CURRENT_RUN_PATH"] = shared_run_path
    os.makedirs(shared_run_path, exist_ok=True)

    print("=" * 80)
    print(f"  5G AI DIGITAL TWIN PIPELINE - Run {run_num}")
    print("=" * 80)
    print(f"  Output Directory : {shared_run_path}")
    print(f"  Executing Steps  : {steps}")
    print()

    for s in steps:
        module_name = SCRIPTS.get(s)
        if not module_name:
            print(f"  Step {s} not recognised. Skipping.")
            logger.warning(f"Step {s} not recognised")
            continue

        script_file = os.path.join("src", f"{module_name}.py")
        if not os.path.exists(script_file):
            print(f"  Script not found: {script_file}. Skipping Step {s}.")
            logger.error(f"Script file not found: {script_file}")
            continue

        print(f"\n{'─' * 80}")
        print(f"  [STEP {s}] {SCRIPT_DESCRIPTIONS.get(s, 'Unknown')}")
        print(f"{'─' * 80}")
        logger.info(f"Starting Step {s}: src.{module_name}")

        res = subprocess.run(
            [sys.executable, "-m", f"src.{module_name}"],
            cwd=os.getcwd()
        )

        if res.returncode != 0:
            print(f"\n  CRITICAL FAILURE in Step {s} (src.{module_name}). Aborting.")
            logger.error(f"Step {s} failed with return code {res.returncode}")
            return res.returncode

        print(f"  Step {s} completed successfully.")
        logger.info(f"Step {s} completed successfully")

    print("\n" + "=" * 80)
    print(f"  PIPELINE COMPLETE")
    print(f"  Results saved to: {shared_run_path}")
    print("=" * 80 + "\n")
    logger.info(f"Pipeline completed. Results in {shared_run_path}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="5G Handover AI Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  1  Route Generation     - Generate OSM road-constrained UE routes
  2  Baseline Simulation  - Run 3GPP A3 multi-user simulation
  3  AI Training          - Train Random Forest handover classifier
  4  Final Comparison     - Full KPI comparison, plots, and statistics
  5  Ablation Study       - Sensitivity analysis on stability margin

Examples:
  python main.py                   Run all steps (1-5)
  python main.py --steps 1 2 3 4   Run steps 1 through 4 (skip ablation)
  python main.py --steps 3 4       Re-train and re-compare (routes already exist)
  python main.py --steps 4         Re-run comparison only
  python main.py --steps 5         Run ablation study only
        """
    )
    parser.add_argument(
        '--steps',
        nargs='+',
        type=int,
        default=None,
        help='Steps to execute (e.g. --steps 1 2 3). If omitted, prompted interactively.'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if output artifacts already exist.'
    )
    args = parser.parse_args()

    # Interactive mode if no steps provided
    if args.steps is None:
        print("\nAvailable pipeline steps:")
        print("-" * 60)
        for k in sorted(SCRIPT_DESCRIPTIONS.keys()):
            print(f"  {k}: {SCRIPT_DESCRIPTIONS[k]}")
        print("-" * 60)
        print('  Type step numbers separated by commas, a range (e.g. 1-4),')
        print('  or "all" to run everything.')
        print()

        def parse_selection(s):
            s = s.strip().lower()
            if s == 'all':
                return list(sorted(SCRIPTS.keys()))
            parts = [p.strip() for p in s.replace(' ', ',').split(',') if p.strip()]
            out = set()
            for p in parts:
                if '-' in p:
                    a, b = p.split('-', 1)
                    out.update(range(int(a), int(b) + 1))
                else:
                    out.add(int(p))
            return sorted(out)

        while True:
            sel = input('Enter steps (e.g. "1,3-5" or "all"): ').strip()
            if not sel:
                print('  No input received. Please try again.')
                continue
            try:
                steps = parse_selection(sel)
            except Exception:
                print('  Invalid input. Use numbers, ranges, or "all".')
                continue
            invalid = [s for s in steps if s not in SCRIPTS]
            if invalid:
                print(f'  Invalid step numbers: {invalid}.')
                print(f'  Valid steps are: {sorted(SCRIPTS.keys())}')
                continue
            break
    else:
        steps = args.steps

    sys.exit(main(steps))