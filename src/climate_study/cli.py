import argparse

from .data import prepare
from .demo import run_demo
from .experiment import evaluate, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Climate emulation under distribution shift")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Validate the course CSV and freeze splits")
    prep.add_argument("--csv", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--split", choices=["location", "row"], default="location")
    fit = commands.add_parser("train", help="Select baselines using validation only")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--seeds", type=int, nargs="+", default=[42])
    fit.add_argument("--trees", type=int, default=100)
    test = commands.add_parser("evaluate", help="Evaluate the three held-out regimes")
    test.add_argument("--data", required=True)
    test.add_argument("--run", required=True)
    test.add_argument("--spatial-csv", required=True)
    test.add_argument("--scenario-csv", required=True)
    demo = commands.add_parser("demo", help="Run all stages on explicitly synthetic data")
    demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.csv, args.output, args.seed, args.split)
        elif args.command == "train":
            train(args.data, args.output, args.seeds, args.trees)
        elif args.command == "evaluate":
            evaluate(args.data, args.run, args.spatial_csv, args.scenario_csv)
        else:
            run_demo(args.output)
        print("Completed successfully.")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
