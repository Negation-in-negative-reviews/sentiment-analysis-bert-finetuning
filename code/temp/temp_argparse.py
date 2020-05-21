import argparse

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--input_file",
                    default=None,
                    type=str,
                    required=True,
                    help="")
args = parser.parse_args()

print(args.input_file)