import argparse

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-o", "--Output", help="Show Output")
    parser.add_argument(
        "-d", "--dir", help="Directory of video and where to save", required=True
    )
    parser.add_argument(
        "-f", "--file", help="File of video and where to save", required=True
    )
    parser.add_argument(
        "-t",
        "--times",
        nargs="+",
        help="Times in hh:mm:ss format; where to split the video",
        required=True,
    )

    # Read arguments from command line
    print(parser.parse_args()._get_kwargs())
    dict_args = dict(parser.parse_args()._get_kwargs())
    print(dict_args["file"])

    for a, value in parser.parse_args()._get_kwargs():
        if value is not None:
            print(f"{a}\t{value}")
