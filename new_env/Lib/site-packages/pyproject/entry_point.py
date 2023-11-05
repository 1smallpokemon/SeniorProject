#! /usr/bin/env python
import argparse
import logging
import os
import sys
from .factory import create_package
from .context import parameters

_VARIABLES_DEFAULTS = {}

parser = argparse.ArgumentParser(usage="%(prog)s [options] args...")
parser.add_argument("-v", action="append_const", const=1, dest="verbosity", default=[],
                    help="Be more verbose. Can be specified multiple times to increase verbosity further")
parser.add_argument("dest_dir")

parameters.populate_parser(parser)

def main(args):
    ctx = parameters.build_context(args, interactive=True, defaults={"name": os.path.basename(args.dest_dir)})
    unspecified = {arg for arg, value in ctx.items() if value is None and parameters[arg].required}
    if unspecified:
        parser.error("Unspecified value(s): {}".format(", ".join(sorted(unspecified))))
    create_package(args.dest_dir, context=ctx)
    return 0

################################## Boilerplate ################################
def _configure_logging(args):
    verbosity_level = len(args.verbosity)
    if verbosity_level == 0:
        level = "WARNING"
    elif verbosity_level == 1:
        level = "INFO"
    else:
        level = "DEBUG"
    logging.basicConfig(
        stream=sys.stderr,
        level=level,
        format="%(asctime)s -- %(message)s"
        )


#### For use with entry_points/console_scripts
def main_entry_point():
    args = parser.parse_args()
    _configure_logging(args)
    sys.exit(main(args))


if __name__ == "__main__":
    main_entry_point()
