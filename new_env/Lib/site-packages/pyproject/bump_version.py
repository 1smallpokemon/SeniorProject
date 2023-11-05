#! /usr/bin/python
from __future__ import print_function
import argparse
import os
import re
import sys
import subprocess

parser = argparse.ArgumentParser(usage="%(prog)s [options] args...")
parser.add_argument("version")
parser.add_argument("--no-tag", dest="tag", action="store_false", default=True)

def main(args):
    version_filename = _get_version_filename(args)
    if version_filename is None:
        print("Version file not found!", file=sys.stderr)
        return -1
    with open(version_filename, "w") as outfile:
        print('__version__ = "{}"'.format(args.version),
              file=outfile)
    _shell("git commit -a -m 'bump version'")
    if args.tag:
        _shell("git tag {}".format(args.version))
    return 0

def _bump(version, args):
    returned = version[:]
    if args.major_bump:
        returned = [returned[0] + 1] + [0 for x in returned[1:]]
    elif args.minor_bump:
        returned = [returned[0], returned[1] + 1] + [0 for x in returned[2:]]
    else:
        returned[-1] += 1
    return returned

def _shell(cmd):
    returncode = subprocess.call(cmd, shell=True)
    if returncode != 0:
        raise Exception("shell command failed: {!r}".format(cmd))

def _get_version_filename(args):
    matched = []
    for path, dirnames, filenames in os.walk("."):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if path == ".":
            for excluded in ['build']:
                if excluded in dirnames:
                    dirnames.remove(excluded)
        for filename in filenames:
            if filename == "__version__.py":
                matched.append(os.path.join(path, filename))
    if not matched:
        return None
    if len(matched) > 1:
        raise Exception("Too many version files found: {}".format(matched))
    return matched[0]

def _get_version(filename):
    content = open(filename, "rb").read()
    match = re.match(r"^__version__\s+=\s+\"(\d)\.(\d+)\.(\d+)\"\s*$", content.decode('utf-8'), re.S | re.M)
    if not match:
        return _get_latest_tag()
    return list(map(int, match.groups()[:-1]))

def _get_latest_tag():
    p = subprocess.Popen("git tag", shell=True, stdout=subprocess.PIPE)
    if p.wait() != 0:
        raise RuntimeError("Cannot extract tags")
    returned = []
    for tag in p.stdout:
        if tag.startswith("v"):
            tag = tag[1:]
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)\s*$", tag)
        if not match:
            continue
        returned.append(list(map(int, match.groups())))
    returned.sort()
    return returned[-1]

#### For use with entry_points/console_scripts
def main_entry_point():
    args = parser.parse_args()
    sys.exit(main(args))
if __name__ == '__main__':
    main_entry_point()
