import os
import jinja2

jinja_env = jinja2.Environment(undefined=jinja2.StrictUndefined)

_SKELETON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "_skeleton"))

def create_package(dest_dir, context):
    _transfer(_SKELETON_DIR, dest_dir, context)

def _transfer(src_path, dest_path, context):
    if not os.path.isdir(dest_path):
        os.makedirs(dest_path)
    for filename in os.listdir(src_path):
        src_filename = os.path.join(src_path, filename)
        dest_filename = os.path.join(dest_path, filename)
        if "{" in dest_filename:
            dest_filename = jinja_env.from_string(dest_filename).render(context)
        if os.path.isdir(src_filename):
            _transfer(src_filename, dest_filename, context)
        else:
            with open(src_filename) as src:
                contents = src.read()
                if src_filename.endswith(".j2"):
                    dest_filename = dest_filename[:-3]
                    contents = jinja_env.from_string(contents).render(context)
            with open(dest_filename, "w") as dest:
                dest.write(contents)
