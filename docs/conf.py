# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import sys
from datetime import datetime
from inspect import getsourcefile
from pathlib import Path
from typing import List

import toml
from sphinx_gallery.sorting import ExplicitOrder

HERE = Path(__file__)

sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from sphinxext.githublink import make_linkcode_resolve


def replace_gitlab_links(base_url, text):
    regex = base_url + r"-/(merge_requests|issues|commit)/(\w+)"

    def substitute(matchobj):
        tokens = {"merge_requests": "!", "issues": "#"}
        if matchobj.group(1) == "commit":
            return f"[{matchobj.group(2)[:5]}]({matchobj.group(0)})"
        token = tokens[matchobj.group(1)]
        return f"[{token}{matchobj.group(2)}]({matchobj.group(0)})"

    return re.sub(regex, substitute, text)


def convert_github_links(base_url, text):
    regex = base_url + r"/(pull|issues|commit)/(\w+)"

    def substitute(matchobj):
        if matchobj.group(1) == "commit":
            return f"[{matchobj.group(2)[:5]}]({matchobj.group(0)})"
        return f"[#{matchobj.group(2)}]({matchobj.group(0)})"

    return re.sub(regex, substitute, text)


OLD_URL = "https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/"
URL = "https://github.com/mad-lab-fau/gaitmap/"

# -- Project information -----------------------------------------------------

# Info from poetry config:
info = toml.load("../pyproject.toml")["tool"]["poetry"]

project = info["name"]
author = ", ".join(info["authors"])
release = info["version"]

copyright = f"2020 - {datetime.now().year}, MaD-Lab FAU, Digital Health and Gait-Analysis Group"

# -- Copy the README and Changelog and fix image path --------------------------------------
HERE = Path(__file__).parent
with (HERE.parent / "README.md").open() as f:
    out = f.read()
out = out.replace("./docs/_static/logo/gaitmap_logo_with_text.png", "./_static/logo/gaitmap_logo_with_text.png")
with (HERE / "README.md").open("w+") as f:
    f.write(out)

with (HERE.parent / "CHANGELOG.md").open() as f:
    out = f.read()
out = replace_gitlab_links(OLD_URL, out)
out = convert_github_links(URL, out)
with (HERE / "CHANGELOG.md").open("w+") as f:
    f.write(out)

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.imgconverter",
    "sphinx_gallery.gen_gallery",
    "recommonmark",
]

# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

# Taken from sklearn config
# For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# (useful for viewing the doc offline)
if os.environ.get("NO_MATHJAX"):
    extensions.append("sphinx.ext.imgmath")
    imgmath_image_format = "svg"
    mathjax_path = ""
else:
    extensions.append("sphinx.ext.mathjax")
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/" "tex-chtml.js"

autodoc_default_options = {"members": True, "inherited-members": True, "special_members": True}
# autodoc_typehints = 'description'  # Does not work as expected. Maybe try at future date again

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True
autosummary_generate_overwrite = True

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "templates"]

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "literal"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Activate the theme.
html_theme = "pydata_sphinx_theme"
html_favicon = "_static/logo/gaitmap.ico"
html_logo = "_static/logo/gaitmap_logo.png"
html_theme_options = {
    "github_url": "https://github.com/mad-lab-fau/gaitmap",
    "use_edit_page_button": True,
    "show_prev_next": False,
    # Workaround until pydata-sphinx-theme 0.13 is released (https://github.com/pydata/pydata-sphinx-theme/issues/1094)
    "logo": {
        "image_light": "logo/gaitmap_logo.png",
        "image_dark": "logo/gaitmap_logo.png",
    }
}

html_context = {
    "github_user": "mad-lab-fau",
    "github_repo": "gaitmap",
    "github_version": "master",
    "doc_path": "docs",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for extensions --------------------------------------------------
# Intersphinx

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": (" https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "tpcp": ("https://tpcp.readthedocs.io/en/latest", None),
}

user_agent = "Mozilla/5.0 (X11; Linux x86_64; rv:25.0) Gecko/20100101 Firefox/25.0"

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    **intersphinx_module_mapping,
}

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["./auto_examples"],
    "reference_url": {"gaitmap": None},
    # 'default_thumb_file': 'fig/logo.png',
    "backreferences_dir": "modules/generated/backreferences",
    "doc_module": ("gaitmap",),
    "filename_pattern": re.escape(os.sep),
    "remove_config_comments": True,
    "show_memory": True,
    "subsection_order": ExplicitOrder(
        [
            "../examples/full_pipelines",
            "../examples/preprocessing",
            "../examples/gait_detection",
            "../examples/stride_segmentation",
            "../examples/event_detection",
            "../examples/trajectory_reconstruction",
            "../examples/parameters",
            "../examples/datasets_and_pipelines",
            "../examples/advanced_features",
            "../examples/generic_algorithms",
        ]
    ),
}

# Linkcode


def get_nested_attr(obj, attr):
    attrs = attr.split(".", 1)
    new_obj = getattr(obj, attrs[0])
    if len(attrs) == 1:
        return new_obj
    else:
        return get_nested_attr(new_obj, attrs[1])


linkcode_resolve = make_linkcode_resolve(
    "gaitmap",
    f"{URL}/blob/{{revision}}/{{package}}/{{path}}#L{{lineno}}",
)


def skip_properties(app, what, name, obj, skip, options):
    """This removes all properties from the documentation as they are expected to be documented in the docstring."""
    if isinstance(obj, property):
        return True


GAITMAP_MAD_TEST = """

.. note:: This algorithm is only available via the `gaitmap_mad` package and distributed under a AGPL3 licence.
          To use it, you need to explicitly install the `gaitmap_mad` package.
          Learn more about that here: TODO.

"""


def add_info_about_origin(app, what, name, obj, options, lines: List[str]):
    """Add a short info text to all algorithms that are only available via gaitmap_mad."""
    if what != "class":
        return
    try:
        file_name = getsourcefile(obj)
    except TypeError:
        return
    if file_name and "gaitmap_mad" in file_name:
        lines_to_insert = GAITMAP_MAD_TEST.split("\n")
        lines_to_insert.reverse()
        for l in lines_to_insert:
            lines.insert(2, l)


def setup(app):
    app.connect("autodoc-skip-member", skip_properties)
    app.connect("autodoc-process-docstring", add_info_about_origin)
