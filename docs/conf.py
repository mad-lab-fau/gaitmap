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
from importlib import import_module
from inspect import getsourcelines, getsourcefile
from pathlib import Path

import toml
from sphinx_gallery.sorting import ExplicitOrder

import gaitmap

sys.path.insert(0, os.path.abspath(".."))

URL = "https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap/-/blob/master"

# -- Project information -----------------------------------------------------

# Info from poetry config:
info = toml.load("../pyproject.toml")["tool"]["poetry"]

project = info["name"]
author = ", ".join(info["authors"])
release = info["version"]

copyright = "2020 - {}, MaD-Lab FAU, Digital Health and Gait-Analysis Group".format(datetime.now().year)


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
    "gitlab_url": "https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/gaitmap",
    "show_prev_next": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# -- Options for extensions --------------------------------------------------
# Intersphinx

# intersphinx configuration
intersphinx_module_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "tslearn": ("https://tslearn.readthedocs.io/en/stable/", None),
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    **intersphinx_module_mapping,
}

# Sphinx Gallary
sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["./auto_examples"],
    "reference_url": {"gaitmap": None, **{k: v[0] for k, v in intersphinx_module_mapping.items()}},
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


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    module = import_module(info["module"])
    obj = get_nested_attr(module, info["fullname"])
    code_line = None
    filename = ""
    try:
        filename = str(Path(getsourcefile(obj)).relative_to(Path(getsourcefile(gaitmap)).parent.parent))
    except:
        pass
    try:
        code_line = getsourcelines(obj)[-1]
    except:
        pass
    if filename:
        if code_line:
            return "{}/{}#L{}".format(URL, filename, code_line)
        return "{}/{}".format(URL, filename)


def skip_properties(app, what, name, obj, skip, options):
    """This removes all properties from the documentation as they are expected to be documented in the docstring."""
    if isinstance(obj, property):
        return True


def setup(app):
    app.connect("autodoc-skip-member", skip_properties)
