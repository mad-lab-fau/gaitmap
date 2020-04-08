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
import sys
from datetime import datetime
from importlib import import_module
from inspect import getsourcelines

import sphinx_bootstrap_theme

sys.path.insert(0, os.path.abspath(".."))

URL = "https://mad-srv.informatik.uni-erlangen.de/newgaitpipeline/gaitmap/-/blob/master"

# -- Project information -----------------------------------------------------


project = "gaitmap"
copyright = "2020 - {}, MaD-Lab FAU, Digital Health and Gait-Analysis Group".format(datetime.now().year)
author = (
    "Arne Küderle <arne.kuederle@fau.de>, "
    "Martin Ullrich <martin.ullrich@fau.de>, "
    "Nils Roth <nils.roth@fau.de>, "
    "Malte Ollenschläger <Malte.Ollenschlaeger@fau.de>, "
    "Alzhraa Ahmed <alzhraa.ahmed@fau.de>, "
    "Felix Kluge <felix.kluge@fau.de>"
)

# The full version, including alpha/beta/rc tags
release = "0.1.0"


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
    # "sphinx_gallery.gen_gallery",
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

# Add any paths that contain templates here, relative to this directory.
templates_path = ["templates"]

# generate autosummary even if no references
autosummary_generate = True

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
html_theme = "bootstrap"
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]

# -- Options for extensions --------------------------------------------------
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
    try:
        code_line = getsourcelines(obj)[-1]
    except:
        pass
    filename = info["module"].replace(".", "/")
    if code_line:
        return "{}/{}.py#L{}".format(URL, filename, code_line)
    return "{}/{}.py".format(URL, filename)


# Intersphinx

# intersphinx configuration
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "tslearn": ("https://tslearn.readthedocs.io/en/latest/", None),
}
