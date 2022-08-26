import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = "OptimUS"
author = "OptimUS team"

with open("../optimus/version.py") as f:
    version = f.read().strip()
    release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "autoapi.extension",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
language = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = None

html_theme = "sphinxdoc"
# html_theme_options = {}
html_static_path = ["_static"]
# html_sidebars = {}
htmlhelp_basename = "optimusdocs"

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

latex_documents = [
    (master_doc, "optimus.tex", "OptimUS documentation", "OptimUS team", "manual"),
]

man_pages = [(master_doc, "optimus", "OptimUS documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "OptimUS",
        "OptimUS documentation",
        author,
        "OptimUS team",
    ),
]


epub_title = project

# epub_identifier = ''

# epub_uid = ''

epub_exclude_files = ["search.html"]

# -- Extension configuration -------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../optimus"]
autoapi_root = "doc"
