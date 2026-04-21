# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Project source code
import os
import sys

# Make sure src is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# AutoAPI config
autoapi_type = "python"
autoapi_dirs = [os.path.abspath(os.path.join("..", "..", "src"))]

autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

# Optional but nice:
autoapi_root = "autoapi"
autoapi_add_toctree_entry = False   # If false, must link it manually from index.rst
autoapi_keep_files = True           # keep generated .rst files in docs/source/autoapi
autoapi_own_page_level = "function"

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aPriori'
copyright = 'Copyright © 2026, Lorenzo Piu, Heinz Pitsch, Alessandro Parente'
author = 'Lorenzo Piu'
release = '1.3.1'
html_title = "aPriori Documentation"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/Numpy style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    # "myst_parser",                # if myst_nb is active, comment this line to avoid conflicts.
    # "myst_nb",                    # Jupyter notebook tutorials
    "sphinx_autodoc_typehints",
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "sphinx_design",
]

autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Allow markdown via MyST
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "smartquotes",
]

templates_path = ['_templates']
exclude_patterns = []

# Option for myst_nb
# nb_execution_mode = "auto"  # "off" I don't know what it does, or "auto" if you want the notebooks to run at build time

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme settings
# html_theme = 'furo' 
html_theme = "shibuya"
html_static_path = ['_static']
html_css_files = [
    "custom.css",
]

html_theme_options = {
    # already included probably:
    "accent_color": "auto",
    "light_logo": "_static/figures/logo/aPriori-logo-flame.png",
    "dark_logo": "_static/figures/logo/aPriori-logo-flame.png",
    "accent_color": "orange",
    # These are the additional links in the navigation bar
    "nav_links": [
        {
            "title": "aPriori Documentation       ",
            "url": "index"
        },
        {
            "title": "Related Projects",
            "url": "none",
            "children": [
                {
                    "title": "BLASTNet",
                    "url": "https://blastnet.github.io",
                    "summary": "Bearable Large Accessible Scientific Training Network-of-Datasets",
                    "external": True
                },
                {
                    "title": "PyCSP",
                    "url": "https://github.com/rmalpica/PyCSP",
                    "summary": "A collection of tools based on Computational Singular Perturbation for the analysis of chemically reacting systems.",
                    "external": True
                },
            ]
        },
        {
            "title": "Publications",
            "url": "publications"
        },
        {
            "title": "Cite",
            "url": "how_to_cite"
        },
        {
            "title": "Contribute",
            "url": "contribute"
        },
    ],
    "github_url": "https://github.com/LorenzoPiu/aPrioriDNS",
    "linkedin_url": "https://www.linkedin.com/in/lorenzopiu/",
}

html_context = {
    "source_type": "github",
    "source_user": "LorenzoPiu",
    "source_repo": "aPriori",
    "source_version": "main",
    "source_docs_path": "/docs/source/",
}

bibtex_bibfiles = ["references.bib"]

html_favicon = "_static/figures/logo/aPriori-logo-flame.png"
