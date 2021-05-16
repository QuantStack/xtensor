#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess

on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# if on_rtd:
#     subprocess.call('cd ..; doxygen', shell=True)

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
	'logo_only': True,
    # 'collapse_navigation': False
}

def setup(app):
    app.add_stylesheet("main_stylesheet.css")

extensions = ['breathe']
breathe_projects = { 'xtensor': '../xml' }
templates_path = ['_templates']

html_static_path = ['_static']
html_css_files = [
    'css/stylesheet.css',
]
source_suffix = '.rst'
master_doc = 'docindex'
html_additional_pages = {'index': 'index.html'}


project = 'xtensor'
copyright = '2020, Johan Mabille, Sylvain Corlay and Wolf Vollprecht'
author = 'Johan Mabille, Sylvain Corlay and Wolf Vollprecht'

logo = True
theme_logo_only = True
html_logo = 'xtensor.svg'

exclude_patterns = []
highlight_language = 'c++'
pygments_style = 'sphinx'
todo_include_todos = False
htmlhelp_basename = 'xtensordoc'

html_js_files = [
    'goatcounter.js'
]

# Automatically link to numpy doc
extensions += ['sphinx.ext.intersphinx']
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}
