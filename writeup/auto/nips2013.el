(TeX-add-style-hook
 "nips2013"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amssymb" "psamsfonts") ("graphicx" "dvips" "pdftex")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "nips13submit_e"
    "times"
    "hyperref"
    "url"
    "amssymb"
    "graphicx")
   (TeX-add-symbols
    "fix"
    "new"
    "RR"
    "Nat"
    "CC")
   (LaTeX-add-labels
    "gen_inst"
    "headings"
    "others"
    "sample-table"))
 :latex)

