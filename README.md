# Meganet.jl

A fresh approach to deep learning written in Julia. 

Build status: [![Build Status](https://travis-ci.org/XtractOpen/Meganet.jl.svg?branch=master)](https://travis-ci.org/XtractOpen/Meganet.jl)

Code coverage: [![Coverage Status](https://coveralls.io/repos/github/XtractOpen/Meganet.jl/badge.svg?branch=master)](https://coveralls.io/github/XtractOpen/Meganet.jl?branch=master)

## Installation

In Julia, type the following to install the package:
```
Pkg.clone("https://github.com/XtractOpen/Meganet.jl.git")
```

## Editing the Source Code
To properly edit the Meganet source code in Julia, use the following steps:
```
Pkg.checkout("Meganet")           # check out the master branch
<here, make sure your bug is still a bug and hasn't been fixed already>
cd(Pkg.dir("Meganet"))
;git checkout -b myfixes         # create a branch for your changes
<edit code>                      # be sure to add a test for your bug
Pkg.test("Meganet")               # make sure everything works now
;git commit -a -m "Fix foo by calling bar"   # write a descriptive message
using PkgDev
PkgDev.submit("Meganet")
```
(As per https://docs.julialang.org/en/stable/manual/packages/#Package-Development-1)

This will present you with a link to submit a pull request to incorporate your changes. The changes will then be reviewed prior to being merged with the stable package.
