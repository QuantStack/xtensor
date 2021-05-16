.. Copyright (c) 2016, Johan Mabille, Sylvain Corlay and Wolf Vollprecht

   Distributed under the terms of the BSD 3-Clause License.

   The full license is in the file LICENSE, distributed with this software.

.. image:: xtensor.svg
   :alt: xtensor

Multi-dimensional arrays with broadcasting and lazy computing.

Introduction
------------

`xtensor` is a C++ library meant for numerical analysis with multi-dimensional
array expressions.

`xtensor` provides

- an extensible expression system enabling **lazy broadcasting**.
- an API following the idioms of the **C++ standard library**.
- tools to manipulate array expressions and build upon `xtensor`.

Containers of `xtensor` are inspired by `NumPy`_, the Python array programming
library. **Adaptors** for existing data structures to be plugged into the
expression system can easily be written.

In fact, `xtensor` can be used to **process numpy data structures in-place**
using Python's `buffer protocol`_. For more details on the numpy bindings,
check out the xtensor-python_ project. Language bindings for R and Julia are
also available.

`xtensor` requires a modern C++ compiler supporting C++14. The following C++
compilers are supported:

- On Windows platforms, Visual C++ 2015 Update 2, or more recent
- On Unix platforms, gcc 4.9 or a recent version of Clang

Licensing
---------

We use a shared copyright model that enables all contributors to maintain the
copyright on their contributions.

This software is licensed under the BSD-3-Clause license. See the LICENSE file
for details.


.. toctree::
   :caption: INSTALLATION
   :maxdepth: 1

   installation
   changelog

.. toctree::
   :caption: USAGE
   :maxdepth: 2

   getting_started
   expression
   container
   scalar
   adaptor
   operator
   histogram
   view
   indices
   builder
   missing
   file_loading
   build-options
   pitfall

.. toctree::
   :caption: QUICK REFERENCE
   :maxdepth: 2

   quickref/basic
   quickref/builder
   quickref/operator
   quickref/math
   quickref/reducer
   view

.. toctree::
   :caption: API REFERENCE
   :maxdepth: 2

   api/expression_index
   api/container_index
   api/function_index
   api/io_index
   api/xmath

.. toctree::
   :caption: DEVELOPER ZONE
   :maxdepth: 2

   compilers
   dev-build-options
   developer/xtensor_internals
   external-structures
   releasing

.. toctree::
   :caption: MISCELLANEOUS

   numpy
   numpy-differences
   closure-semantics
   related
   bindings

.. _NumPy: http://www.numpy.org
.. _Buffer Protocol: https://docs.python.org/3/c-api/buffer.html
.. _libdynd: http://libdynd.org
.. _xtensor-python: https://github.com/QuantStack/xtensor-python
