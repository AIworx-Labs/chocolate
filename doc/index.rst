Chocolate Documentation
=======================

Chocolate is a completely asynchronous optimisation framework relying solely on a
database to share information between workers. Chocolate uses no master process for
distributing tasks. Every task is completely independent and only gets its
information from the database. Chocolate is thus ideal in controlled computing
environments where it is hard to maintain a master process for the duration
of the optimisation.

Chocolate has been designed and optimized for hyperparameter optimization where
each function evaluation takes very long to complete and is difficult to parallelize.

Chocolate is licensed under the `3-Clause BSD License
<https://opensource.org/licenses/BSD-3-Clause>`_

* **Tutorials**

  * :doc:`Basics (Start here!) <tutorials/basics>`
  * :doc:`Something a bit more realistic <tutorials/sklearn>`
  * :doc:`Optimizing multiple models at once <tutorials/multimodel>`
  * :doc:`Optimizing multiple losses at once <tutorials/multiobjective>`
  * :doc:`Let's go to Tensor Flow <tutorials/tf>`
  * :doc:`tutorials/results`
  * :doc:`How to choose your algorithm <tutorials/algo>`
  * :doc:`Cross-validating the results <tutorials/crossvalidation>`

* :doc:`installation`
* :doc:`api/index`
* :doc:`releases`
* :doc:`about`

.. toctree::
  :hidden:

  tutorials/basics
  tutorials/sklearn
  tutorials/multimodel
  tutorials/multiobjective
  tutorials/tf
  tutorials/results
  tutorials/algo
  tutorials/crossvalidation
  installation
  api/index
  releases
  about
