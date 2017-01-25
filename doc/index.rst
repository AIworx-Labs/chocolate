Chocolate Documentation
=======================

Chocolate is a completely asychrone optimisation framework relying solely on a
database to share information between workers. Chocolate has no master process
distributing tasks. Every task is completely independent and only gets its
information from the database. Chocolate is thus ideal in controlled computing
environments where it is hard to maintain a master process for the whole duration
of the optimisation.

Chocolate has been though for hyperparameter optimization where each function
evaluation takes very long to complete and is hardly parallelizable by
itself.

Chocolat is licensed under the `3-Clause BSD License
<https://opensource.org/licenses/BSD-3-Clause>`_

* **Tutorials**
  
  * :doc:`Basics (Start here!) <tutorials/basics>`
  * :doc:`Something a bit more realistic <tutorials/sklearn>`
  * :doc:`Optimizing multiple models at once <tutorials/multimodel>`
  * :doc:`Lets go to Tensor Flow <tutorials/tf>`
  * :doc:`For developpers <tutorials/dev>`


* :doc:`installation`
* :doc:`api/index`
* :doc:`releases`
* :doc:`about`

.. toctree::
  :hidden:

  tutorials/basics
  tutorials/sklearn
  tutorials/multimodel
  tutorials/tf
  tutorials/dev
  installation
  api/index
  releases
  about