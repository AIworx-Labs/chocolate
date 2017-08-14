Library Reference
=================

.. toctree::
   :hidden:
   :maxdepth: 2

   space
   connection
   sample
   search
   conditional
   crossvalidation
   multiobjective

.. currentmodule:: chocolate

.. rubric:: :doc:`space`

.. autosummary::
   :nosignatures:

   Space
   Constant
   Distribution
   ContinuousDistribution
   QuantizedDistribution
   uniform
   quantized_uniform
   log
   quantized_log
   choice

.. rubric:: :doc:`connection`

.. autosummary::
   :nosignatures:

   SQLiteConnection
   MongoDBConnection
   DataFrameConnection

.. rubric:: :doc:`sample`

.. autosummary::
   :nosignatures:

   Grid
   Random
   QuasiRandom

.. rubric:: :doc:`search`

.. autosummary::
   :nosignatures:

   Bayes
   CMAES
   MOCMAES

.. rubric:: :doc:`conditional`

.. autosummary::
   :nosignatures:

   ThompsonSampling

.. rubric:: :doc:`crossvalidation`

.. autosummary::
   :nosignatures:

   Repeat

.. rubric:: :doc:`multiobjective`

.. autosummary::
   :nosignatures:

   mo.argsortNondominated
   mo.hypervolume_indicator
   mo.hypervolume