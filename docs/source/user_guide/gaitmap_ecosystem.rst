The Gaitmap Ecosystem
=====================

.. figure:: /images/gaitmap_ecosystem.svg
    :alt: Overview over all relevant packages in the Gaitmap ecosystem (tpcp, gaitmap, gaitmap-datasets, gaitmap-bench,
          gaitmap-challenges)
    :figclass: align-center

+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| Package                                                                                 | Description                                                                           |
+=========================================================================================+=======================================================================================+
| `gaitmap <https://github.com/mad-lab-fau/gaitmap>`_                                     | The core package of the ecosystem, providing the gait analysis algorithms.            |
+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| `gaitmap-datasets <https://github.com/mad-lab-fau/gaitmap-datasets>`_                   | A collection of dataset loaders for existing open source gait datasets.               |
+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| `gaitmap-challenges <https://github.com/mad-lab-fau/gaitmap-bench>`_                    | A collection of pre-defined challenges that can be used to evaluate gait analysis     |
|                                                                                         | algorithms.                                                                           |
+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| `gaitmap-bench <https://gaitmap-bench.readthedocs.io/en/latest/challenges/index.html>`_ | Reference results and visualisations for all gaitmap algorithms and other algorithms  |
|                                                                                         | to compare performances.                                                              |
+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+
| `tpcp <https://github.com/mad-lab-fau/tpcp>`_                                           | A domain-agnostic framework for building pipelines, performing parameter optimization,|
|                                                                                         | and evaluating the performance of your pipelines.                                     |
|                                                                                         | It forms the foundation for all gaitmap algorithms and challenges.                    |
+-----------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------+


Gaitmap implements easy to use and dataset-independent gait analysis algorithms.
While this on its own can be useful for researchers, to actually make using these algorithms easy and support building
applications on top of that, we need to provide a lot more.
A lot more, is where the gaitmap ecosystem comes in.
To better understand the need and motivation behind this, have a look at the
`gaitmap paper <https://www.techrxiv.org/articles/preprint/Gaitmap_-_An_Open_Ecosystem_for_IMU-based_Human_Gait_Analysis_and_Algorithm_Benchmarking/24047493>`_.

In the following we will provide a simplified and more practical overview over the what and why structured along two
core questions (you probably have asked yourself already):

How to actually use the algorithms?
-----------------------------------
Gaitmap provides examples on how to apply the algorithms to some data you loaded and tries to provide some guidance on
how you can prepare your data, and how to interpret the results.
However, typically you are planning to do a lot more.
You don't just want to run a single algorithms on a single gait recording.
You want to run an entire pipeline on an entire dataset.
And often you want to evaluate the performance of your pipeline against some reference data or even compare multiple
algorithms against each other.
And while you are doing that, you likely want to optimize the hyper-parameters of your pipeline or even train new
machine learning models specific to your dataset.

All of that is complex and requires experience far outside the realm of gait analysis.
If we want to improve the way, we conduct research and build applications in gait analysis, we need to simplify doing
these things.

As a general starting point for these types of problems, we developed the domain-agnostic
`tpcp <https://github.com/mad-lab-fau/tpcp>`_ package.
It provides a clear way of structuring and loading your data, building pipelines, performing parameter optimization,
and evaluating the performance of your pipelines.
Gaitmap is build on top of tpcp, so that all the gaitmap-algorithms can be directly used in tpcp pipelines and
tpcp-tools can be used to optimize and evaluate gaitmap pipelines.
In combination with the tooling provided in ``gaitmap.evaluation_utils``, this allows you to easily build and evaluate
gait analysis pipelines.

Because of the way tpcp abstracts the interfaces between datasets, pipelines, algorithms, and evaluation
(`learn more <https://tpcp.readthedocs.io/en/latest/guides/algorithms_pipelines_datasets.html#pipelines>`_), writing
proper code once, allows you to easily rerun a specific evaluation with a different algorithm or on a different dataset.
This allows us to provide further pre-defined components that can be used with your custom algorithms for your custom
application.

Namely, we provide a set of dataset-loaders with the
`gaitmap-datasets <https://mad-lab-fau.github.io/gaitmap-datasets/>`_ package that allow you to easily load existing
open source datasets, without worrying about the specific format they are stored in or other weird quirks of the
recordings.
These datasets can be used to prototype and validate your algorithms.

If you simply want to know, how good your custom algorithms are in a typical context, or want to understand what
happens if you change some parameters, we go a step further and provide pre-defined challenges that will give you
extensive evaluation metrics calculated on the datasets described above.
These are implemented in the `gaitmap-challenges <https://mad-lab-fau.github.io/gaitmap-bench/>`_ package.

How good are the provided algorithms?
-------------------------------------
If you want to build a new gait analysis application, you want to use the best algorithms available or at least be
confident that the algorithms you use are good enough.
The original papers of most of the algorithms we provide are already a good starting point for this.
However, the reality is that most papers only provide performance numbers in a very narrow context (i.e. only a single
sensor setup, only a single dataset, only a single evaluation metric, etc.).
Further, the actual performance does not just depend on the algorithm, but also on the implementation.
From experience, simply going from one programming language to another can change the outcome of an evaluation, as
specific fundamental functions are implemented differently.
On top of that, each new implementation will usually have its own set of bugs, specific implementation improvements, or
simply small enhancements that are not extensively described in the paper or published anywhere else.

Simply speaking, if you want to know how good the gaitmap algorithms are, we need to give you end-to-end reproducible
performance numbers on a wide range of datasets.
Ideally, you should have the possibility to look into the specific aspects of performance that are relevant for you,
or even perform a completely new evaluation with your own datasets/in your own application context.

Therefore, we not just provide the tooling to evaluate algorithms on your own, but we provide full reproducible
benchmark results for the main gaitmap algorithms on the
`gaitmap-bench website <https://gaitmap-bench.readthedocs.io/en/latest/challenges/index.html>`_.
The code to generate these results can be found in the accompanying
`github repo <https://github.com/mad-lab-fau/gaitmap-bench/tree/main/entries/gaitmap_algos>`_ .
On the website we try to visualize the most import performance parameters.
However, the raw results are also available for download, so that you can perform your own analysis.

In the future, we plan to further extend the website, by allowing other researchers to submit their own algorithms and
evaluations, allowing you to compare your algorithms to the state-of-the-art in gait analysis and pick the best
algorithm for your application.
