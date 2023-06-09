Contribution Guide
==================

Issues and bugs
---------------
If you are using the library and you find a bug or want to suggest a feature, please report it on the `issue tracker <https://github.com/mad-lab-fau/gaitmap/issues>`_.
Please provide sufficient information (ideally reproducible code) to reproduce the bug that you have found.

We are always happy to help, but please understand that it can take some time to answer your questions!

We are also happy to accept pull requests for bug fixes and features (see below)!
However, before you start working on any major changes, please open an issue first to discuss the changes with us.

Scientific questions
--------------------
When working with the algorithms in this library, you might have questions about the underlying algorithms.
We are happy to answer these questions, but please understand that we cannot provide free consulting.

Ideally use `github discussions <https://github.com/mad-lab-fau/gaitmap/discussions/>`_ instead of the issue
tracker for these questions.

New algorithms and features
---------------------------
We hope our efforts in open-sourcing our own algorithms will motivate you to do the same!
If you have an algorithm that you would like to share with the community, we want to help you making it happen!

Depending on what work you already have done, we suggest the following steps:

I have an algorithm implemented in a way that is not directly compatible with this library
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
In case, your algorithm is implemented in a different programming language or in a different way, that does not fit
our algorithm structure (see :ref:`own_algorithm`), yet, we suggest the following steps:

1. Open an issue on the `issue tracker <https://github.com/mad-lab-fau/gaitmap/issues>`_ and describe the algorithm
   you want to contribute.
   Then we can discuss together how much work it would be to integrate your algorithm into this library and where we
   can help best.
2. In case you want to implement the algorithm yourself into a compatible format, we suggest to do that outside of
   the repository first (unless the implementation of the algorithm requires changes to the library itself).
   Check our guide on this: :ref:`own_algorithm`.
3. Once you have a working implementation, create a fork of the repository, and copy your algorithm into the appropriate
   folder.
   Then create a pull request to merge your changes into the main repository (see below for more details on this).
4. We will then review your changes and discuss with you any necessary changes.
   There will likely be some back and forth until we are happy with the changes, as we have relatively high requirements
   for the code quality, documentation and tests, to make sure the code-base remains maintainable.
   But, don't worry, we will help you with that!
5. Once we are happy with the changes, we will merge your pull request and your algorithm will be part of the library! 🎉🎉

I have an algorithm implemented in a way that is directly compatible with this library
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Awesome! You did the hard work already! Just follow the steps below:

1. Open an issue on the `issue tracker <https://github.com/mad-lab-fau/gaitmap/issues>`_ and describe the algorithm
   you want to contribute.
2. Once, we aggreed that the algorithm is a good fit for the library, create a fork of the repository, and copy your
   algorithm into the appropriate folder.
   Then create a pull request to merge your changes into the main repository (see below for more details on this).
3. We will then review your changes and discuss with you any necessary changes.
   There will likely be some back and forth until we are happy with the changes, as we have relatively high requirements
   for the code quality, documentation and tests, to make sure the code-base remains maintainable.
   But, don't worry, we will help you with that!
4. Once we are happy with the changes, we will merge your pull request and your algorithm will be part of the library! 🎉🎉

In both cases, if we don't think the algorithm is a good fit for the library itself, you can still publish it yourself
as a separate library that uses gaitmap as a dependency.
This way, you can still benefit from the infrastructure we provide, but you are not bound to our requirements and stay
in control of your own code.
In any case, let us know! We are happy to link your project in our documentation!

Developing Code for Gaitmap
---------------------------

After discussing with us, you are ready to start developing code for gaitmap.
First of all, we are happy to have you on board! 🎉🎉

To get started, you should follow these steps:

1. Fork the repository on github (button on the top right on the Github page).
   This will create your own copy of the repository, that you can work on.
2. Clone your fork to your local machine.
3. Set up your local environment and IDE (see :ref:`development guide <dev_guide>`).
4. Test your setup by running the tests once (`poetry run poe test`). They should all pass. If not, please let us know
   by opening an issue on Github.
5. Read through the guides on :ref:`project and algorithm structure <proj_struct>` and :ref:`development <dev_guide>`.
   If you don't understand something or find outdated information, please let us know by opening an issue on Github.
   We always strive to improve our documentation.
6. Create a new branch in your local repository for your changes (e.g. `git checkout -b my_new_feature`).
7. We recommend directly pushing the new branch (`git push -u origin my_new_feature`) to your fork on Github and to
   create a pull request (`click here <https://github.com/mad-lab-fau/gaitmap/compare>`_ and then select your branch in
   the right dropdown).
8. Start implementing your changes and push them regularly to your branch on Github.
   This way we can monitor your progress and help you if you get stuck.
9. Ask for a review **early**, even if your implementation is not finished yet.
   This way we can prevent you wasting time on implementing something that we don't want to merge.
   We prefer, multiple review cycles over a single big one at the end.

