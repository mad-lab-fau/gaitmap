.. _algorithm_evaluation:

===============================================
Algorithm Evaluation and Parameter Optimization
===============================================

Whenever we want to apply an algorithm in a new scenario (e.g. different sensors, different patient cohorts, different
movement patterns, ...), it is important to perform an evaluation study during which ground truth information is
captured (e.g. motion capture, manual labeling, ...).
Based on this ground truth data we can estimate the performance of our algorithms on future unlabeled data.
Further, we can use such a labeled dataset to optimize parameters of our algorithms (this could be training a
machine-learning model or just optimizing a threshold).
However, when doing so, we need to make sure that we follow correct procedure to not introduce biases during this
optimization step, which could lead to an overly optimistic performance prospect and poor generalization on future data.
In the following we will explain how to perform such parameter optimization and evaluation correctly using the example
of gait analysis algorithms.
In particular, we will discuss when and how to use cross-validation.

While this concept of evaluation and labeled data (and the related naming) is heavily influenced by machine learning,
the concepts discussed here apply, or rather need to be applied, for all "non-machine-learning" algorithms as well.
In the following we will provide guides for specific "groups" of algorithms, depending on whether they contain
data-driven learning or not.

Categories of Algorithms
========================

In the following we will introduce 3 different categories of algorithms, we will need to treat differently.
These categories differentiate between algorithms with and without *trainable* components, with and without
*hyperparameter*, and with and without regular *parameter*.

trainable
    We consider all algorithms trainable that have components that can be directly learned from (labeled) data.
    This includes all algorithms that we would consider machine learning (e.g. a decision tree) or that use data
    directly, for example to create a template.
    We will call the output of such a training step a *model*.
    We specifically do not describe algorithms as *trainable* that just have parameters (like a decision threshold) that
    can only be optimized in a brute-force way (e.g. via grid search).
hyper-parameters
    Most algorithms that are trainable also have *hyper-parameters*.
    These are parameters that govern or influence the training aspect of the algorithm.
    This means a change in hyper-parameters will result in a different trained model from the same data.
    Usually only brute-force methods exist to find the optimal parameter set.
    Note, that parameters related to steps like pre-processing of the data can also be considered *hyper-parameters*.
    Adjusting them changes the training data and hence, the model.
parameters
    All remaining variable parts of a model (i.e. values that can not be trained, or do not influence the training) are
    simply called *parameters*.
    Like hyper-parameters, we assume that only brute-force methods exist to optimize these values.

With these definitions three distinct groups of algorithms emerge:

Group 1
    Traditional algorithms without any data-trained aspects fall into this group.
    Hence, this group has just *parameters*.
    An example for this is :class:`~gaitmap.event_detection.RamppEventDetection`.
    It has a couple of adjustable parameters exposed in the init, but none of them can be "trained".
    Note, that we also consider algorithms to be in *Group 1*, if they contain trainable components, but we are not
    modifying them in the context of a certain evaluation/parameter optimization.
    For example, :class:`~gaitmap.stride_segmentation.BarthDtw` contains a template that can be learned from data and
    a set of parameters that influence how the template (after it was learned) is applied to new data.
    This means, we would usually consider this algorithm *Group 3* (see below).
    However, if we are not planning to change the template (i.e. the model), but use a pretrained one, we only have
    traditional *parameters* left to optimize and you should follow the guidelines for *Group 1* algorithms.
Group 2
    We consider all traditional machine learning algorithms to belong to this group.
    These algorithms only have *hyper-parameters* and a *trainable* model, but no normal *parameters*.
    Thanks to the ongoing hype for machine learning you can find various resources online that explain how to evaluate
    and optimize such algorithms (for example the
    `sklearn documentation <https://scikit-learn.org/stable/model_selection.html>`__).
    Therefore, we will only briefly discuss this group in the following.
Group 3
    This group consists of all algorithms that are *trainable* (and have *hyper-parameters*), but have additional
    *parameters* that can not be learned and that do not influence the model.
    These are often more complex algorithms that consist of multiple steps.
    :class:`~gaitmap.stride_segmentation.BarthDtw` is again a good example.
    It has a trainable aspect (the generation and matching of a template) and afterwards a peak detection and post
    processing governed by a set of non-trainable parameters.
    These parameters are not considered hyper-parameters, as they do not influence the model (the result of the
    trainable part).
    On the other hand, variables modifying the pre-processing that we would apply to the data before template generation
    are hyper-parameters, as they change the template/model.
    We will see later, why this is an important distinction.


Evaluating an Algorithm (Part 1)
================================

.. note::
    If you don't want to optimize parameters or train your algorithm, you don't need this guide.
    Just apply your algorithm to your data and report the result.
    However, as soon as you start to make changes to your algorithms, because you are not happy with the performance you
    saw, you are optimizing parameters, and should make sure that you follow the procedures explained here.

Independent of which group your algorithm belongs to, you must never evaluate its performance on the **same** data you trained/
optimized parameters on.
Otherwise, we have a *train-test* leak that results in a far too optimistic evaluation of your algorithm and the
performance of the algorithm in the "real world" (aka on unlabeled data in the future) will be much worse than you
initially assumed.
This means we need to split our labeled data into a *train* set that we can freely use for training and parameter
optimization and a *test* set, which we **never** touch except when we want to perform the final evaluation of our
algorithm.
In other words, the only data we and our algorithms see during development and training is the *train* data.
We must treat our test data as if it doesn't even exist.

This means when we discuss parameter optimization in the next section, we will only work on the *train* data.
We will come back to our *test* set in :ref:`eval_algorithms_2`.
In this second part, we will also discuss more sophisticated methods than a simple train-test split.
But for now we will assume we just sectioned of a small part of our labeled data as a *test* set and take the remaining
*train* set into the parameter optimization.

What are we evaluating?
-----------------------
This might seem like a stupid question, but it is important to understand that in the context of algorithms that have
parts that can be trained or optimized, we do not evaluate a specific instance of an algorithm (algorithm + model +
parameter set), but rather our *entire approach* (algorithm + training method + parameter optimization method).
Yes, the final performance parameter, that we get when applying our optimized algorithm to the *test* data, tells us how
good a specific instance of our algorithm is.
However, because we trained and optimized this instance using our overall approach, it also tells us how good we can
expect another instance of our algorithm to be when it is trained with the same approach on different (but similar)
data.


Algorithm Optimization
======================

As mentioned in the previous chapter, algorithm optimization and training must only be performed on the *train* data in
the context of the evaluation.
However, it is important to understand that algorithm optimization/training also exists outside the context of
evaluation.
To facilitate this idea, we will just use the term "data" and not "*train*-set" in this section to refer to all the data
"we are allowed to use".

.. warning::
    In the context of evaluation, the term "*data*" in this chapter should be read as "*train set*".
    And if the text tells you to use **all** your data, I mean that you should use all the data you want to use for
    training and optimization.
    If you are currently performing an algorithm evaluation, this means your *train data*.
    **Don't touch your test data!**

What are we optimizing?
-----------------------
Like with evaluation, we need to understand what we are actually optimizing.
For evaluation we concluded that we evaluated an *approach* that contains the algorithm and the methods used for
algorithm optimization.
This chapter now describes the potential approaches for algorithm optimization methods.
So as the name suggests, we are actually optimizing the algorithm or rather its adjustable components.
These can be weights describing a machine learning model or neural network, parameters like thresholds or cutoffs that
are used in other parts of the algorithm, and hyper-parameters that change how our weights are adjusted based on our
data during training.
We want to set all of them to values that lead to the best possible performance on our *test* data or our future data
in production.
However, as we don't have access to our *test* data or labeled future data during our optimization, we need to choose
the values to provide the best performance on the data we have available and then hope that this also results in good
performance on our unseen data.
This usually requires some measures to prevent
`overfitting <https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html?highlight=overfitting>`__.

With that in mind let's discuss how we would optimize the weights and parameters of the algorithms in our different
groups.

Group 2 - Simple Machine Learning
-----------------------------------

We start with *Group 2* as this might be the group you are already most familiar with and which has the most amount of
information available in the internet, lectures, and books.

Without Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For traditional ML-models we need to differentiate cases with hyper-parameter optimization from cases without.
We will start with the easier one:
*Any* machine learning algorithm must be trained.
Aka, the weights that make up its internal model need to be set to values that fit your data.
Each algorithm has a different approach on how this should be done most efficiently.
Luckily, tools like scikit-learn, and also trainable algorithms in gaitmap, provide sufficient abstractions, so that
you just need to call the correct method with your training data as parameter.

With Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In basically all cases it is advisable to adjust available hyper-parameters of your model.
They influence the training and control, for example, how well your model will be able to fit to your data or prevent
your model from overfitting.
However, as no intrinsic method exists to optimize these parameters (in most cases), we usually have to try different
value combinations and see how well they perform.
Instead of doing that by hand - or as someone on twitter called it: "graduate student decent" -, we usually use
brute-force methods (grid search, random search, ...) to test out various parameter combinations.
In any case, we need to train a model with each of the parameter combinations, then calculate a performance value for
each of them and select the best one.

This raises the question, which data should we use to calculate these performance values?
We can not use the *test* set in the context of evaluation.
As we established, this is complete off-limits.
However, we can also not use the same data we used for training.
Otherwise, we would promote models that heavily overfit.
The solution is to split off an additional part of the data that we can use for performance evaluation during
hyper-parameter optimization.
We call this the *validation* set to avoid confusion with the *test* set.

With that we can use the following workflow (represented as pseudo code): ::

    # Optimize hyper-parameter
    # We are using the term `inner_train_...` here to avoid confusion with the train set we established during
    # evaluation. However, the procedure remains the same, even if we are not in the context of an evaluation.
    #
    # Note: "..._gt" is the ground truth information in the respective part of the data.
    inner_train_data, inner_train_gt, validation_data, validation_gt = split_validation_set(data, ground_truth)

    best_parameter = None
    best_performance = 0

    for parameter in parameter_space:
        model = algorithm(parameter).train(inner_train_data, inner_train_gt)
        performance = evaluate(model.predict(validation_data), validation_gt)
        if performance > best_performance:
            best_parameter = parameter
            best_performance = performance

    # Retrain model with best parameters on all data
    final_model = algorithm(best_parameter).train(data, ground_truth)


Note, that after we optimized the hyper-parameters, we didn't just take the best available model, but just the best
hyper-parameters and then retrained the model on all the data we had available during optimization (aka all *train* data
if we perform a evaluation).
This ensures that our model can make use of as much data as possible.

This is actually a critical point.
In many situations, we don't have sufficient data available to create a *validation* set without risking that our
hyper-parameter optimization will heavily depend on which data ends up in the *validation* set.
As this split is usually performed randomly, we do not want to take the chance that our entire model fails, because of a
bad random split.
The solution for that (and actually the recommended way in general) is to use a
`cross-validation <https://scikit-learn.org/stable/modules/cross_validation.html>`__.
This allows us to use all (*train*) data during the hyper parameter optimization by creating multiple validation splits
and averaging over all of them.::

    # Optimize hyper-parameter
    best_parameter = None
    best_performance = 0

    for parameter in parameter_space:
        performance_over_folds = []
        for fold in range(n_cv_folds):
            inner_train_data, inner_train_gt, validation_data, validation_gt = get_cv_fold(data, ground_truth, fold)
            model = algorithm(parameter).train(inner_train_data, inner_train_gt)
            performance = evaluate(model.predict(validation_data), validation_gt)
            performance_over_folds.append(performance)

        mean_performance_over_folds = mean(performance_over_folds)
        if  mean_performance_over_folds > best_performance:
            best_parameter = parameter
            best_performance = mean_performance_over_folds


    # Retrain model with best parameters on all data
    final_model = algorithm(best_parameter).train(data, ground_truth)

Note, that we perform the exact same series of data-splits for each parameter combination and then calculate the
**average** performance over all folds for each parameter combination.
The combination with the best average performance can then be used to retrain our model.

.. note::
    When using Grid Search, we need to pick a metric we use for evaluation.
    This depends on your very specific application.
    But typical candidates are "accuracy", "F1-score", or the "Youden-index".
    We can also calculate a combination of multiple values, but we need to have a way to decide on the best overall
    result.

For further explanation and ways to implement that easily, see the
`sklearn guide <https://scikit-learn.org/stable/modules/grid_search.html>`__

Group 1 - Simple Algorithms
---------------------------

Algorithms that don't have any components that would be considered *trainable* require brute-force methods for
optimization.
This means we just try out all the different parameter combinations we want to have and pick the best one.
However, the same question as before arises: Which data should I use to get these performance parameters?

The (maybe surprising) answer is: All my data! We do not need to provide a separate validation set, if we do not have
a training step.

Let's understand why with two explanation approaches:

Let's look at our brute-force method as a black-box that provides an optimal set of parameters given some "training"
data.
If we now simply replace the word "parameters" with "weights" we actually described the concept of a machine
learning algorithm.
A poor one - yes.
But this means we can actually treat our simple algorithm analogous to a machine learning algorithm without
hyper-parameters.
And as we learned in the previous section for such algorithms, we (simply) try to minimize the loss on all available
data and hope that the solution generalizes well to unseen data.

A different way to understand, why we shouldn't use a validation set to perform a brute force optimization of a simple
algorithms, is to just try it and see what happens:
So let's consider the pseudo code for the simple validation split introduced above.
Because we don't have a train step, we directly `call` predict on the algorithm with the given parameter set.
Effectively, we just ignored the inner training set entirely.
This means, splitting off a *validation* set, is equivalent to simply throwing away data.
If we would perform a cross-validation, we wouldn't throw away any data, but in each fold, we would again ignore the
training data.
This means that the cross-validation would be equivalent to performing the grid search on batches of the data and
averaging at the end.
In case of a leave-one-out cross-validation (only a single sample/subject is in the train set each fold) this would lead
to the exact same result as a Grid Search on all of our data and otherwise the result would be slightly different
(because the mean of batch-means is not equal to the mean over all data), but not better in any way.

So, no validation set it is! And we can just perform a simple grid search: ::

    # Optimize parameters
    best_parameter = None
    best_performance = 0

    for parameter in parameter_space:
        performance = evaluate(algorithm(parameter).predict(data), ground_truth)
        if performance > best_performance:
            best_parameter = parameter
            best_performance = performance

    final_algorithm = algorithm(best_parameter)

But doesn't that lead to over-fitting? Maybe...
But, with a brute-force approach we can not do anything about it as we have no ways to optimize hyper-parameters or
apply regularisation to conquer over-fitting, like we could for *Group 2* algorithms.
The only thing we can do is to evaluate our approach thoroughly and see if the results are sufficient for our
application.

Group 3 - The Hybrid
--------------------
The remaining group describes complicated algorithms that are basically hybrids that have both trainable components and
parameters that do not effect training, but the final performance.
These systems can further have hyper-parameters.
However, we will see that this does not change our approach to parameter optimization.
But to build an understanding of the approach, let's start with an algorithm for which we do not want to optimize
hyper-parameters.

Without Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There are (or at least there were for me) two approaches that seem plausible at first glance:
First, we could consider the parameter optimization as part of the training.
This means, in the train step of our algorithm, we first train the actual model component and then, using the predicted
outcome on the same data, we use a simple grid search to optimize the remaining parameters.
The second approach would be to ignore that our parameters do not effect training and treat them identical to
hyper-parameters.
This means, we would apply the same strategies we applied to trainable algorithms with hyper-parameter optimization.

Let's analyze these approaches:
In the first case, it seems plausible that we can separate the training of the model component and the optimization of
remaining parameters.
The model is fix and we can not accidentally make the model more prone to over- or underfit by selecting parameters.
However, in this approach we might overfit the parameter optimization.
Consider the following example:
We use :class:`~gaitmap.stride_segmentation.BarthDtw` and first "train" a new template based on our data.
Then we want to optimize the matching threshold in a second step.
For this we would need to generate potential stride matches using different threshold values on the data we already
trained the template on.
We expect the match between the template and the data to be pretty good, and hence, the required matching threshold
will likely be small.
In result our optimization selects a relatively small threshold.
However, on our real data (or *test* data) the template matches less well.
Therefore, a higher threshold would be required to find all strides.
Aka, our selected algorithm parameters could not generalize well to unseen data, because we overfitted the threshold
optimization.
This means, our first approach is not suitable, unless we are sure that our model will perform equally well on the
*train* and the *test* data.

However, the second approach appears to be working.
If we treat normal parameters as hyper-parameters and take out a separate *validation* set, we will train our algorithm
on the inner-train set and then test all different combinations on the *validation* set.
This way, we perform the parameter optimization based on the model output on unseen data, which should resemble
real-world performance.

We could actually reuse the pseudo code shown above (for both the *validation* set and the cross-validation version),
however, this would result in multiple useless calls to `train` method of our algorithm.
As our parameters do not effect the training, it is sufficient to call `train` only once per CV fold and then only
perform the grid search on the *validation* data.
A elegant solution to this, could be to cache the calls to `train` internally in the model to avoid repeated
calculations.
To gain further performance, the part of the `predict` method that just belongs to the model could also be cached.

With Hyper-Parameter Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
If we now reintroduce hyper-parameter, we can stick to the same approach.
However, we have to repeat the training for all hyper-parameter combinations.
All combinations of the regular parameters can then again be tested based on a cached model.

Group 3 is kind of strange ...
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
When reading this, you might ask yourselves, which algorithms (besides template matching), could actually be considered
*Group 3*.
The special requirement is that it needs to be possible to train the trainable components of the algorithm, even though
we only have ground truth available for the final output.
But there are a couple of algorithms that fulfil this requirement!
In general, all algorithms where the *trainable* part of the algorithm doesn't get us all the way, and post processing
steps are required to calculate the final output, can be considered *Group 3*.
For example, this could be stride detection algorithms that only provide stride candidates that then are selected based
on further criteria.
If these criteria have adjustable parameters, we have a *Group3* situation..
This could be machine learning algorithms for trajectory estimation that need smoothing of the final output.
In both cases, we can train just the machine learning part on the provided training data.
But we never expect the algorithm to perfectly reproduce that ideal output.
Hence, post processing steps are required that can for various reasons not directly be implemented in the model itself.

But, in the grand scheme of things, it is right to say that few algorithms fall in into *Group 3*, as many do not
fulfill the requirement stated above.
For most complicated and chained algorithms, we need to produce ground truth for the intermediate outputs so that parts
of the algorithm can be trained/optimized independently.
But even in these cases, we should follow the optimization concept outlined above.
If we first train our model based on the intermediate ground truth and then optimize the parameters of the remaining
steps independently, but on the same data pool, we still risk over-fitting these parameters as explained above.
Simply, we expect the output of our model to be better on our training data than on unseen data.
This means we optimize our remaining parameters based on this “best case” output (btw. the same would be true if we
use the intermediate ground truth as input for the parameter optimization).
Depending on the exact model and algorithm, this might not generalize well to unseen data, where the output of the model
component is less ideal.
Therefore, it would be safer to tune these parameters based on the prediction on unseen data, as shown above in the
cross-validation approach.

.. _eval_algorithms_2:

Evaluating an Algorithm (Part 2)
================================

Now that we have learned about optimization and training of the algorithm we can get back to evaluating our algorithms
and optimization approaches.
Our general approach we introduced so far is as follows:

1. Split the data into train and test data
2. Perform parameter optimization and algorithm training on the *train* data
3. Apply the optimized algorithm instance to the *test* data and calculate performance parameters.

Note that this is a **general** approach and Step 2 can be substituted with any of the optimization approaches we
learned about above.

In general this way of evaluating is totally fine, but, like with the *validation* set before, if we do not have enough
data, our reported performance might depend on the random split we performed in step 1.
In return, our approach could actually be better or worse on real world data.
Therefore, to get a more robust (but in general slightly pessimistic) performance estimate, we can repeat the evaluation
multiple times with different train-test splits.
Aka, we perform a cross-validation.
This simply means, we repeat our evaluation workflow in a loop while changing out the train and test split in step 1
in each iteration.
As final performance we will report the mean over all cross-validation folds.

.. note::
    When using a cross-validation to evaluate a trainable algorithm with additional (hyper-)parameters, we basically
    perform a cross-validation within a cross-validation.
    This is often called a *nested cross-validation*.
    I think, this is a bad term to describe the process.
    It makes it seem that we "apply" a nested cross-validation to an algorithm.
    But, what we are actually doing is "Using a cross-validation to evaluate an algorithmic approach that uses cross
    validation for (hyper-)parameter tuning".
    Explaining it that way, it is clearer that one cross-validation is an integral part of our approach (even outside
    the concept of evaluation) and the other one is added to perform the evaluation.

Group 1 Caching
---------------
To aid understanding the similarities between the algorithm types, we described the Grid Search for *Group 1* equivalent
to training a *Group 2* algorithm.
However, if we break open that black-box abstraction we can increase the cross-validation performance for *Group 1*
algorithms quite dramatically.

Where training *Group 2* and *Group 3* algorithms requires complicated interplay of all the provided training data,
the grid search to optimize a *Group 1* algorithm only involves calculating the performance for each training sample
(e.g. one gait test or one patient) and then averaging the performance to get the estimate for each parameter
combination.

If we perform the training multiple times during cross-validation, we have a large overlap between the trainings data of
the folds.
This means we need to calculate the performance for the same trainings sample over and over again.
Therefore, it can be very helpful to cache the output of the prediction step.
This basically speeds up a k-fold cross validation by a factor of k-1.

Alternatively to caching, you could also precalculate the performance of each sample in your dataset for all parameter
combinations you want to test.
Then you simply average over different parts of the calculated results in the different cross-validation folds.

.. _putting_everything_together:

Putting everything together
===========================

In real life applications, evaluating the performance of an algorithmic approach is not where things end.
Usually, our overarching goal is to create an algorithms instance (a "production model") that can be used on future
unseen (and unlabeled) data to serve our application.
Evaluation is there to tell us if our approach will be good enough for the application we want to accomplish in the
real world.

But how do we generate this final algorithms instance?
Remember, that our evaluation actually evaluates our **approach** and not an individual algorithm instance.
This means we can confidently use our approach to optimize and train an algorithm instance on **all** labeled
data we have (train and test data), **without** performing any final evaluation.
This seems a little bit risky, but if our prior evaluation was thorough, we can expect our final algorithmic instance
to be at least as good as what we saw during the evaluation process.

Before we go through the final workflow to do that, let's step back for a second and answer the question, why we would
want to do that.
During the evaluation we already created multiple algorithmic instances (one per cross-validation fold) that ideally
were already very capable.
Why shouldn't we use one of these models in production?
The issue with each of these models is that it was trained only on a portion of all available labeled data we have,
as we needed to hold back some data for testing in each fold.
In general, we assume that more data for training/optimization will always lead to better performance.
This means, we would expect each of the algorithmic instances we created during cross-validation to generalize a little
bit worse on unseen data than an algorithmic instance that was created with all the labeled data we have available.
Because of this, it is usually better to create your final model by repeating your training and optimization on your
**entire** dataset, even though you can not provide any performance parameters for this final model.

So with all of that in mind, our full workflow (from idea to production model) would look like that:

1. Do some experimenting with data and the algorithm to get a better understanding on potential pitfalls and narrow down
   parameter ranges for potential brute-force optimization.
   In an ideal world, you would do that *after* you put some data away, which could serve as some sort of "ultimate"
   test set, which would even be free from human biases.
   But, in reality this is rarely done...
   To learn more about this dilemma of biasing yourself and consequently over-fitting the hyper-parameters see for
   example `this paper <https://www.sciencedirect.com/science/article/abs/pii/S0149763420305868>`__.
   The authors suggest the term "over-hyping" to refer to any issues related to hyper-parameter over-fitting.
2. Evaluate your approach using cross-validation.
   In each fold you run your *entire* approach including your chosen method for parameter optimization and/or training.
3. Take the average performance result from your cross-validation and decide if the results are good enough for your
   application.
   If yes, continue with Step 4.
   If not, go back to the drawing board.
   Again: in an ideal world, you should not reuse your dataset, as you now have seen the performance results on the test
   sets.
   If you then start to change/optimize your approach based on this knowledge you risk "over-fitting" because of your
   human bias.
   But again, this is unrealistic in the real world.
4. Create a production model by taking **all** your available labeled data.
   Use the production model for all future unlabeled data.
   In the real world, it is usually advisable to check if future data points are still "in-distribution" (e.g. from
   the same patient population, measured in the same context, ...).
   If they are not, you would need to obtain labeled data for the new use case and run through the evaluation again and
   potentially adapt your production model.
   Further, it is common to update your production model once further labeled data becomes available.
   This makes sure that your production model always performs as good as possible.
   It is also common practice to define certain benchmark datasets that can be used to track your changes in performance
   over multiple iterations of your approach or your production model.
   In the first case you would perform the entire described evaluation process on these benchmark datasets.
   In the latter case, the benchmark dataset is used as "ultimate" test set that is excluded from training, even in the
   production model.
   This becomes feasible if you have a lot of data and tracking your performance in production is important.

Summary Table
=============

.. warning::
    Read the entire guide before using this table!

Here is a quick summary of how to implement the full approach for each of our algorithm Groups:

Group 1
    Optimization
        - Grid Search
    Evaluation
        - cross-validation were you perform a grid search in each fold and select an optimal parameter set per fold.
          **This is different from `GridSearchCV` in `sklearn`!**.
          Note, that the results for each individual train sample can be cached so that you effectively only have to
          calculate a "single fold"
Group 2 (without hyper-parameters)
    Optimization
        - Algorithm specific training
    Evaluation
        - cross-validation were you apply the algorithm specific training to each fold of train data.
Group 2 (with hyper-parameters)
    Optimization
        - Grid Search with an embedded cross-validation for hyper-parameter optimization.
          You obtain one performance value per parameter combination and fold.
          Pick the parameter combination that has the best *average* performance over all folds.
          **This is `GridSearchCV` in `sklearn`.**
        - Take the best parameter combination and retrain on all the optimization data with the algorithm specific
          training method.
    Evaluation
        - cross-validation, where you perform the entire optimization and retrain step per fold.
Group 3
    Optimization
        - Identical to Group 2, but with optional caching to speed up the process.
    Evaluation
        - Identical to Group 2

For all of these approaches you can retrain/reoptimize on all of your data to generate your production model.

A note on ...
=============

... brute-force methods
-----------------------
In this guide we used brute-force methods basically synonymously with Grid Search.
This is not entirely correct.
There exist multiple approaches to "just trying out multiple parameter combinations".
There are random search methods, adaptive grid search methods, and methods like Bayes Optimization, that can be much
faster than naive Grid Search in certain cases.
All of these methods are suitable substitutes to Grid Search and can be used in the same way.

Further, we sometimes just say that parameters can only be optimized by brute-force methods, because we are lazy and do
not want to do the math.
If you want the best results for one of your parameters (in particular in the *Group 1* methods), think about if it is
possible to calculate a gradient over your entire algorithm.
Basically, can you find a mathematical formulation for the question "How does my performance parameter change if I
make a small change of my parameter?".
If this appears feasible, you can use traditional optimization methods to find the optimal parameter values.
Tools that can automatically calculate gradients over complicated functions (like
`jax <https://github.com/google/jax>`__) can help with that.

... cross-validation
--------------------
In this guide we used cross-validation whenever we performed an evaluation multiple times, because we feared that a
single *train*-*validation/test* split might be too unstable.
However, we did that in two very different scenarios with two different purposes:

First, we learned to use cross-validation during parameter optimization.
In this use-case, cross-validation actively prevents over-fitting of **hyper-parameters** (not the actual model btw.).
By performing the hyper-parameter search on multiple validation-test splits, we ensure that our hyper-parameter set
would work well on multiple random subsets and not overfit to the validation data of a single split.

The other context in which we learned about cross-validation is evaluation.
In this scenario cross-validation can give us a slightly pessimistic generalisation performance.
Note, that it does **not actively prevent** over-fitting, as we must not modify our model based on the outcome of the
evaluation.
However, the evaluation can provide us insights in how well our approach will generalize.

To summarize: In the first use-case cross-validation is actually a way to improve our model.
It is just a "trick" we can use to prevent over-fitting of the hyper-parameters ("overhyping").
If we are confident that our model would work without it, we could just stick to a single validation-test split.
This would just be another approach to model training - maybe not optimal - but still correct.
This might be something you want to do, if training is extremely expensive (= high computation time).

If we use cross-validation for evaluation, it is used to give us an estimate of the actual real world performance.
A single train-test split will often not represent this performance well (unless our dataset is really large).
This means, not performing a cross-validation for evaluation could actually be considered a methodological error
and you should be highly skeptical of performance results produced on a single train-test split.

The other important thing to note about cross-validation is that different types of cross-validation exist and that
there are other algorithms that could fulfill the same function as cross-validation.
Depending on your data and your application, other methods (like repeated random splits) might be
better than simple cross-validation.
Such methods can be used equivalently to cross-validation in the context of this guide.

... computation time
--------------------
Using cross-validation and grid search requires our algorithms to be trained over and over again (sometimes even on the
same data).
This is expensive and can take a loooooooong time.
The reality is that real-life constraints on computational power sometimes prevent us to follow all the "ideal world"
guidelines.
In particular in the deep learning community where datasets are large and training times are long, cross-validation is
often substituted with a single train-test split and - instead of grid search - parameters are often optimized based on
experience.
While this might be less robust, or even might lead to accidental train-test leaks in the hyper-parameter selection due
to human biases, it is better than not being able to do an experiment at all.
This should absolutely not incentivize you to do the same, if you are annoyed by your computer needing to work for
5 min, but it should simply show you that this guide assumes an "ideal world", which you can not always expect.
