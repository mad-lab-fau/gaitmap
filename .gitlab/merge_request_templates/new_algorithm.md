# New Algorithm: <The algorithm name here>

/label ~new-algorithm
/target_branch master
/wip

**TODO**: Thank you for adding a new algorithm!
Please go through all todos in this template (and delete the todo text afterwards).
Then follow the [*current state*](#current-state) checklist.
If you have questions at any point, please ask!

**TODO**: Add short description of the feature and why it is important

**TODO**: List related issues and other resources

## Current State

- [ ] Merge request open and marked as WIP
- [ ] Merge request linked to related issues, tagged correctly, and added to milestones
- [ ] All features complete
- [ ] All "Self-Checks" marked as complete
- [ ] Review requested by assigning the issue to the reviewer

## New Features

**TODO**: *List the features that are/need to be implemented as a checkbox list here*


### Self Checks

All boxes should be marked before requesting the final review.
If a check is not relevant for your feature, mark it as complete.
If any of *optional* points are not completed, they can be ticked **after** a *TODO* is added to the code or a new issue
is open, clearly explaining what needs to be done in the future. 

### Documentation

- [ ] All parameters and attributes of public functions/classes/methods are documented properly
- [ ] A very simple example exists in the docstring of all important classes of functions
- [ ] A reference to the paper of the algorithm exists in its docstring
- [ ] Functions/classes are embedded in the Sphinx documentation
- [ ] The Sphinx documentation of the new functions/classes renders properly
- [ ] (optional) Information about the algorithm are added to other parts of the documentation if appropriate
- [ ] (optional) Additional information about the algorithm is provided in the Notes section
- [ ] (optional) At least one example exists that uses the algorithm

### Tests

- [ ] Unit tests for simple parameter handling exist
- [ ] Unit tests for all supported input and output types exist
- [ ] Unit tests for obvious edge cases exist
- [ ] Unit tests for specific math implementations exist
- [ ] Regression tests for common parameter combinations on actual data exist (should store the output)
- [ ] (optional) Implementation tests for the connections with other algorithms exist

## For Reviewer

**The following boxes should only be ticked by the reviewer**

As reviewer, please check the following items below.
If any of *optional* points are not completed, they can be ticked if a *TODO* is added to the code or a issue exists, 
that clearly explains what needs to be done in the future. 
If in doubt, ask for help.
If changes are required add comments to this issue and the code diff.
You can make changes to the branch yourself.
However, **communicate this with other people working on the branch**.

Once all issues are resolved:

- resolve WIP status
- rebase the branch onto master (if required)
- merge + delete the branch

### Implementation

- [ ] contains all listed features
- [ ] follows the contribution guidelines
- [ ] is at a sensible place in the library
- [ ] uses the proper base classes and global helpers (no duplicated implementations)
- [ ] (optional) has usable performance (or can not be further improved easily)

### Documentation

- [ ] All parameters and attributes of public functions/classes/methods are documented properly
- [ ] A very simple example exists in the docstring of all important classes of functions
- [ ] A reference to the paper of the algorithm exists in its docstring
- [ ] Functions/classes are embedded in the Sphinx documentation
- [ ] The Sphinx documentation of the new functions/classes renders properly
- [ ] (optional) Information about the algorithm are added to other parts of the documentation if appropriate
- [ ] (optional) Additional information about the algorithm is provided in the Notes section
- [ ] (optional) At least one example exists that uses the algorithm

### Tests

- [ ] Unit tests for simple parameter handling exist
- [ ] Unit tests for all supported input and output types exist
- [ ] Unit tests for obvious edge cases exist
- [ ] Unit tests for specific math implementations exist
- [ ] Regression tests for common parameter combinations on actual data exist (should store the output)
- [ ] (optional) Implementation tests for the connections with other algorithms exist
