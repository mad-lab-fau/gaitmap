# Compatibility policy for the constrained DTW issue 93 fix

- Preserve the public DTW parameters and result formats for this fix.
- Update affected internal helpers, in-repository consumers, and tests together.
- Corrected constrained DTW results may change; document the numerical behavior changes and their observed effect on example recordings.
