# Coding Challenges

Complete these extensions after running `python examples/run_practice.py`.

## Beginner

1. **Probability rules:** Change the spam example's prior probability and observe how Bayes' theorem changes the posterior.
2. **Expectation:** Create a discrete random variable with at least four outcomes. Check that its probabilities sum to 1, then calculate its expectation, variance, and standard deviation.
3. **Z-score:** Use `z_score()` to compare observations from two Gaussian distributions with different means and standard deviations.
4. **MLE:** Change the coin-toss sequence. Does the Bernoulli MLE always equal the observed proportion of heads?

## Intermediate

5. **Central Limit Theorem:** Replace the exponential population with a uniform or strongly skewed population. Compare the sampling-mean histograms for sample sizes 1, 5, 20, and 50. State what changes and what does not.
6. **Confidence intervals:** Compare the Wald and Wilson intervals for 0/20, 10/20, and 19/20 successes. Explain why the Wald method may behave poorly near 0 or 1.
7. **Hypothesis tests:** Change the null proportion from 0.50 to 0.55. Recalculate the z statistic and p-value. Explain the difference between the p-value and the probability that the null hypothesis is true.
8. **Classification threshold:** Convert probability scores into labels using thresholds 0.3, 0.5, and 0.7. Recalculate Precision and Recall at each threshold.

## Challenge

9. **Metric selection:** Invent a scenario where false negatives are much more costly than false positives. Explain which metrics you would monitor and why.
10. **Calibration experiment:** Group predictions into probability bins and compare each bin's mean predicted probability with its observed positive rate.

Write your answers and observations in a short report. Use fixed random seeds when comparing simulations so the experiments are reproducible.
