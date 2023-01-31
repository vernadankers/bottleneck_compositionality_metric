# Bottleneck Compositionality Metric (BCM)

### Section 4: Arithmetic evaluation

Launch all required model training using the `tree_lstms/scripts/submit_train_arithmetic.sh <SETUP>` script.
- Launch it twice, using `bcm_pp` and `bcm_tt` as setups.
- The latter can only be started if the former finished running.
- Afterwards, reproduce the graphs contained in the paper in section 4 using the `analysis/arithmetic/visualise.ipynb` notebook.

### Section 5.1-5.3: Sentiment analysis

Launch all required model training using the `tree_lstms/scripts/submit_train_sentiment.sh <SETUP>` script.
- Launch it twice, using `bcm_pp` and `bcm_tt` as setups.
- The latter can only be started if the former finished running.

Afterwards, reproduce the graphs contained in the paper using the following notebooks:
1. Figure 5: `analysis/sentiment/visualise_task_performance.ipynb`
2. Figure 6: you can obtain the predictions from `analysis/sentiment/run_baseline.py`. The figure was created by hand afterwards.
3. Figure 7: `analysis/sentiment/compare_to_baseline.ipynb`

### Section 5.4: Sentiment analysis, example use cases

Launch all required model training using the `sentiment_training/scripts/submit_train.sh` script.
Afterwards, reproduce the graphs contained in the paper using the `analysis/sentiment/visualise_task_performance.ipynb` notebook.

#### Appendix B.2:
First obtain:
1. Topographic similarity: `python topographic_similarity.py`
2. Memorization: memorization values of Zhang et al., that extracts memorization values as demonstrated in their notebook https://github.com/xszheng2020/memorization/blob/master/sst/05_Atypical_Phase.ipynb

Then visualise Figures 15 and 16 with the `analysis/sentiment/alternative_metrics.ipynb` notebook.
