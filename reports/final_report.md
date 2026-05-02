# Final Report: Cross-lingual Sense Transfer with Low-Resource Indic Languages

## Dataset
- Total raw pairs: 627
- Validated sample size: 126
- Distinct target words: 44

## Model Comparison
| model     |   threshold |   validation_macro_f1 |   validation_accuracy |   test_accuracy |   test_macro_f1 |   test_pairs |   validation_pairs |
|:----------|------------:|----------------------:|----------------------:|----------------:|----------------:|-------------:|-------------------:|
| xlmr      |        0.85 |              0.373506 |              0.468254 |        0.448692 |        0.338299 |          497 |                126 |
| mbert     |        0.77 |              0.531717 |              0.531746 |        0.539235 |        0.53863  |          497 |                126 |
| muril     |        0.1  |              0.318919 |              0.468254 |        0.468813 |        0.319178 |          497 |                126 |
| indicbert |        0.9  |              0.492573 |              0.515873 |        0.529175 |        0.489222 |          497 |                126 |

## Clustering
| model     | target_word   |        ari |
|:----------|:--------------|-----------:|
| indicbert | ಅರ್ಥ          |  0.339623  |
| indicbert | ಕಬ್ಬಿ         |  0.444444  |
| indicbert | ಕರಣ           | -0.0606061 |
| indicbert | ಕಾಲ           |  0.0425532 |
| indicbert | ಕಾಲೆ          | -0.296296  |
| indicbert | ಕಿರಿಯ         |  0.166667  |
| indicbert | ಕೂಲಿ          |  0.444444  |
| indicbert | ಕೊಡು          |  0         |
| indicbert | ಗಟ್ಟ          |  0.0740741 |
| indicbert | ಗುರುತ್ವ       |  0.242424  |
| indicbert | ಚಕ್ರ          |  0.339623  |
| indicbert | ಚಿತ್ರ         |  0         |
| indicbert | ಜಲ            |  0.0740741 |
| indicbert | ತೆಗೆ          |  0         |
| indicbert | ಧರಣಿ          |  0.0740741 |
| indicbert | ಧರಿಸು         | -0.0606061 |
| indicbert | ನಕಲಿ          | -0.296296  |
| indicbert | ನೀಡ           | -0.0606061 |
| indicbert | ನೀಲ           | -0.296296  |
| indicbert | ನೇತೃತ್ವ       | -0.296296  |
| indicbert | ನೋಡು          |  0         |
| indicbert | ಪುಲ           |  0.444444  |
| indicbert | ಪೆರೆಲೆ        | -0.25      |
| indicbert | ಪ್ರಭು         |  0.166667  |
| indicbert | ಪ್ರೀತಿ        | -0.296296  |
| indicbert | ಬರು           |  0         |
| indicbert | ಬಾಗ           |  0.0740741 |
| indicbert | ಬೀಜ           | -0.0606061 |
| indicbert | ಬೆಳೆ          | -0.0606061 |
| indicbert | ಬೋಳ           |  0.0816327 |
| indicbert | ಭಕ್ತ          |  0.0740741 |
| indicbert | ಮಾಡು          |  0.04      |
| indicbert | ಮಾತು          | -0.103448  |
| indicbert | ಮುಕ್ತ         | -0.296296  |
| indicbert | ರಸ            |  0.0689655 |
| indicbert | ಲೇ            |  0.0740741 |
| indicbert | ಲೋಭಿ          |  0         |
| indicbert | ವಹಿಸು         | -0.296296  |
| indicbert | ಸಾಕು          |  0.0740741 |
| indicbert | ಸಾಧಾರಣ        | -0.5       |
| indicbert | ಹಸ್ತ          | -0.0833333 |
| indicbert | ಹಿರಿಯ         |  0         |
| indicbert | ಹೊಡೆ          |  0.166667  |
| indicbert | ಹೋಗು          | -0.0769231 |
| mbert     | ಅರ್ಥ          | -0.188679  |
| mbert     | ಕಬ್ಬಿ         | -0.0606061 |
| mbert     | ಕರಣ           | -0.25      |
| mbert     | ಕಾಲ           |  0.376238  |
| mbert     | ಕಾಲೆ          | -0.296296  |
| mbert     | ಕಿರಿಯ         |  0.0740741 |
| mbert     | ಕೂಲಿ          |  0.0740741 |
| mbert     | ಕೊಡು          |  0.222222  |
| mbert     | ಗಟ್ಟ          | -0.0606061 |
| mbert     | ಗುರುತ್ವ       |  0.444444  |
| mbert     | ಚಕ್ರ          |  0.339623  |
| mbert     | ಚಿತ್ರ         | -0.206897  |
| mbert     | ಜಲ            |  0.242424  |
| mbert     | ತೆಗೆ          |  0         |
| mbert     | ಧರಣಿ          | -0.0606061 |
| mbert     | ಧರಿಸು         |  0.444444  |
| mbert     | ನಕಲಿ          |  0.444444  |
| mbert     | ನೀಡ           |  0.444444  |
| mbert     | ನೀಲ           |  0.0740741 |
| mbert     | ನೇತೃತ್ವ       |  0.0740741 |
| mbert     | ನೋಡು          |  0.222222  |
| mbert     | ಪುಲ           | -0.0606061 |
| mbert     | ಪೆರೆಲೆ        |  0.0740741 |
| mbert     | ಪ್ರಭು         |  0.242424  |
| mbert     | ಪ್ರೀತಿ        | -0.0606061 |
| mbert     | ಬರು           |  0         |
| mbert     | ಬಾಗ           |  0.0740741 |
| mbert     | ಬೀಜ           |  0.444444  |
| mbert     | ಬೆಳೆ          | -0.0606061 |
| mbert     | ಬೋಳ           |  0.347826  |
| mbert     | ಭಕ್ತ          | -0.0606061 |
| mbert     | ಮಾಡು          |  0.04      |
| mbert     | ಮಾತು          |  0.0344828 |
| mbert     | ಮುಕ್ತ         | -0.296296  |
| mbert     | ರಸ            |  0.019802  |
| mbert     | ಲೇ            |  0.0740741 |
| mbert     | ಲೋಭಿ          |  0         |
| mbert     | ವಹಿಸು         | -0.296296  |
| mbert     | ಸಾಕು          |  0.444444  |
| mbert     | ಸಾಧಾರಣ        | -0.5       |
| mbert     | ಹಸ್ತ          | -0.0833333 |
| mbert     | ಹಿರಿಯ         | -0.5       |
| mbert     | ಹೊಡೆ          | -0.0606061 |
| mbert     | ಹೋಗು          | -0.0769231 |
| muril     | ಅರ್ಥ          |  0.339623  |
| muril     | ಕಬ್ಬಿ         | -0.0606061 |
| muril     | ಕರಣ           |  0.0740741 |
| muril     | ಕಾಲ           |  0.425532  |
| muril     | ಕಾಲೆ          | -0.0606061 |
| muril     | ಕಿರಿಯ         |  0.444444  |
| muril     | ಕೂಲಿ          | -0.296296  |
| muril     | ಕೊಡು          |  0         |
| muril     | ಗಟ್ಟ          | -0.0606061 |
| muril     | ಗುರುತ್ವ       |  0.242424  |
| muril     | ಚಕ್ರ          |  0.275862  |
| muril     | ಚಿತ್ರ         |  0         |
| muril     | ಜಲ            |  0.242424  |
| muril     | ತೆಗೆ          |  0         |
| muril     | ಧರಣಿ          | -0.0606061 |
| muril     | ಧರಿಸು         | -0.0606061 |
| muril     | ನಕಲಿ          | -0.296296  |
| muril     | ನೀಡ           | -0.0606061 |
| muril     | ನೀಲ           | -0.0606061 |
| muril     | ನೇತೃತ್ವ       | -0.0606061 |
| muril     | ನೋಡು          |  0         |
| muril     | ಪುಲ           |  0.444444  |
| muril     | ಪೆರೆಲೆ        | -0.0606061 |
| muril     | ಪ್ರಭು         | -0.0606061 |
| muril     | ಪ್ರೀತಿ        |  0.242424  |
| muril     | ಬರು           | -0.188679  |
| muril     | ಬಾಗ           |  0.0740741 |
| muril     | ಬೀಜ           |  0.444444  |
| muril     | ಬೆಳೆ          |  0.444444  |
| muril     | ಬೋಳ           |  0.347826  |
| muril     | ಭಕ್ತ          | -0.296296  |
| muril     | ಮಾಡು          |  0.04      |
| muril     | ಮಾತು          |  0.304348  |
| muril     | ಮುಕ್ತ         | -0.296296  |
| muril     | ರಸ            |  0.166667  |
| muril     | ಲೇ            |  0.444444  |
| muril     | ಲೋಭಿ          |  1         |
| muril     | ವಹಿಸು         | -0.296296  |
| muril     | ಸಾಕು          |  0.444444  |
| muril     | ಸಾಧಾರಣ        |  1         |
| muril     | ಹಸ್ತ          | -0.03125   |
| muril     | ಹಿರಿಯ         |  0         |
| muril     | ಹೊಡೆ          | -0.0606061 |
| muril     | ಹೋಗು          | -0.0769231 |
| xlmr      | ಅರ್ಥ          |  0.275862  |
| xlmr      | ಕಬ್ಬಿ         | -0.0606061 |
| xlmr      | ಕರಣ           |  0.0740741 |
| xlmr      | ಕಾಲ           |  0.425532  |
| xlmr      | ಕಾಲೆ          | -0.25      |
| xlmr      | ಕಿರಿಯ         |  0.444444  |
| xlmr      | ಕೂಲಿ          | -0.296296  |
| xlmr      | ಕೊಡು          |  0.222222  |
| xlmr      | ಗಟ್ಟ          | -0.296296  |
| xlmr      | ಗುರುತ್ವ       |  0.444444  |
| xlmr      | ಚಕ್ರ          |  0.603774  |
| xlmr      | ಚಿತ್ರ         |  0         |
| xlmr      | ಜಲ            |  0.0740741 |
| xlmr      | ತೆಗೆ          |  0         |
| xlmr      | ಧರಣಿ          | -0.25      |
| xlmr      | ಧರಿಸು         |  0.0740741 |
| xlmr      | ನಕಲಿ          |  0.0740741 |
| xlmr      | ನೀಡ           |  0.0740741 |
| xlmr      | ನೀಲ           |  0.0740741 |
| xlmr      | ನೇತೃತ್ವ       | -0.296296  |
| xlmr      | ನೋಡು          |  0.339623  |
| xlmr      | ಪುಲ           |  0.444444  |
| xlmr      | ಪೆರೆಲೆ        | -0.296296  |
| xlmr      | ಪ್ರಭು         |  0.0740741 |
| xlmr      | ಪ್ರೀತಿ        |  0.444444  |
| xlmr      | ಬರು           | -0.188679  |
| xlmr      | ಬಾಗ           |  0.0740741 |
| xlmr      | ಬೀಜ           |  0.0740741 |
| xlmr      | ಬೆಳೆ          | -0.296296  |
| xlmr      | ಬೋಳ           | -0.153846  |
| xlmr      | ಭಕ್ತ          |  0.444444  |
| xlmr      | ಮಾಡು          |  0.325     |
| xlmr      | ಮಾತು          |  0.0344828 |
| xlmr      | ಮುಕ್ತ         | -0.296296  |
| xlmr      | ರಸ            |  0.114754  |
| xlmr      | ಲೇ            |  0.0740741 |
| xlmr      | ಲೋಭಿ          |  1         |
| xlmr      | ವಹಿಸು         | -0.296296  |
| xlmr      | ಸಾಕು          |  0.444444  |
| xlmr      | ಸಾಧಾರಣ        |  0         |
| xlmr      | ಹಸ್ತ          | -0.0588235 |
| xlmr      | ಹಿರಿಯ         |  0         |
| xlmr      | ಹೊಡೆ          |  0.0740741 |
| xlmr      | ಹೋಗು          | -0.188679  |

## Gloss Baseline
| model     |      mrr |   n |
|:----------|---------:|----:|
| xlmr      | 0.570125 | 294 |
| mbert     | 0.605329 | 294 |
| muril     | 0.619048 | 294 |
| indicbert | 0.592347 | 294 |

## Notes
- The gloss baseline uses a doc-derived gloss inventory because no local IndoWordNet export was present.
- The pipeline keeps preprocessing identical across the four encoders for a fair comparison.
- Phase outputs are stored under /data, /outputs, /results, and /reports.