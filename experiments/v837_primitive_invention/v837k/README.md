# V837k ? Training-budget calibration

Conditional follow-up to V837j. The only scientific variable is AdamW optimizer-step budget. Architectures, parameter counts, task generators, data episode IDs, optimizer type/hyperparameters, and capacity gate remain fixed. V837j supplies the 1x condition; this variant runs 2x and only runs 4x if 2x does not establish >=4/5 reference learnability.
