# V837z frozen research specification

Question: does within-timestep sequential candidate staging explain the remaining representation failure?

- Z0: exact selected V837y Y3 historical mixed message schedule.
- Z1: at each timestep snapshot previous states and previous outputs; every edge reads previous outputs; compute all messages/candidates/states before commit.
- Parent mechanisms remain joint global scalar controller + rank-4 cross-block candidate coupling.
- 512 development episodes/family, 128 validation episodes/family, 5 families, 5 replicates, 192 AdamW steps.
- 3,200 unique family/seed episodes reused across both conditions/replicates.
- Pass: >=4/5 families.
- No fresh audit, structural search, primitive mining, or V838.
