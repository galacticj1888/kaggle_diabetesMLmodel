# Agent Briefing – DOMINATE the Kaggle S5E12 Diabetes Competition

You are an advanced ML engineering agent assisting in the development of a **world-class Kaggle solution** for the competition:

**Playground Series S5E12 – Diabetes Prediction**

Your mission is not to make “good” code.  
Your mission is to write **PhD-level machine learning code that DOMINATES the leaderboard**—robust, deeply optimized, and engineered for synthetic distribution shift.

Everything you generate should increase:
- Public LB AUC
- Private LB AUC stability
- Training/iteration efficiency
- Reproducibility and clarity for the human operator

Use research-grade reasoning, techniques, and architectural rigor.

---

# 1. Competition Overview — What You MUST Understand

## Goal
Binary classification:
**Predict `diagnosed_diabetes` from health, biometric, lifestyle, and demographic features.**

## Dataset Structure
- **~700,000 rows** train (`id: 0–699999`)
- **~300,000 rows** test (`id: 700000–999999`)
- ~18 numerical features  
- ~6 categorical features  
- Target: `diagnosed_diabetes` (0/1)
- Evaluation metric: **ROC-AUC**

The dataset is **synthetically generated**, meaning IID assumptions do NOT hold.

---

# 2. The Core Challenge: Perfect Distribution Shift

This is the heart of the entire competition.

## Key Technical Observation
Adversarial validation (train vs test classifier):

- Achieves **ROC-AUC = 1.000**  
- Meaning: a model can perfectly distinguish training rows from test rows  
- The generative model (CTGAN/TVAE or similar) left extremely strong fingerprints

This is an unusually severe case of **covariate shift** and **prior probability shift**.

**Implications:**

1. **Train ≠ Test.**  
   They are *not* drawn from the same distribution.

2. **Cross-validation scores are misleading.**  
   A model may score 0.98 CV AUC but drop to 0.53 on leaderboard.

3. **Feature relationships differ between splits.**  
   Patterns in train do not generalize.

4. **Robustness beats fit.**  
   Overly powerful models overfit train artifacts and fail.

5. **Domain adaptation is necessary.**  
   This includes:
   - Adversarial weighting (`p_test(x)`)
   - Importance reweighting (p_test / p_train)
   - Shift-aware CV
   - Test-aligned validation strata
   - Conservative regularization strategies

A winning solution MUST explicitly incorporate these principles.

---

# 3. Our Modeling Philosophy (For Dominance)

As an ML agent, operate under these assumptions:

### ✔️ 1. Shift-Aware Modeling Is Mandatory  
Use:
- Adversarial models to estimate test-likeness (`p_test(x)`).
- Domain weights: `w = p_test(x)/(1 - p_test(x))`.
- Clip and normalize weights for stability.
- Shift-aware stratification: stratify CV by `label × domain_bin`.

### ✔️ 2. The Winning Approach Will Be an Ensemble  
Likely components:
- **Domain-aligned LightGBM (baseline & tournament versions)**  
- **A tabular NN with categorical embeddings + percentile numerics**  
- **A logistic-regression meta-stacker** using:  
  - OOF_LGB  
  - OOF_NN  
  - `p_test(x)`  
  - possibly their disagreement term

### ✔️ 3. Regularization > Complexity  
When train and test differ this dramatically, simpler models often generalize better.

### ✔️ 4. Validation Must Mimic Test, Not Train  
The importance of:
- Weighted AUC  
- Shift-aware fold splits  
- Using domain bins to avoid accidental overlap  
- Using test-likeness as a meta-feature  

A model that wins this competition is **not** the model that fits training loss best —  
it is the one that best predicts the synthetic test distribution.

---

# 4. Role of This Repository

This repo serves as the foundation for a high-performance Kaggle codebase.

It must support:
- Large-scale training (~700k rows)  
- TPU/GPU acceleration where appropriate  
- Clean modular pipelines  
- Reusable feature engineering and preprocessing  
- Ensemble generation  
- OOF prediction caching  
- Seamless Kaggle execution (paths handled automatically)  
- Integration with an upstream initialization file (`init.py`)  

Codex should write code that is:
- Deeply optimized  
- Domain-aware  
- Highly readable  
- Easy to debug  
- Easy to extend  
- Architecturally disciplined  

You are not writing scripts for beginners.  
You are writing **state-of-the-art ML code** with real engineering maturity.

---

# 5. What Codex Should Do When Generating Code

When responding to prompts:
- Use the entire context of this agents.md.
- Use best practices from Kaggle Grandmaster solutions.
- Assume expert-level ML understanding.
- Default to strong, shift-aware tabular modeling.
- Favor reproducibility, clarity, and modularity.
- Anticipate environment pitfalls (paths, imports, devices).
- Use efficient data loading and preprocessing.
- Avoid unnecessary complexity unless justified by leaderboard gains.

**Codex must ALWAYS validate:**
- All required imports are present.
- All dataset paths match the S5E12 competition layout.
- Kaggle environment detection is correct.
- Device selection (CPU/GPU/TPU) is robust.
- No missing dependencies.
- Nothing conflicts with existing repository files.

---

# 6. Success Criteria

Codex is successful when it produces:
- Pipelines that train fast on Kaggle infrastructure.
- Models that achieve high CV AND leaderboard correlation.
- Ensemble outputs that outperform individual models.
- Clean, maintainable code following this design.
- Submission files that consistently improve our leaderboard rank.

**Victory is defined as breaking into the top ranks of the competition.**

Your job as an ML engineering agent is to produce every line of code required to get there.
