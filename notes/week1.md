# Week 1
---

## 1. Why Data Foundations Determine Machine Learning Success

Machine learning systems do not learn from the world directly; they learn from **data representations of the world**. Every conclusion a model produces—every prediction, classification, or discovered pattern—is constrained by how the data was collected, represented, processed, and interpreted. As a result, many failures attributed to “poor models” are, in reality, failures of **data foundations**.

At a conceptual level, machine learning is often described as the task of learning a mapping from inputs to outputs, or of discovering structure in observations. However, this framing can be misleading if it obscures the fact that both the inputs and outputs are **human-defined abstractions**. Data does not arrive in a neutral or objective form: it reflects design choices, measurement limitations, institutional processes, and social context. These choices shape what a model can learn—and what it cannot.

### 1.1 Data as the Primary Source of Assumptions

Every dataset encodes assumptions, whether explicitly or implicitly. These include assumptions about:

* what entities matter (e.g. individuals, events, transactions),
* what properties of those entities are observable,
* how those properties are measured,
* and what relationships are considered meaningful.

Machine learning models do not question these assumptions; they operationalise them. If a dataset measures customer behaviour only during office hours, a model trained on it implicitly assumes that off-hours behaviour is irrelevant. If a medical dataset excludes patients with incomplete records, the resulting model assumes that these patients are either unimportant or statistically similar to those retained. In this sense, **data precedes modelling as the dominant source of inductive bias**.

This is why two teams using the same algorithm can arrive at radically different conclusions when working with different datasets—or even different versions of the same dataset. The algorithm amplifies structure that already exists in the data; it does not create structure ex nihilo.

### 1.2 “Garbage In, Garbage Out” Is a Pipeline Problem

The phrase “garbage in, garbage out” is often invoked to dismiss poor results as inevitable consequences of bad data. At MSc level, this slogan is too simplistic. Poor outcomes are rarely caused by a single catastrophic flaw; instead, they emerge from **a sequence of small, locally reasonable decisions** made throughout the data pipeline.

For example:

* removing rows with missing values may appear sensible in isolation,
* standardising variables may be appropriate for certain models,
* deriving features may improve apparent performance on historical data.

Individually, none of these steps is necessarily wrong. Collectively, however, they may introduce bias, distort representativeness, or leak information from the future into the present. The resulting model may perform well under evaluation yet fail when deployed. Importantly, this failure is not a modelling failure—it is a **pipeline failure**.

Understanding machine learning therefore requires shifting focus away from isolated techniques and toward **end-to-end reasoning**: how data moves from real-world processes into analytical form, and how each transformation constrains valid inference.

### 1.3 Models Are Constrained by What the Data Makes Visible

A machine learning model can only learn patterns that are **encoded in the data it receives**. If a relevant variable is unmeasured, poorly measured, or systematically missing, the model cannot recover its influence through clever optimisation or increased complexity. More sophisticated models do not compensate for missing information; they often obscure its absence.

This limitation has practical consequences. In predictive settings, models trained on historical data may fail when the underlying process changes (for example, policy changes, behavioural adaptation, or external shocks). In exploratory settings, unsupervised methods may identify structure that reflects data collection artefacts rather than meaningful phenomena. In both cases, the issue lies not in the algorithm, but in **the relationship between data and reality**.

Recognising this constraint early is essential. It encourages practitioners to ask:

* What does this dataset allow me to observe?
* What does it systematically hide?
* Under what conditions would conclusions drawn from this data cease to be valid?

### 1.4 Why This Perspective Matters for the Rest of the Module

The remainder of this module introduces a range of learning paradigms and techniques, from supervised classification to probabilistic modelling and unsupervised structure discovery. Each of these methods relies on the same foundational premise: that the data provided is an appropriate representation of the problem being studied.

By beginning with data foundations, we establish a unifying principle for all subsequent weeks: **algorithms are tools, but data is the substrate**. Mastery of machine learning at postgraduate level is therefore not defined by the ability to run models, but by the ability to reason critically about the data pipelines that make those models meaningful—or misleading.

---

**Common misconception (Section 1)**
*“Better algorithms can compensate for weak data.”*
In practice, more complex models often magnify data problems rather than resolve them, producing results that appear confident but rest on fragile or invalid assumptions.

---

*End of Section 1.*

---

Below is **Section 2** of the Week 1 canonical notes. This section introduces **data types and representation** with **carefully constrained Python examples** that illustrate *conceptual issues*, not modelling techniques.

---

## 2. Data Types and Representation

Before any learning can occur, real-world phenomena must be translated into **data representations**. This translation is neither neutral nor automatic. Choices about data types, units, granularity, and structure determine what operations are valid, what assumptions are reasonable, and what errors are even detectable. At MSc level, understanding data types is not about memorising categories; it is about recognising how representation shapes inference.

### 2.1 Practical Data Types in Machine Learning Contexts

In applied machine learning, data types are best understood in terms of **what operations they meaningfully support**, rather than how they are stored in software.

**Numeric data** represents quantities where arithmetic operations are meaningful. This includes both continuous measurements (e.g. temperature, income) and discrete counts (e.g. number of purchases). Although both are numeric, they often carry different assumptions: averaging a count may be sensible, while averaging an identifier is not.

**Categorical data** represents membership in a set of labels. Nominal categories (e.g. country, colour) have no intrinsic order, whereas ordinal categories (e.g. education level, satisfaction rating) encode relative ranking but not necessarily equal spacing. Treating nominal categories as ordered introduces artificial structure; treating ordinal categories as purely nominal discards useful information.

**Temporal data** records when observations occur. Time introduces ordering, duration, and dependency. A timestamp is not merely a number: subtracting two timestamps yields a duration with meaning, while averaging timestamps rarely does.

**Text and other unstructured data** do not directly support arithmetic or ordering. From a machine learning perspective, these are not “difficult types” but **representation problems**: the challenge lies in converting them into structured features without losing meaning or introducing bias. In Week 1, we acknowledge their existence without operationalising them.

### 2.2 Representation Is More Than Storage Type

Software libraries often expose data types through storage-level abstractions (e.g. integers, floats, strings). These are **implementation details**, not semantic guarantees. A column stored as an integer may represent a count, a category code, or an identifier—each with radically different implications.

The key question is not *how* data is stored, but *what it means*. This distinction is critical because machine learning algorithms operate on numerical representations regardless of whether arithmetic operations are semantically valid.

#### Illustrative Python example: storage type vs meaning

```python
import pandas as pd

df = pd.DataFrame({
    "age": [23, 45, 31],
    "customer_id": [1023, 1045, 1098],
    "satisfaction_score": [1, 2, 3]
})

df.info()
```

All three columns may appear as numeric types. However:

* `age` supports arithmetic comparisons and averaging,
* `customer_id` is an identifier where arithmetic is meaningless,
* `satisfaction_score` is ordinal: comparisons make sense, but differences may not.

A model cannot distinguish these interpretations on its own. If representation does not encode meaning, the model will impose structure where none exists.

### 2.3 Units, Ranges, and Validity

Data validity depends on **contextual constraints**. Numeric values are only meaningful within acceptable ranges and units. Violations often signal data quality issues, not rare phenomena.

For example, a negative age or a percentage above 100 is not an “outlier” in the statistical sense—it is an invalid measurement. Treating such values as legitimate observations can distort downstream analysis.

#### Illustrative Python example: detecting semantic invalidity

```python
df = pd.DataFrame({
    "age": [25, -3, 140, 42]
})

df[df["age"] < 0]
```

The code detects a numeric condition, but the *interpretation*—that negative age is invalid—comes from domain reasoning, not computation. Machine learning workflows that skip this reasoning risk learning patterns from impossible data.

### 2.4 Granularity and the Unit of Analysis

Every dataset makes an implicit choice about **what one row represents**. This unit of analysis may be a person, a transaction, a time window, or an aggregated entity. Many analytical errors arise when this choice is unclear or inconsistent.

For example, combining per-day measurements with per-person attributes without careful alignment can introduce duplication, weighting some entities more heavily than others. Similarly, aggregating data may remove variability that is essential for certain learning tasks.

At MSc level, it is essential to ask:

* What real-world entity does one row correspond to?
* Is this consistent across all columns?
* Would changing the unit of analysis change the question being answered?

### 2.5 Structured Thinking: Rows, Columns, and Keys

A common conceptual model is to treat:

* rows as observational units,
* columns as variables describing those units,
* identifiers as keys that preserve meaning across joins and transformations.

Identifiers deserve special attention. They are necessary for linking data, but they should **never be treated as features** for learning. Including identifiers as numeric inputs invites models to learn spurious patterns that reflect ordering or assignment schemes rather than real relationships.

#### Illustrative Python example: identifiers masquerading as features

```python
df = pd.DataFrame({
    "customer_id": [1001, 1002, 1003],
    "total_spend": [250.0, 120.0, 310.0]
})

df.describe()
```

The summary statistics for `customer_id` are mathematically correct but semantically meaningless. This is a clear signal that representation choices must be audited before modelling.

---

**Common misconception (Section 2)**
*“If data has the right type in Python, it has the right meaning.”*
Storage types encode how data is held in memory, not what operations are conceptually valid. Semantic meaning must be imposed deliberately and checked continuously.

---

*End of Section 2.*

---

Below is **Section 3** of the Week 1 canonical notes. This section focuses on **data quality issues**, emphasising *reasoning and diagnosis* rather than automated fixes, with **minimal Python examples** used only to surface conceptual points.

---

## 3. Data Quality Issues: What Can Go Wrong and Why It Matters

Data quality problems are not peripheral annoyances; they fundamentally shape what machine learning systems can learn and how reliable their conclusions can be. At MSc level, data quality should be understood not as a checklist of fixes, but as a set of **analytical risks** that must be identified, interpreted, and managed in relation to the problem being studied.

A critical principle is that *the same apparent issue can have different meanings in different contexts*. An extreme value may indicate a measurement error in one domain and a rare but important phenomenon in another. Consequently, data quality assessment is inseparable from domain understanding and problem framing.

### 3.1 Missing Data: Patterns and Consequences

Missing data is often treated as a technical inconvenience, but it is more accurately viewed as **information about the data collection process**. The absence of a value can itself be informative.

At a high level, missingness can occur:

* **Sporadically**, due to random failures or noise in collection,
* **Systematically**, affecting specific groups, time periods, or conditions,
* **By design**, where values are undefined or intentionally unrecorded.

The consequences of missing data depend not just on how much is missing, but *what is missing and why*. Removing rows with missing values may simplify analysis, but it can also change the effective population being studied.

#### Illustrative Python example: inspecting missingness

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 34, None, 41],
    "income": [32000, None, 28000, None],
    "employed": [True, True, False, True]
})

df.isna().sum()
```

This summary reveals how many values are missing, but not *why*. If income is missing primarily for unemployed individuals, removing those rows would bias the dataset toward higher earners, even though the code itself performs no “incorrect” operation.

### 3.2 Noise, Outliers, and Anomalies

Noise refers to random variation introduced during measurement or recording. Outliers and anomalies are observations that deviate markedly from the majority of the data. These concepts are often conflated, but they are not equivalent.

An outlier may arise from:

* measurement error,
* data entry mistakes,
* genuine but rare events.

From a machine learning perspective, the key question is not whether a value is extreme, but whether it is **credible and relevant**. Automatically removing outliers can eliminate precisely the cases a model should learn to recognise, such as fraud, system failures, or medical complications.

#### Illustrative Python example: detecting extreme values

```python
df = pd.DataFrame({
    "daily_usage": [2.1, 2.4, 2.3, 45.0, 2.2]
})

df[df["daily_usage"] > 10]
```

The code flags an extreme value, but interpretation requires domain insight. Is this a sensor malfunction, or a legitimate high-usage event? Machine learning workflows that skip this question risk either learning noise or discarding signal.

### 3.3 Inconsistency and Duplication

Inconsistencies arise when the same concept is recorded in multiple incompatible ways. Duplication occurs when the same real-world entity or event appears more than once in the dataset. Both issues can silently distort analysis.

Duplicates are particularly dangerous because they effectively **re-weight observations**, giving some entities more influence than others. In supervised learning, this can bias class distributions; in unsupervised settings, it can exaggerate apparent structure.

#### Illustrative Python example: detecting duplicates

```python
df = pd.DataFrame({
    "transaction_id": [101, 102, 102, 103],
    "amount": [50, 75, 75, 20]
})

df.duplicated()
```

Whether duplicates should be removed depends on whether they represent repeated measurements, data entry errors, or legitimate repeated events. The decision cannot be made mechanically.

### 3.4 Data Leakage as a Quality Issue (Introductory View)

Data leakage occurs when information that would not be available at prediction time is inadvertently included in the dataset used for training or evaluation. While often discussed in the context of model evaluation, leakage is fundamentally a **data quality problem**: it reflects a mismatch between how data is constructed and how it will be used.

Leakage can enter a dataset through:

* features derived from future outcomes,
* aggregation over time windows that extend beyond the point of prediction,
* preprocessing steps applied before data splitting.

In Week 1, the goal is not to exhaustively categorise leakage, but to recognise it as a **threat to validity** that originates early in the pipeline.

#### Illustrative Python example: a subtle leakage risk

```python
df = pd.DataFrame({
    "application_date": ["2024-01-01", "2024-01-05"],
    "approval_date": ["2024-01-10", "2024-01-07"],
    "approved": [True, True]
})

df["processing_time"] = (
    pd.to_datetime(df["approval_date"]) -
    pd.to_datetime(df["application_date"])
).dt.days
```

If `processing_time` is used to predict approval, the feature directly encodes information derived from the outcome itself. The dataset appears valid, but the representation violates the intended temporal logic of prediction.

### 3.5 Why Data Quality Is a Reasoning Task, Not a Recipe

Many tools offer automated cleaning pipelines, anomaly detectors, and imputation strategies. While these can be useful, they do not replace the need for **explicit reasoning about data quality**. Automated fixes operate on patterns; they cannot assess whether those patterns reflect errors, bias, or genuine structure.

At MSc level, the emphasis is therefore on:

* identifying potential data quality risks,
* understanding their origins,
* evaluating their impact on inference,
* and documenting the rationale behind any corrective action.

---

**Common misconception (Section 3)**
*“Data quality issues can be fixed once and forgotten.”*
In practice, data quality concerns recur at every stage of the pipeline, and their impact depends on how the data will be used, not just on how it looks initially.

---

*End of Section 3.*

---

Below is **Section 4** of the Week 1 canonical notes. This section makes a **clear conceptual separation** between *cleaning*, *preprocessing*, and *feature engineering*, using **one running Python example** to show how the *same raw column* can be treated differently depending on intent.

---

## 4. Cleaning, Preprocessing, and Feature Engineering: Distinct but Connected Roles

In applied machine learning workflows, the terms *cleaning*, *preprocessing*, and *feature engineering* are often used interchangeably. At MSc level, this lack of precision is problematic. These activities serve **different purposes**, operate under **different assumptions**, and introduce **different risks**. Confusing them leads to fragile pipelines and invalid conclusions.

A useful way to distinguish these stages is to ask **why** a transformation is being performed, not **how** it is implemented.

### 4.1 Data Cleaning: Restoring Correctness and Validity

**Data cleaning** aims to ensure that the dataset faithfully represents what was intended to be measured. The goal is *correctness*, not optimisation for a particular model.

Typical cleaning actions include:

* fixing obvious data entry errors,
* standardising formats (e.g. dates, categorical labels),
* resolving duplicates,
* enforcing valid ranges and units.

Crucially, cleaning should not introduce new information or assumptions beyond what is already implicit in the data collection process.

#### Illustrative Python example: cleaning invalid values

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, -2, 41, 130],
    "gender": ["Male", "female", "FEMALE", "male"]
})

# Enforce validity
df.loc[df["age"] < 0, "age"] = None
df.loc[df["age"] > 120, "age"] = None

# Standardise categories
df["gender"] = df["gender"].str.lower()
```

Here, the actions reflect domain knowledge: negative or implausibly large ages are invalid, and inconsistent casing in categories is a formatting issue. No modelling assumptions are introduced.

### 4.2 Preprocessing: Making Data Model-Ready

**Preprocessing** transforms *valid* data into a form that algorithms can consume. Unlike cleaning, preprocessing is **model- and method-aware**, even if the specific model has not yet been chosen.

Common preprocessing operations include:

* encoding categorical variables,
* scaling or normalising numeric features,
* handling missing values in a systematic way.

Preprocessing changes the *representation* of data, but not its *meaning*. However, it can introduce assumptions—particularly when handling missing values—that must be made explicit.

#### Illustrative Python example: preprocessing the same cleaned data

```python
# Simple preprocessing choices (conceptual)
df_processed = df.copy()

# Encode gender as binary indicator (illustrative only)
df_processed["gender_female"] = (df_processed["gender"] == "female").astype(int)

# Impute missing ages with a simple statistic (mean, for illustration)
mean_age = df_processed["age"].mean()
df_processed["age"] = df_processed["age"].fillna(mean_age)
```

These steps are not “cleaning”: replacing missing ages with a mean does not restore truth—it **imposes an assumption** about what missing values represent. That assumption may or may not be appropriate, depending on context.

### 4.3 Feature Engineering: Creating Representations That Carry Meaning

**Feature engineering** creates new variables that aim to capture structure relevant to the task at hand. This is the most creative—and most dangerous—stage of the pipeline.

Feature engineering often uses:

* domain knowledge,
* aggregation,
* ratios, thresholds, or indicators,
* combinations of existing variables.

Unlike cleaning, feature engineering actively injects *interpretation* into the data. Unlike preprocessing, it can alter what information is available to the model.

#### Illustrative Python example: feature engineering from the same data

```python
# Feature engineering: age group indicator
df_features = df_processed.copy()

df_features["is_senior"] = (df_features["age"] >= 65).astype(int)
```

This new feature encodes a substantive assumption: that the distinction between “senior” and “non-senior” is meaningful for the problem. Whether this is appropriate depends entirely on the application context.

### 4.4 Comparing the Three Stages Side by Side

The same raw column (`age`) has now been:

* **cleaned** to remove invalid values,
* **preprocessed** to remove missingness in a model-compatible way,
* **engineered** into a higher-level indicator.

Each step answers a different question:

* Cleaning: *Is this value even plausible?*
* Preprocessing: *Can the model handle this representation?*
* Feature engineering: *Does this representation capture something meaningful?*

Treating all three as a single “data wrangling” step obscures these distinctions and makes reasoning about errors and bias significantly harder.

### 4.5 Risks and Responsibilities

Each stage introduces different risks:

* Cleaning can silently remove rare but important cases.
* Preprocessing can encode unjustified assumptions (especially around missingness).
* Feature engineering can introduce leakage or amplify bias.

For this reason, all three stages should be:

* documented explicitly,
* justified in relation to the problem,
* and revisited when results appear unexpectedly strong or weak.

At MSc level, **the quality of reasoning behind transformations matters more than the transformations themselves**.

---

**Common misconception (Section 4)**
*“Once data is cleaned and preprocessed, it is neutral and objective.”*
In reality, every transformation reflects choices and assumptions that shape what the model can learn and how its outputs should be interpreted.

---

*End of Section 4.*

---

Below is **Section 5** of the Week 1 canonical notes. This section treats **sampling and representativeness** as *foundational reasoning problems*, with Python examples used only to **surface bias**, not to estimate models or performance.

---

## 5. Sampling, Representativeness, and Bias

Machine learning systems are trained on **samples**, but deployed in **populations**. The validity of any conclusion therefore depends on how well the sample represents the population of interest. At MSc level, sampling should not be viewed as a preliminary technical step, but as a **central inferential decision** that determines what a model’s outputs can legitimately be said to mean.

A model trained on a biased or unrepresentative sample may perform well according to internal metrics, yet fail systematically when exposed to real-world data. Understanding this mismatch requires reasoning about *who or what is included*, *who or what is excluded*, and *why*.

### 5.1 Population vs Sample: What Is the Model Really For?

The **target population** is the set of entities for which predictions or insights are intended to apply. The **sample** is the subset of that population that happens to be observed and recorded.

Crucially, the target population is a conceptual object, not just “all rows in the dataset.” For example:

* A dataset of loan applicants may represent *only approved applications*.
* A dataset collected during business hours may exclude certain user behaviours.
* Historical data may reflect outdated policies or incentives.

Machine learning models implicitly assume that the sample reflects the population they will face in deployment. When this assumption fails, predictive accuracy and fairness degrade—even if no errors occur during training.

### 5.2 Sampling Bias and Its Sources

**Sampling bias** occurs when the probability of inclusion in the dataset is not uniform across the population and is correlated with variables of interest. Common sources include:

* **Selection bias**: only certain cases are observed (e.g. successful outcomes).
* **Survivorship bias**: failures or dropouts are systematically missing.
* **Measurement bias**: data quality varies across groups.

These biases are rarely visible through summary statistics alone; they often require reasoning about the data collection process.

#### Illustrative Python example: apparent representativeness vs hidden bias

```python
import pandas as pd

df = pd.DataFrame({
    "income": [22000, 24000, 26000, 28000, 30000],
    "defaulted": [False, False, False, False, False]
})

df.describe()
```

This dataset appears clean and internally consistent. However, if it only includes customers who were granted loans (and excludes rejected applicants), then default risk is artificially suppressed. The issue is not missing values or noise—it is **who was never observed**.

### 5.3 Representativeness Is Context-Dependent

A sample can be representative for one purpose and inappropriate for another. For instance, a dataset may be suitable for predicting short-term behaviour but unsuitable for long-term forecasting if conditions change.

Representativeness depends on:

* time (are future conditions similar to the past?),
* subpopulations (are all relevant groups included?),
* operational context (will the model be used under the same constraints as the data was collected?).

At MSc level, students should be able to articulate *why* a dataset is representative—or not—rather than assuming representativeness by default.

### 5.4 Train/Test Splits as Sampling Decisions (Introductory View)

Splitting data into training and test sets is often presented as a technical requirement. Conceptually, it is a **sampling exercise**: the test set is meant to approximate future or unseen data.

If the split does not respect the structure of the data (e.g. time ordering, group membership), evaluation results can be misleading even in the absence of explicit leakage.

#### Illustrative Python example: naive random splitting risk

```python
df = pd.DataFrame({
    "user_id": [1, 1, 2, 2],
    "event_time": [1, 2, 1, 2],
    "outcome": [0, 1, 0, 1]
})

df.sample(frac=0.5, random_state=42)
```

A random split may place data from the same user into both training and test sets, inflating apparent performance. While full treatment of splitting strategies belongs in later weeks, recognising the *sampling nature* of evaluation is essential from the outset.

### 5.5 Consequences of Biased Sampling

Biased samples can lead to:

* overconfident predictions,
* systematic errors for underrepresented groups,
* misleading estimates of uncertainty,
* ethical and legal risks in deployment.

Importantly, these issues often persist even when models are retrained or tuned. The problem lies upstream, in the relationship between data and population.

### 5.6 Sampling as an Ongoing Responsibility

Sampling is not a one-time concern resolved during dataset creation. As systems are updated, populations change, and data pipelines evolve, representativeness must be reassessed.

At MSc level, responsible practice involves:

* documenting assumptions about the population,
* explicitly stating known exclusions,
* revisiting sampling decisions when model performance changes.

---

**Common misconception (Section 5)**
*“A large dataset is automatically representative.”*
Size does not guarantee coverage. Large datasets can encode large biases if inclusion mechanisms systematically exclude or distort parts of the population.

---

*End of Section 5.*

---

Below is **Section 6** of the Week 1 canonical notes. This section introduces **data leakage** as a *conceptual validity problem*, not a technical mistake, with **minimal Python examples** designed to make leakage visible rather than to fit models.

---

## 6. Data Leakage: Using Forbidden Information (Introductory View)

Data leakage occurs when information that would not be available at the time a prediction is made is inadvertently used during model development or evaluation. Although often discussed in the context of surprisingly high accuracy, leakage is fundamentally a **data construction and representation problem**, not a modelling trick.

At MSc level, it is important to recognise leakage early—*before* algorithms are introduced—because once leakage enters the dataset, no amount of careful modelling can restore validity.

### 6.1 What Makes Information “Forbidden”

Information is forbidden when it violates the **temporal, causal, or operational logic** of the prediction task. Whether a feature is legitimate depends on *when* it becomes available and *how* it is generated.

Common sources of forbidden information include:

* values computed after the outcome has occurred,
* features that directly encode the target variable,
* aggregates that silently mix past and future observations.

Crucially, leakage can exist even when all variables appear sensible and well-defined. The issue lies not in variable names, but in their **relationship to the prediction task**.

### 6.2 Leakage Is a Validity Failure, Not a Performance Issue

Leakage is sometimes framed as a reason why models perform “too well.” This framing is misleading. High performance in the presence of leakage does not indicate success—it indicates that the evaluation no longer measures what it claims to measure.

When leakage occurs:

* test performance ceases to approximate real-world performance,
* comparisons between models become meaningless,
* deployment failures are almost guaranteed.

From a scientific perspective, leakage invalidates inference. From an engineering perspective, it produces brittle systems.

### 6.3 Common Pathways Through Which Leakage Enters Data

Leakage often arises unintentionally through seemingly reasonable steps, such as:

* computing summary statistics on the full dataset before splitting,
* encoding information derived from outcomes as features,
* aligning records in ways that collapse temporal order.

Because these steps are often automated or reused across projects, leakage can persist unnoticed unless explicitly checked.

#### Illustrative Python example: feature derived from the outcome

```python
import pandas as pd

df = pd.DataFrame({
    "application_date": ["2024-01-01", "2024-01-05"],
    "approval_date": ["2024-01-10", "2024-01-07"],
    "approved": [True, True]
})

# Derived feature
df["processing_time_days"] = (
    pd.to_datetime(df["approval_date"]) -
    pd.to_datetime(df["application_date"])
).dt.days
```

If `processing_time_days` is used to predict `approved`, the feature directly depends on information that only exists *after* approval. The dataset looks coherent, but its structure violates the intended prediction scenario.

### 6.4 Leakage vs Correlation: A Critical Distinction

Not all strong associations indicate leakage. Some variables are legitimately predictive because they are observed *before* the outcome and are causally or operationally linked to it.

Leakage is present when:

* the feature would be unavailable at prediction time, or
* the feature encodes the outcome itself, directly or indirectly.

Distinguishing between strong correlation and leakage requires careful reasoning about **data generation**, not statistical thresholds.

### 6.5 Why Leakage Is Hard to Detect Automatically

Automated checks can identify suspicious patterns, but they cannot fully determine whether information is forbidden. This is because leakage is defined relative to:

* the deployment scenario,
* the timing of data availability,
* institutional or process constraints.

As a result, leakage prevention is primarily a **conceptual discipline**, not a software feature. It relies on documenting assumptions and interrogating each feature’s provenance.

### 6.6 Early Detection as a Design Principle

Because leakage compromises validity at the foundation, it must be addressed **before** model selection, training, or evaluation. Treating leakage as a late-stage debugging problem is both inefficient and risky.

At MSc level, good practice involves:

* articulating the prediction moment explicitly (“What is known at this point?”),
* tracing each feature back to its source,
* rejecting features whose availability cannot be justified.

---

**Common misconception (Section 6)**
*“If a feature improves test accuracy, it must be useful.”*
Improved accuracy can result from leakage, in which case the feature improves performance only by violating the prediction scenario.

---

*End of Section 6.*

---

Below is **Section 7** of the Week 1 canonical notes. This section introduces **supervised vs unsupervised learning** as *problem-framing choices*, not algorithmic procedures, with **very light Python illustrations** used only to expose the *presence or absence of labels*.

---

## 7. Supervised vs Unsupervised Learning: Framing the Learning Task

One of the earliest—and most consequential—decisions in a machine learning project is whether the problem is **supervised** or **unsupervised**. This choice is not determined by the algorithm one prefers, but by the **structure of the available data** and the **question being asked**. At MSc level, the distinction should be understood conceptually, before any specific methods are introduced.

### 7.1 Supervised Learning: Learning With Targets

In **supervised learning**, each observation is associated with a known outcome or **target variable**. The goal is to learn a mapping from inputs (features) to outputs (targets) that generalises to unseen data.

Supervised learning is appropriate when:

* the target variable is well-defined and measurable,
* labels are available (or can be obtained) for a sufficient number of cases,
* predicting the target has clear operational meaning.

Depending on the nature of the target, supervised tasks are commonly described as:

* **classification**, when the target takes discrete values,
* **regression**, when the target is continuous.

At this stage, the important point is not how these tasks are solved, but that the **existence and definition of the target variable** fundamentally shape the learning problem.

#### Illustrative Python example: explicit presence of a target

```python
import pandas as pd

df = pd.DataFrame({
    "age": [25, 40, 31],
    "income": [28000, 45000, 36000],
    "defaulted": [False, True, False]
})

df.columns
```

The presence of a clearly defined target column (`defaulted`) makes this a supervised learning setup. How well the task is posed, however, depends on how that target was measured and whether it aligns with the real-world objective.

### 7.2 Unsupervised Learning: Learning Without Targets

In **unsupervised learning**, no explicit target variable is provided. Instead, the aim is to discover **structure, regularities, or patterns** within the data itself.

Unsupervised learning is appropriate when:

* labels are unavailable, unreliable, or prohibitively expensive,
* the goal is exploratory rather than predictive,
* the notion of “ground truth” is ambiguous or contested.

Common unsupervised goals include grouping similar observations, identifying unusual cases, or summarising high-dimensional data. Importantly, unsupervised learning does *not* imply the absence of assumptions; it merely shifts them from targets to notions such as similarity and structure.

#### Illustrative Python example: absence of a target

```python
df = pd.DataFrame({
    "age": [25, 40, 31],
    "income": [28000, 45000, 36000]
})

df.columns
```

Here, there is no explicit outcome variable. Any analysis performed on this dataset would aim to *describe or organise* the data rather than predict a known result.

### 7.3 The Decision Is About the Question, Not the Tool

A common mistake is to equate supervised learning with “prediction” and unsupervised learning with “exploration,” as if these were mutually exclusive categories. In practice, the same dataset can support different learning paradigms depending on the question being asked.

For example:

* With labels: *Can we predict which customers will churn?* (supervised)
* Without labels: *Are there distinct customer segments?* (unsupervised)

The dataset has not changed; the **analytical intent** has.

### 7.4 Labels Are Not Neutral Objects

Labels themselves are data, subject to noise, bias, and measurement error. A poorly defined or inconsistently measured target can undermine supervised learning just as severely as missing features undermine unsupervised analysis.

At MSc level, students should be able to question:

* how labels were generated,
* whether they reflect stable or shifting definitions,
* whose perspective they encode.

In some cases, an unsupervised approach may be preferable precisely because available labels are unreliable or contentious.

### 7.5 Choosing Between Supervised and Unsupervised Learning

Choosing the appropriate learning paradigm requires balancing:

* availability and quality of labels,
* clarity of the desired outcome,
* costs of misclassification or misinterpretation,
* ethical and operational constraints.

This decision should be revisited as new data becomes available or as project goals evolve.

---

**Common misconception (Section 7)**
*“Unsupervised learning is just supervised learning without labels.”*
Unsupervised learning addresses different questions and relies on different assumptions; removing labels does not simply make a supervised problem unsupervised.

---

*End of Section 7.*

---

Below is **Section 8** of the Week 1 canonical notes. This section focuses on **translating real-world questions into machine learning task definitions**, emphasising *reasoning and reformulation* rather than algorithms, with **light Python illustrations** to show how framing choices reshape data.

---

## 8. From Real-World Questions to Machine Learning Task Types

Machine learning does not begin with algorithms; it begins with **translation**. Real-world questions are typically vague, qualitative, or operationally ambiguous. To apply machine learning meaningfully, these questions must be reformulated into **well-posed analytical tasks** with explicit inputs, outputs, and units of analysis.

At MSc level, this translation step is where many projects succeed or fail. Poor task framing leads to models that optimise the wrong objective, answer a different question than intended, or produce results that cannot be acted upon.

### 8.1 Decomposing a Real-World Question

A useful starting point is to break any real-world question into three components:

1. **What is the entity of interest?**
   (person, transaction, event, time window, device, etc.)

2. **What is known at the decision point?**
   (features available before the outcome or intervention)

3. **What is the desired outcome or insight?**
   (prediction, grouping, summarisation, anomaly detection)

For example, the question *“Can we reduce customer churn?”* does not yet define:

* what counts as a customer,
* when churn is assessed,
* or whether the goal is prediction, explanation, or segmentation.

### 8.2 Identifying the Unit of Analysis

The **unit of analysis** determines what one row in the dataset represents. Changing the unit of analysis can change the learning task entirely, even when using the same raw data.

Consider behavioural logs:

* one row per event → fine-grained, temporal analysis,
* one row per user → aggregated, static analysis,
* one row per user-month → hybrid temporal framing.

Each choice encodes assumptions about stability, causality, and relevance.

#### Illustrative Python example: same data, different units

```python
import pandas as pd

events = pd.DataFrame({
    "user_id": [1, 1, 2, 2, 2],
    "event_type": ["click", "purchase", "click", "click", "purchase"],
    "amount": [0, 50, 0, 0, 30]
})

# Aggregate to user level
user_summary = events.groupby("user_id").agg(
    total_spend=("amount", "sum"),
    num_events=("event_type", "count")
)

user_summary
```

The aggregation implicitly reframes the question from *event-level behaviour* to *user-level characteristics*. Any subsequent learning task now operates at the user level, whether this was intended or not.

### 8.3 Defining the Target (or the Discovery Objective)

In supervised learning, the **target variable** formalises what is to be predicted. Defining it requires care:

* Is the target observable at prediction time?
* Is it stable over time?
* Does it align with the decision being supported?

In unsupervised learning, there is no explicit target, but there is still an objective: discovering structure, similarity, or deviation. This objective must be articulated just as clearly, even if it is not encoded as a column.

Ambiguity at this stage propagates downstream, often resulting in models that are technically correct but practically useless.

### 8.4 Mapping Questions to Task Types

Once the unit of analysis and objective are clear, the learning task can usually be categorised at a high level:

* **Classification**: predicting a discrete outcome (e.g. yes/no, category)
* **Regression**: predicting a continuous quantity
* **Structure discovery**: identifying patterns, groups, or regularities without labels

This categorisation is a *consequence* of problem framing, not a design choice made in isolation.

#### Illustrative Python example: task framing via target presence

```python
df = pd.DataFrame({
    "user_id": [1, 2, 3],
    "total_spend": [120, 300, 80],
    "churned": [False, True, False]
})

# Supervised framing
features = df.drop(columns=["churned"])
target = df["churned"]

features, target
```

Removing or redefining the target column would immediately change the nature of the task, even though the underlying data remains the same.

### 8.5 Success Criteria and Constraints

A well-posed machine learning task also requires **success criteria**. These may include:

* acceptable error rates,
* costs of different types of mistakes,
* interpretability requirements,
* ethical or legal constraints.

These considerations influence whether machine learning is appropriate at all, and if so, how results should be interpreted. Importantly, success criteria are often **external to the data** and must be specified deliberately.

### 8.6 Why Task Framing Is an Iterative Process

Task formulation is rarely correct on the first attempt. As data limitations become visible and assumptions are challenged, the framing may need to be revised:

* redefining the unit of analysis,
* adjusting the target definition,
* switching between supervised and unsupervised perspectives.

At MSc level, this iteration is a strength, not a failure. It reflects increasing alignment between the analytical task and the real-world problem.

---

**Common misconception (Section 8)**
*“Once a dataset is loaded, the machine learning task is fixed.”*
In reality, the task emerges from how the question is framed, what is defined as an observation, and what is treated as an outcome or objective.

---

*End of Section 8.*

---

Below is **Section 9**, the concluding section of the Week 1 canonical notes. This section consolidates the week’s ideas, makes misconceptions explicit, and clearly states what students should now be able to **reason about**.

---

## 9. Common Misconceptions and Week 1 Summary

The concepts introduced in this week are foundational rather than technical. As such, many misunderstandings arise not from mathematical difficulty, but from **implicit assumptions** carried over from programming or statistics courses. Making these misconceptions explicit helps prevent fragile reasoning later in the module.

### 9.1 Common Misconceptions Revisited

1. **“Better algorithms can compensate for weak data.”**
   More complex models do not create information that is absent or distorted in the data. They often amplify biases and artefacts, producing confident but unreliable results.

2. **“Cleaning, preprocessing, and feature engineering are interchangeable steps.”**
   These stages serve different purposes: restoring validity, enabling computation, and injecting meaning. Confusing them obscures where assumptions enter the pipeline.

3. **“If data is numeric, arithmetic operations are always meaningful.”**
   Numeric storage does not imply semantic validity. Identifiers, codes, and ordinal values require careful interpretation.

4. **“Large datasets are automatically representative.”**
   Size does not guarantee coverage or fairness. Large datasets can encode systematic exclusion just as effectively as small ones.

5. **“High test accuracy means the model will work in practice.”**
   Evaluation results are only meaningful if the data construction and sampling logic reflects real-world deployment conditions.

6. **“Unsupervised learning is what you do when labels are missing.”**
   Unsupervised methods address specific exploratory goals and rely on assumptions about similarity and structure, not simply the absence of labels.

7. **“Data leakage is a modelling mistake.”**
   Leakage is a data design failure that occurs before modelling begins. Once present, it invalidates evaluation regardless of algorithm choice.

### 9.2 How the Pieces Fit Together

Week 1 establishes a unifying perspective that will apply throughout the module:

* **Data representation** determines what relationships can be learned.
* **Data quality** shapes the validity of any inference.
* **Sampling and representativeness** link models to real populations.
* **Task framing** defines what learning problem is actually being solved.
* **Leakage awareness** protects against false confidence.

These elements are interdependent. A change in representation can introduce leakage; a cleaning decision can alter representativeness; a reframed task can change whether supervision is appropriate. Machine learning should therefore be understood as a **coherent pipeline of reasoning**, not a sequence of independent technical steps.

### 9.3 What You Should Now Be Able to Reason About

By the end of Week 1, you should be able to:

* Explain why data foundations, rather than algorithms, often determine machine learning success.
* Distinguish clearly between cleaning, preprocessing, and feature engineering.
* Identify common data quality risks and reason about their potential impact.
* Assess whether a dataset is likely to be representative of a target population.
* Recognise potential data leakage before any modelling is attempted.
* Decide whether a problem is better framed as supervised or unsupervised.
* Translate a real-world question into a well-defined machine learning task.

These abilities form the conceptual backbone for all subsequent weeks. The techniques introduced later in the module should be viewed as **tools applied within this framework**, not as replacements for it.

---

*End of Section 9. End of Week 1 canonical notes.*

