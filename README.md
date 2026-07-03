# Zoo Animal Classification: K-Nearest Neighbors (KNN) Framework

This repository features an end-to-end Machine Learning implementation focused on analyzing and classifying zoo animals into distinct taxonomic categories using a `Zoo.csv` dataset. The codebase transitions systematically from data cleaning and scaling to exploratory visualization, model evaluation, and an overview of hyperparameter optimization matrices.

---

## Pipeline Architecture & Workflow

The notebook processes animal physiological and behavioral features through a structured pipeline to classify species targets:

1. **Environment Setup & Data Ingestion**: Clean parsing of feature attributes from local sources.
2. **Feature Wrangling**: Renames the column headers (`animal name` $\rightarrow$ `animal_name`) to conform to clean Python syntax conventions and sets the target variable (`type`) as an explicit categorical factor.
3. **Exploratory Visualizations**: 
   * **Pairplots**: Applied specifically across numerical feature sub-spaces to capture multidimensional attribute relationships relative to classification classes.
   * **Boxplots**: Built across standardized feature ranges to detect scale skewness and potential outlier markers.
4. **Data Normalization**: Implements a `StandardScaler` to handle scale variance across the binary and continuous vectors.
5. **Model Training & Evaluation**: Instantiates a baseline $K$-Nearest Neighbors classification architecture using an 80/20 train-test validation split strategy.
6. **2D Decision Region Boundary Mapping**: Utilizes `mlxtend` to isolate the first two component vectors, mapping how local coordinate spaces are explicitly partitioned.

---

## Performance Matrix & Evaluation Metrics

The baseline model configured with $K = 5$ achieves a high overall categorization capability, explicitly detailed across individual classification classes below:

### Summary Performance
* **Overall Evaluation Accuracy**: `0.9524` (95.24%)
* **Macro Average Precision**: `0.7778`
* **Macro Average Recall**: `0.8333`
* **Macro Average F1-Score**: `0.8000`

### Complete Classification Breakdown Report

| Class Type | Precision | Recall | F1-Score | Support Count |
| :---: | :---: | :---: | :---: | :---: |
| **Class 1** | $1.00$ | $1.00$ | $1.00$ | 12 |
| **Class 2** | $1.00$ | $1.00$ | $1.00$ | 2 |
| **Class 3** | $0.00$ | $0.00$ | $0.00$ | 1 |
| **Class 4** | $0.67$ | $1.00$ | $0.80$ | 2 |
| **Class 6** | $1.00$ | $1.00$ | $1.00$ | 3 |
| **Class 7** | $1.00$ | $1.00$ | $1.00$ | 1 |

> **Note on Imbalance**: Class 3 exhibits low support count within the randomized evaluation split, which flags an explicit precision warning due to zero predicted samples under unweighted evaluations.

---

## Hyperparameter Configuration Space

To optimize performance boundaries, the codebase establishes a design matrix covering key tuning dimensions for `KNeighborsClassifier`:

* **Number of Neighbors ($K$)**: Controls the model variance-bias tradeoff. Lower values are susceptible to localized noise; larger values generalize but smooth out finer class clusters.
* **Distance Metric Frameworks**:
  * `euclidean`: Evaluates straight-line geometric distance.
  * `manhattan`: Computes absolute grid coordinate deltas (ideal for binary or blocky configurations).
  * `minkowski`: Generalized metric governed by power parameter $p$ ($p=1 \rightarrow \text{Manhattan}$, $p=2 \rightarrow \text{Euclidean}$).
  * `hamming`: Tracks mismatches within explicit categorical dimensions.
  * `cosine`: Evaluates vector orientation patterns rather than metric magnitude.
* **Weight Functions**:
  * `uniform`: Every neighbor receives equal voting privileges.
  * `distance`: Closer neighbors carry a heavier mathematical influence on predictions.

### Automated Optimization Workflow
The notebook details an optimization blueprint leveraging `GridSearchCV` to locate optimal configurations:

```python
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}
