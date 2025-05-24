# COS314A3

# Machine Learning Classifier Suite (MLP, GP, DT)

This project includes implementations of three supervised machine learning models:

- **Multilayer Perceptron (MLP)**
- **Genetic Programming (GP)**
- **Decision Tree (DT)**

The main deliverable is an executable `.jar` file that runs **MLP**, **GP**, and a **Graph Creator** for visualizing results. The **Decision Tree (DT)** model will be provided in a separate `.jar`.

---

## Classifier Descriptions

- **MLP (Multilayer Perceptron):**  
  A fully connected feedforward neural network trained using backpropagation.

- **GP (Genetic Programming):**  
  An evolutionary algorithm that evolves classification trees over generations. It uses selection, crossover, and mutation to optimize structure and performance.

- **DT (Decision Tree):**  
  A traditional classification model that splits data recursively based on feature thresholds. Implemented using the J48 algorithm (a version of C4.5).  
  *Note: This model is compiled separately and will have its own executable.*

- **Graph Creator:**  
  A utility to visualize classification performance metrics (e.g., bias over iterations).

---

## Report

The full project report, detailing design decisions, results, and evaluation metrics, can be accessed here:  
[https://github.com/amJohnnyma/COS314A3/blob/main/COS314A3.pdf]

---

## How to Run

Ensure you have Java installed (`java -version`).

### Run UI (MLP, GP, Graphing tool), or DT

```bash
java -jar UI.jar
java -jar DT.jar
```

## Contributors

- **Dewald Colesky** — `u23536030`  
- **Liam van Kasterop** — `u22539761`  
- **Henco Pretorius** — `u23525381`  
- **Herrie Engelbrecht** — `u22512374`
