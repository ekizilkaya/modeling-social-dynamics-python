# Modeling Social Dynamics: Python Simulations

This repository provides a Python implementation of the simulations featured in Paul Smaldino’s *Mathematical and Agent-Based Models of Social Dynamics and Cultural Evolution*.

This project is a functional port and enhancement of the original Julia implementation by [BenjaFried/modsoc_Julia](https://github.com/BenjaFried/modsoc_Julia).

## Why a Python Port?

While the original Julia code offers excellent performance, this Python version was developed to:

* **Broaden Accessibility:** Python is the primary language for many social scientists and data researchers; this port allows them to engage with Smaldino’s models using a familiar stack.
* **Ease of Integration:** By bringing these models to Python, they can be easily integrated with standard data science libraries like **Pandas**, **Matplotlib**, and **Scikit-learn**.
* **Educational Utility:** The code serves as a "prosthesis for the imagination," helping students visualize complex emergent behaviors like bird flocking, disease contagion, and wealth inequality.

---

## Key Enhancements in this Port

This version includes several improvements over a direct 1:1 translation to ensure the codebase is readable, performant, and "Pythonic":

### 1. Descriptive File Naming

To improve navigation and clarity, file names have been updated from abbreviated versions to descriptive titles.

* *Example:* `2_Groups.py` has been renamed to `Two-Group Norm Dynamics.py`.

### 2. Standardized In-Code Documentation

Every script now begins with a detailed header description. These docstrings outline the model’s core logic, its parameters, and the specific chapter of the textbook it corresponds to, making the code self-documenting for researchers and students.

### 3. Performance & Optimization

Julia’s Just-In-Time (JIT) compilation is naturally faster for heavy numerical loops. To mitigate the performance gap in Python:

* **Vectorization:** We utilize **NumPy** for array-based operations where possible.
* **Numba Integration:** For computationally intensive simulations (such as **Chapter 7: Coordination with Asymmetric Payoffs**), we have implemented **Numba**'s JIT compilation to ensure execution speeds remain viable.

---

## Repository Contents

* **Chapter 2: Particles** – Introduction to agent-based modeling (ABM) using mobile agents and qualitative analysis.
* **Chapter 3: The Schelling Chapter** – Exploration of spatial modeling, cellular automata, and parameter sweeps through the famous Segregation model.
* **Chapter 4: Contagion** – Compartment models for studying the spread of disease and innovation, combining math with simulations.
* **Chapter 5: Opinion Dynamics** – Modeling social influence and polarization through continuous opinion values.
* **Chapter 6: Cooperation** – Game theory and evolutionary dynamics, demonstrating the link between ABMs and mathematical proofs.
* **Chapter 7: Coordination** – Focused on the emergence of group-beneficial norms and the division of labor (*Optimized with Numba*).
* **Chapter 8: The Scientific Process** – Using Bayes’ theorem and ABMs to model scientific inquiry and professional incentives.
* **Chapter 9: Networks** – An introduction to network science, exploring metrics and algorithms on complex interaction architectures.

---

## About the Source Material

Prof. [Paul Smaldino](https://smaldino.com/wp/) argues that formal modeling is essential for scientific progress because it forces researchers to simplify reality into precise, logical assumptions.

This repository supports both **Equation-Based Models (EBMs)** and **Agent-Based Models (ABMs)**. While EBMs offer mathematical elegance and expected averages, these Python ABMs allow for the exploration of individual heterogeneity, spatial structure, and the stochastic fluctuations that define real-world social systems.

## License and Attribution

This project is a derivative work. Please attribute the original Julia developers and the textbook author in any academic use.

**Original Julia Repository:** [BenjaFried/modsoc_Julia](https://github.com/BenjaFried/modsoc_Julia)

**Original Author:** [Benjamin Fried](https://csh.ac.at/95906-2/)

**Textbook:** [Modeling Social Dynamics: Mathematical and Agent-Based Models of Social Dynamics and Cultural Evolution](https://press.princeton.edu/books/paperback/9780691224145/modeling-social-behavior) by Paul Smaldino.
