# Ternary Structures Exploration and Recommendation tool
Jupyter notebook interface was made for an exploration and recommendation tool that identifies promising new compound compositions based on chemical similarity and site-specific elemental substitution. 
The notebook acts as a front-end to backend Python scripts that carry out advanced data processing and machine learning analysis.
## How to run 

```bash
pip install -r requirements.txt
```
run `main.ipynb`

## How it works

1.	**Data Parsing and Grouping**

    User selects what atomic sites are grouped based on the elements occupying them. 

2. **Machine Learning Analysis**
   For each group of sites:
   - **Partial Least Squares Discriminant Analysis (PLS-DA)** is conducted using features derived from elemental properties.
   - **Principal Component Analysis (PCA)** is then performed on the loadings from PLS-DA to visualize chemical similarity in a lower-dimensional space.
3. **Chemical Similarity Projection**  
   All remaining elements are projected into the PCA space, allowing the user to analyze and visualize chemical similarity between elements.

4. **Visualization**  
   Each compound is visualized with atomic composition mapped to PCA-derived coordinates, representing the weighted contributions of the constituent elements.

5. **Substitution Recommendation**  
   Based on the PCA space, the program recommends element substitutions under three modes:
   1. **Fixed-site substitution** — with some atomic sites fixed, find the best substitutes for the remaining sites.
   2. **Alternative fixed-site combinations** — explore other fixed-site combinations using known substitutions.
   3. **Exploration within PCA-defined chemical space** — identify promising elements within the learned chemical space.

6. **Compound Generation**  
   After evaluating all recommendation modes, the most promising candidates are selected, and new compound compositions are proposed, considering **stoichiometric constraints**.



