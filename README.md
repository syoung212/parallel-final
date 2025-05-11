# Optimizing MCL 

This repository contains implementations of the Markov Clustering (MCL) algorithm and utilities to compare their performance.

---

## To Build and Run the Program

1. Run this to make the file
   ```bash
   make all
   ```

2. Run this to compare the implementations

   ```bash
   ./mcl_all [PATH_TO_MATRIX]
   ```

   * **`[PATH_TO_MATRIX]`**: Path to an input matrix in (`.mtx`) format.

---

## To Generate a Random Matrix

We have provided a Python script to generate a synthetic square matrix:

```bash
python data/generate_matrix.py [RANGE_OF_NUMBERS] [SIZE_OF_MATRIX] [NAME_OF_OUTPUT_FILE]
```

* **`[RANGE_OF_NUMBERS]`**: Maximum integer value (entries will be between `0` and this number).
* **`[SIZE_OF_MATRIX]`**: Dimension of the square matrix (e.g., `2000` for a 2000×2000 matrix).
* **`[NAME_OF_OUTPUT_FILE]`**: Output file name (e.g., `my_matrix.mtx`).

### Example

Generate a 2000×2000 matrix with values in the range `[0–100]` and save it as `my_matrix.mtx`:

```bash
python data/generate_matrix.py 100 2000 my_matrix.mtx
```

---

## Testing

While you could generate your own matrix, we’ve included several sample matrices in the `data/` folder for simplicity. To run a comparison on one of these, simply:

```bash
./mcl_all data/<preset_matrix_name>.mtx
```

Replace `<preset_matrix_name>` with the file name of any test matrix.
