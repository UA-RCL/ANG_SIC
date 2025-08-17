# Synaptic Input Consolidation (SIC) and Artificial Neurogenesis (ANG)

## Build Process

* Create a `build` folder in the repo root directory and switch to the folder

    ```bash
    mkdir build
    cd build
    ```

* Generate a Makefile through CMake using the following command

    ```bash
    cmake ..
    ```

* Run the build process using Makefile

    ```bash
    make -j
    ```
    This should generate the binary `sic` in the `build` directory.

## Datasets

* Use the provided script to fetch the MNIST `.bin` files into `data/`.

  From the /data/MNIST folder:

  ```bash
  ./mnist_download.sh
  ```
  
## Running SIC

* The `sic` binary requires a configuration file (`.cfg`) and a mode as input arguments.

  ```bash
  ./sic -config ../config_files/MNIST_ANG.cfg -mode 0
  ```

  Use `-mode <id>` to select behavior:

  | Value | Mode name          | Effect            |
  |:----: |-------------------------|-------------------|
  | 0     | `BUILD_COMPLETE_NETWORK`       | Build/train a full seed network from scratch.          |
  | 1     | `REBUILD_COMPLETE_NETWORK` | Reload an existing full network and continue/evaluate.    |
  | 2     | `BUILD_CLASS_LEVEL_NETWORK` | Build per-class networks from scratch.    |
  | 3     | `REBUILD_CLASS_LEVEL_NETWORK`     | Reload per-class networks and continue/evaluate.        |
  | 4     | `NEUROGENESIS`     | Run the ANG growth phase on the seed network.        |
  | 5     | `REBUILD_NEUROGENESIS`     | Resume/reevaluate a saved ANG growth run.        |
  
* Configure SIC/pruning by adding one of these keys to the `.cfg`.

  | Key in `.cfg`  | Printed banner            | Effect            | Example line in `.cfg` |
  |---|---|---|---|
  | `SIC`          | `----- SIC -----`         | SIC only          | `SIC;`               |
  | `SIC_PRUNE`    | `----- SIC Prune -----`   | SIC then prune    | `SIC_PRUNE;`         |
  | `PRUNE_SIC`    | `----- Prune SIC -----`   | Prune then SIC    | `PRUNE_SIC;`         |
  | `PRUNE`        | `----- Prune -----`       | Prune only        | `PRUNE;`             |





