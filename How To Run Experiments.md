Of course. Here is the organized markdown guide on how to use the training and sweep framework.

-----

### **How to Use the Training & Sweeps Framework**

This guide explains how to run individual training jobs and how to launch large-scale hyperparameter sweeps using Weights & Biases.

### **1. For Single Runs**

Use the scripts in `launchers/single_runs/` to manually launch a single training job. This is ideal for testing, debugging, or running a one-off experiment with fixed parameters.

#### **VAE Training**

The `train_ae_[task].sh` scripts accept the number of latent channels (`z_channels`) as an argument.

  * **Train a VAE for TB with `z_channels=1`:**

    ```bash
    ./launchers/single_runs/vae/train_ae_tb.sh 1
    ```

  * **Train a VAE for Pneumonia with `z_channels=3`:**

    ```bash
    ./launchers/single_runs/vae/train_ae_pneumonia.sh 3
    ```

#### **LDM Training**

The `train_ldm_[task].sh` scripts accept a training mode as an argument.

**Available modes:** `full_train`, `overfit_one`, `overfit_16`, `overfit_32`.

**❗ Important:** Before running, you **must edit** the corresponding script (e.g., `launchers/single_runs/ldm/train_ldm_tb.sh`) and update the `AE_CKPT_PATH` variable to point to the VAE checkpoint you want to use.

  * **Train an LDM on the full TB dataset:**

    ```bash
    ./launchers/single_runs/ldm/train_ldm_tb.sh full_train
    ```

  * **Train an LDM by overfitting on 16 TB samples:**

    ```bash
    ./launchers/single_runs/ldm/train_ldm_tb.sh overfit_16
    ```

  * **Train an LDM on the full Pneumonia dataset:**

    ```bash
    ./launchers/single_runs/ldm/train_ldm_pneumonia.sh full_train
    ```

-----

### **2. For W\&B Sweeps**

Sweeps automate the process of running many experiments to find optimal models and hyperparameters. The process is always two steps: initialize the sweep and then run the agent.

#### **Step 1: Initialize the Sweep**

Choose the `.yaml` configuration file for the experiment you want to run and use the `wandb sweep` command. This will provide you with a unique **Sweep ID**.

  * **Sweep VAE `z_channels` for TB:**

    ```bash
    wandb sweep sweeps/vae/z_channels/tb.yaml
    ```

  * **Sweep LDM `z_channels` for Pneumonia:**
    *(**Prerequisite:** Ensure you have first trained the VAEs for Pneumonia for each `z_channel` and run the `find_latent_scale.py` script for each one.)*

    ```bash
    wandb sweep sweeps/ldm/z_channels/pneumonia.yaml
    ```

  * **Sweep LDM Hyperparameters for TB (Full Training):**

    ```bash
    wandb sweep sweeps/ldm/hyperparams/tb_full_train.yaml
    ```

  * **Sweep LDM Hyperparameters for Pneumonia (Overfitting on 32 samples):**

    ```bash
    wandb sweep sweeps/ldm/hyperparams/pneumonia_overfit_32.yaml
    ```

#### **Step 2: Run the Agent**

After initializing the sweep, run the `wandb agent` on your cluster's login node using the Sweep ID provided. The agent will automatically start scheduling and running jobs on SLURM.

  * **Run the agent:**
    *(Replace `<your_username>/<your_project>/<sweep_id>` with the actual Sweep ID from Step 1)*
    ```bash
    wandb agent <your_username>/<your_project>/<sweep_id>
    ```

The agent will continue to run in your terminal, submitting new jobs as old ones finish, until the sweep is complete or you stop it. You can monitor all runs from the W\&B dashboard.