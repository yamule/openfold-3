# OpenFold3 Training


This guide covers how to train OpenFold3 on the PDB dataset.


## 1. Download the dataset

```bash
aws s3 sync s3://openfold3-data/of3_dataset_releases/af3_training_data_v14_reupload/ /shared/openfold3/pdb_dataset_releases/v14/
```

## 2. Prepare the training config

The training config is stored in a YAML file, this is what your `training.yaml` could look like. 

**Note** make sure you update the paths match your location, here we're assuming a `/shared/openfold3` that's common to all your traning nodes.

```yaml
experiment_settings:
  mode: train
  output_dir: ./test_train_output 
  seed: 1272025
  restart_checkpoint_path: last

data_module_args:
  batch_size: 1
  num_workers: 1
  epoch_len: 128000 # Ckpt every 500 steps

logging_config:
  log_lr: false
  wandb_config: null

pl_trainer_args:
  devices: 1
  num_nodes: 1
  precision: bf16-mixed
  max_epochs: -1
  log_every_n_steps: 50
  
  mpi_plugin: false
  deepspeed_config_path: null

# Checkpoint settings
checkpoint_config:
  every_n_epochs: 1
  auto_insert_metric_name: false
  save_last: true
  save_top_k: -1

model_update:
  presets: 
    - train

dataset_configs:
  train:
    weighted-pdb:
      dataset_class: WeightedPDBDataset
      weight: 0.5
      config:
        debug_mode: true
        template:
          n_templates: 4
          take_top_k: false
        crop:
          token_crop:
            enabled: true
            token_budget: 384
            crop_weights:
              contiguous: 0.2
              spatial: 0.4
              spatial_interface: 0.4
          chain_crop:
            enabled: true

  validation:
    val-weighted-pdb:
      dataset_class: ValidationPDBDataset
      config:
        debug_mode: true
        msa:
          subsample_main: false
        template:
          n_templates: 4
          take_top_k: true
        crop:
          token_crop:
            enabled: false

dataset_paths:
  weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_dataset_releases/v14/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_dataset_releases/v14/training_cache_with_templates.json
    target_structures_directory: /shared/openfold3/pdb_dataset_releases/v14/preprocessed_pdb/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_dataset_releases/v14/preprocessed_pdb/reference_mols
    template_cache_directory: /shared/openfold3/pdb_dataset_releases/v14/train_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_dataset_releases/v14/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null

  val-weighted-pdb:
    alignments_directory: none
    alignment_db_directory: none 
    alignment_array_directory: /shared/openfold3/pdb_dataset_releases/v14/alignment_arrays
    dataset_cache_file: /shared/openfold3/pdb_dataset_releases/v14/validation_cache_with_templates_with_ab_ag_no_err.json
    target_structures_directory: /shared/openfold3/pdb_dataset_releases/v14/preprocessed_pdb/structure_files
    target_structure_file_format: npz
    reference_molecule_directory: /shared/openfold3/pdb_dataset_releases/v14/preprocessed_pdb/reference_mols
    template_cache_directory: /shared/openfold3/pdb_dataset_releases/v14/val_template_cache
    template_structure_array_directory: /shared/openfold3/pdb_dataset_releases/v14/template_structure_arrays
    template_structures_directory: none
    template_file_format: npz
    ccd_file: null
```

## 3. Launch training


```bash
run_openfold train --runner-yaml training.yaml --seed 42
```
