# Deprecated Configs

The following config files are kept for historical reference but are NOT actively used:

| File | Replaced By |
|------|-------------|
| `model_config.yaml` | `model_config_v3_li.yaml` |
| `model_config_v2.yaml` | `model_config_v3_li.yaml` |
| `model_config_v3.yaml` | `model_config_v3_li.yaml` |
| `phase3_config_log_eah.yaml` | `model_config_v3_li.yaml` |
| `finetune_config.yaml` | N/A — superseded by full training |
| `data_config.yaml` | Built into `build_dataset.py` |

**Active configs:**
- `model_config_v3_li.yaml` — current training config
- `ds_config.json` — DeepSpeed configuration
- `deploy_config.yaml` — deployment configuration
