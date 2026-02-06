# ML Security Practices

These security practices must be followed when working with machine learning code to protect sensitive data and credentials.

## API Keys and Credentials

### Never Log Sensitive Information

- **Never log API keys, tokens, or credentials** in print statements, logs, or experiment trackers:

  ```python
  # Bad - exposes API key in logs
  print(f"Using API key: {api_key}")
  self.log("api_key", api_key)

  # Good - log only that key exists
  print("API key configured")
  ```

### Never Commit Credentials

- **Never commit API keys or tokens to version control**:
  - Add `wandb/` to `.gitignore` (contains W&B credentials)
  - Add `.env` files to `.gitignore`
  - Add `secrets.yaml` or similar to `.gitignore`

- **Example `.gitignore` entries**:

  ```gitignore
  # Credentials and secrets
  .env
  .env.local
  secrets.yaml
  credentials.json

  # W&B credentials
  wandb/
  .wandb/

  # API keys
  *_api_key.txt
  *_token.txt
  ```

### Use Pydantic Settings (Recommended)

- **Use `pydantic-settings` for type-safe environment variable management**:

  Create centralized configuration in `src/{{project_name}}/config/settings.py`:

  ```python
  from pydantic_settings import BaseSettings, SettingsConfigDict
  from pathlib import Path

  class Settings(BaseSettings):
      """Application settings loaded from environment variables."""

      # API Keys and credentials
      wandb_api_key: str
      hf_token: str | None = None

      # Optional settings with defaults
      wandb_project: str = "my-ml-project"
      log_level: str = "INFO"

      # Paths
      data_dir: Path = Path("data")
      output_dir: Path = Path("outputs")

      model_config = SettingsConfigDict(
          env_file=".env",
          env_file_encoding="utf-8",
          case_sensitive=False,
          extra="ignore"
      )

  # Global settings instance
  settings = Settings()
  ```

- **Use `.env` files for local development**:

  ```bash
  # .env (never commit this)
  WANDB_API_KEY=your_key_here
  HF_TOKEN=your_token_here
  WANDB_PROJECT=my-experiment
  LOG_LEVEL=DEBUG
  ```

- **Access settings throughout your application**:

  ```python
  # In your training code
  from my_project.config.settings import settings

  # Type-safe access with autocomplete
  wandb.init(
      project=settings.wandb_project,
      api_key=settings.wandb_api_key
  )
  ```

**Why Pydantic Settings:**

- ✅ **Type safety**: Fields are type-checked at runtime
- ✅ **Validation**: Automatic validation of required fields
- ✅ **IDE support**: Autocomplete and type hints
- ✅ **Defaults**: Easy to specify default values
- ✅ **Multiple sources**: Loads from `.env`, environment variables, and system environment

**Alternative (python-dotenv)**: For simpler projects, you can use `python-dotenv`, but it lacks type safety and validation.

## Logging and Tracking

### Sanitize File Paths

- **Sanitize file paths before logging** to avoid exposing system structure:

  ```python
  import os
  from pathlib import Path

  # Good - log relative path only
  rel_path = Path(file_path).relative_to(project_root)
  self.log("checkpoint_path", str(rel_path))

  # Bad - logs full system path
  self.log("checkpoint_path", "/home/user/secret/project/checkpoint.pt")
  ```

### Sanitize Data Samples

- **Never log raw data samples** that may contain PII (Personally Identifiable Information):

  ```python
  # Bad - may contain sensitive user data
  self.log("sample", str(batch["text"][0]))

  # Good - log only statistics
  self.log("batch_size", len(batch["text"]))
  self.log("avg_sequence_length", batch["lengths"].float().mean())
  ```

## Model Artifacts

### Secure Checkpoints

- **Store model checkpoints securely**:
  - Use appropriate file permissions (e.g., `chmod 600` for sensitive models)
  - Store in non-public directories
  - Consider encrypting checkpoints for sensitive models

- **Example: Set secure permissions**:

  ```python
  import os

  checkpoint_path = "checkpoints/model.ckpt"
  torch.save(model.state_dict(), checkpoint_path)

  # Set read/write for owner only
  os.chmod(checkpoint_path, 0o600)
  ```

### Verify Model Sources

- **Only load models from trusted sources**:

  ```python
  # Good - verify source before loading
  if not checkpoint_path.startswith(trusted_dir):
      raise ValueError(f"Untrusted checkpoint path: {checkpoint_path}")

  model.load_state_dict(torch.load(checkpoint_path))

  # Bad - loading arbitrary checkpoints
  model.load_state_dict(torch.load(user_provided_path))
  ```

## Configuration Files

### Separate Secrets from Configs

- **Keep secrets separate from configuration files**:

  ```yaml
  # config.yaml (can be committed)
  model:
    hidden_dim: 256
    num_layers: 4

  # secrets.yaml (never commit)
  wandb:
    api_key: ${oc.env:WANDB_API_KEY}
  ```

- **Use Hydra's environment variable resolution**:

  ```yaml
  # Reference environment variables in config
  wandb:
    api_key: ${oc.env:WANDB_API_KEY}
    project: ${oc.env:WANDB_PROJECT,my-project}  # with default
  ```

## Data Privacy

### Anonymize Datasets

- **Anonymize sensitive data before training**:
  - Remove PII (names, emails, addresses, phone numbers)
  - Replace with synthetic data or pseudonyms
  - Document anonymization process

### Secure Data Storage

- **Store datasets in secure locations**:
  - Use appropriate file permissions
  - Encrypt sensitive datasets at rest
  - Document data access policies

- **Example data organization**:

  ```text
  data/
  ├── public/           # Non-sensitive, can be shared
  ├── internal/         # Internal use only, restrict access
  └── sensitive/        # Sensitive data, encrypted + restricted
  ```

## Code Injection Prevention

### Validate User Inputs

- **Validate and sanitize user inputs** before using in commands or configs:

  ```python
  import re

  # Good - validate before use
  def validate_experiment_name(name: str) -> str:
      if not re.match(r'^[a-zA-Z0-9_-]+$', name):
          raise ValueError(f"Invalid experiment name: {name}")
      return name

  # Bad - directly using user input
  os.system(f"mkdir experiments/{user_input}")
  ```

### Avoid `eval()` and `exec()`

- **Never use `eval()` or `exec()` on untrusted input**:

  ```python
  # Bad - code injection risk
  config_value = eval(user_input)

  # Good - use safe alternatives
  import json
  config_value = json.loads(user_input)

  # Or use Hydra for configuration
  ```

## Dependency Security

### Pin Dependencies

- **Pin dependency versions** in `pixi.toml` or `requirements.txt`:

  ```toml
  [dependencies]
  torch = "2.1.0"
  pytorch-lightning = "2.1.0"

  # Bad - unpinned versions
  torch = "*"
  ```

### Scan for Vulnerabilities

- **Regularly scan dependencies for vulnerabilities**:

  ```bash
  # Using pip-audit
  pip-audit

  # Using safety
  safety check
  ```

### Update Dependencies

- **Keep dependencies up-to-date** with security patches:
  - Monitor security advisories for key dependencies (PyTorch, NumPy, etc.)
  - Test updates in staging before production
  - Document known vulnerabilities and mitigation strategies

## Audit Trail

### Log Security-Relevant Events

- **Log security-relevant events** (authentication, data access, model loading):

  ```python
  import logging

  logger = logging.getLogger(__name__)

  # Log model loading
  logger.info(f"Loading model from {checkpoint_path}")

  # Log data access
  logger.info(f"Accessing dataset: {dataset_name}")
  ```

### Include Metadata

- **Include relevant metadata in logs**:
  - Timestamp
  - User (if applicable)
  - Action performed
  - Success/failure status

## Summary Checklist

Before committing or deploying:

- [ ] No API keys or tokens in code or logs
- [ ] Sensitive files added to `.gitignore`
- [ ] Credentials loaded from environment variables
- [ ] File paths sanitized before logging
- [ ] Model checkpoints have appropriate permissions
- [ ] User inputs validated and sanitized
- [ ] Dependencies pinned and scanned
- [ ] Security-relevant events logged
- [ ] PII removed or anonymized from datasets
