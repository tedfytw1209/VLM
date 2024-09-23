# VLM

# MONAI-VILA

monai_vila2d

## Contributing

To lint the code, please install these packages:

```bash
pip install -r requirements-ci.txt
```

Then run the following command:

```bash
isort --check-only --diff .  # using the configuration in pyproject.toml
black . --check  # using the configuration in pyproject.toml
ruff check .  # using the configuration in ruff.toml
```

To auto-format the code, run the following command:

```bash
isort . && black . && ruff format .
```
