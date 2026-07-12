# trading-contracts

Dependency-light, versioned Pydantic contracts shared by research, simulation,
DOIN adapters, model serving and live trading.

The package owns data shape and canonical serialization. It does not own model
training, portfolio decisions, broker behavior or transport.

## Development

```bash
python -m pytest -q
python scripts/export_schemas.py
```

Contract models reject unknown fields. Persisted contracts require timezone-
aware timestamps, stable object and trace identifiers, and producer identity.
Canonical hashes use sorted UTF-8 JSON with non-finite numbers forbidden.

