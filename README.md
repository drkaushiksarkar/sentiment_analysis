# Health Nlp Engine

Health-focused NLP engine for biomedical text processing, entity extraction, relation mining, and evidence synthesis.

## Architecture

```
health-nlp-engine/
  src/           # Core modules
  tests/         # Unit and integration tests
  config/        # Configuration files
  docs/          # Documentation
```

## Modules

- **text_processor**: Core text processor functionality
- **entity_extractor**: Core entity extractor functionality
- **relation_miner**: Core relation miner functionality
- **sentiment_scorer**: Core sentiment scorer functionality
- **summary_generator**: Core summary generator functionality

## Quick Start

```bash
pip install -r requirements.txt
python -m health_nlp_engine.main
```

## Testing

```bash
pytest tests/ -v
```

## License

MIT License - see LICENSE for details.
