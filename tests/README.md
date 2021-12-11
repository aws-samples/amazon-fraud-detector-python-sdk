# Unit Tests

Unit tests are executed with the `pytest` framework.

Configuration is in the `pytest.ini` file in the code repo root.

The tests assume access to an AWS cloud service with the necessary access privileges.
  
## Running tests via `pytest` CLI

To run all tests:
```
pytest
```

To run a specific set of test modules:  
```
pytest tests/test_frauddetector.py 
```

To specify the logging level:
```
pytest tests/test_frauddetector.py  --log-cli-level=INFO
```

To print to std-out

```
pytest -s
```