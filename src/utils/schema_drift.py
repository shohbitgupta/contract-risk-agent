SCHEMA_DRIFT_EVENTS: list[str] = []


def log_schema_drift(message: str):
    SCHEMA_DRIFT_EVENTS.append(message)
    print(message)