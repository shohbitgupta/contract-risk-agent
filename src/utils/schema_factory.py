from typing import Type, TypeVar, Dict, Any, Callable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def build_model(
    model_cls: Type[T],
    data: Dict[str, Any],
    *,
    strict: bool = True,
    log_fn: Callable[[str], None] = print
) -> T:
    """
    Schema-aware constructor for Pydantic models.

    - strict=True  -> crash on schema drift
    - strict=False -> drop unknown fields, log them
    """

    allowed = set(model_cls.model_fields.keys())
    incoming = set(data.keys())

    extras = incoming - allowed

    if extras:
        log_fn(
            f"[SCHEMA-DRIFT] {model_cls.__name__} received extra fields: "
            f"{sorted(extras)}"
        )

        if strict:
            raise ValueError(
                f"Schema drift in {model_cls.__name__}: {extras}"
            )

        # Drop unknown fields in non-strict mode
        data = {k: v for k, v in data.items() if k in allowed}

    return model_cls.model_validate(data)
