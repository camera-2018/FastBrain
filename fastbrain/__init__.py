"""FastBrain-9B: Non-Autoregressive System-1 Structured Decision Model."""
from .modeling_fastbrain import (
    ClefModel,
    JointSchemaHead,
    encode_record,
    collate_records,
    load_release_model,
    systemone,
    systemone_answer,
)
from .calibration import (
    SchemaTopology,
    extract_schema_topology,
    classify_action_option_roles,
    apply_schema_aware_calibration,
)

__version__ = "1.0.0"
__all__ = [
    "ClefModel",
    "JointSchemaHead",
    "encode_record",
    "collate_records",
    "load_release_model",
    "systemone",
    "systemone_answer",
    "SchemaTopology",
    "extract_schema_topology",
    "classify_action_option_roles",
    "apply_schema_aware_calibration",
]
