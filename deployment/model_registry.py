from pathlib import Path
import onnxruntime as ort


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPORT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "exports"
)


MODEL_PATHS = {
    "bottle": (
        EXPORT_DIR
        / "bottle"
        / "bottle_finetuned.onnx"
    ),

    "pcb": (
        EXPORT_DIR
        / "pcb"
        / "pcb_finetuned.onnx"
    ),

    "road": (
        EXPORT_DIR
        / "road"
        / "road_resnet18_finetuned.onnx"
    ),
}


MODEL_METADATA = {
    "bottle": {
        "task": "detection",
        "input_size": 640,
        "classes": [
            "Cap",
            "Missing",
            "Wrong bottle",
            "box"
        ]
    },

    "pcb": {
        "task": "detection",
        "input_size": 640,
        "classes": [
            "mouse_bite",
            "spur",
            "missing_hole",
            "short",
            "open_circuit",
            "spurious_copper"
        ]
    },

    "road": {
        "task": "classification",
        "input_size": 224,
        "classes": [
            "Negative",
            "Positive"
        ]
    }
}


class ModelRegistry:

    def __init__(self):

        self.sessions = {}

    def load(
        self,
        name
    ):

        if name not in MODEL_PATHS:

            raise ValueError(
                f"Unknown model: {name}"
            )

        if name not in self.sessions:

            path = MODEL_PATHS[name]

            if not path.exists():

                raise FileNotFoundError(
                    f"Model not found: {path}"
                )

            self.sessions[name] = (
                ort.InferenceSession(
                    str(path),
                    providers=[
                        "CPUExecutionProvider"
                    ]
                )
            )

        return self.sessions[name]

    def list_models(
        self
    ):

        return list(
            MODEL_PATHS.keys()
        )

    def metadata(
        self,
        name
    ):

        if name not in MODEL_METADATA:

            raise ValueError(
                f"Unknown model: {name}"
            )

        return MODEL_METADATA[name]


registry = ModelRegistry()