from pathlib import Path
import shutil
import tempfile

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile
)

from deployment.inference import predict
from deployment.model_registry import registry


app = FastAPI(
    title="Inspectra API",
    version="1.0.0",
    description="Industrial visual inspection inference API"
)


@app.get(
    "/"
)
def root():

    return {
        "name": "Inspectra",
        "status": "online",
        "models": registry.list_models()
    }


@app.get(
    "/health"
)
def health():

    return {
        "status": "healthy",
        "models": registry.list_models()
    }


@app.get(
    "/models"
)
def models():

    return {
        "models": [
            {
                "name": name,
                **registry.metadata(name)
            }
            for name in registry.list_models()
        ]
    }


@app.post(
    "/predict/{model_name}"
)
async def prediction(
    model_name: str,
    file: UploadFile = File(...)
):

    if model_name not in registry.list_models():

        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}"
        )

    suffix = (
        Path(
            file.filename or "image.jpg"
        ).suffix
        or ".jpg"
    )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp:

            temporary_path = Path(
                temp.name
            )

            shutil.copyfileobj(
                file.file,
                temp
            )

        result = predict(
            model_name,
            temporary_path
        )

        return {
            "filename": file.filename,
            "result": result
        }

    finally:

        if (
            temporary_path
            and temporary_path.exists()
        ):

            temporary_path.unlink()