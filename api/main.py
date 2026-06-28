from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from datetime import datetime
import uuid

from api.auth import verify_token

app = FastAPI(
    title="Scandium Labs API",
    description="AI-Driven Solid Electrolyte Discovery",
    version="1.0.0"
)

security = HTTPBearer()

inference_engine = None


def get_inference_engine():
    global inference_engine
    if inference_engine is None:
        try:
            from src.inference.engine import InferenceEngine
            inference_engine = InferenceEngine(
                model_path="checkpoints/best_model.pt",
                device="cpu",
                use_mc_dropout=True,
                mc_samples=20
            )
        except Exception:
            pass
    return inference_engine


class ScreeningRequest(BaseModel):
    material_ids: list[str] | None = None
    formulas: list[str] | None = None
    temperature: float = 300.0
    tasks: list[str] = [
        "log_ionic_conductivity",
        "formation_energy",
        "energy_above_hull",
        "activation_energy"
    ]
    top_k: int = 10


class ScreeningResult(BaseModel):
    job_id: str
    status: str
    results: list | None = None
    created_at: str
    completed_at: str | None = None


@app.post("/screen")
async def screen_materials(
    request: ScreeningRequest,
    user_id: str = Depends(verify_token)
):
    from api.tasks import screen_materials_task

    job_id = str(uuid.uuid4())

    try:
        task = screen_materials_task.delay(
            job_id=job_id,
            material_ids=request.material_ids,
            formulas=request.formulas,
            temperature=request.temperature,
            tasks=request.tasks,
            top_k=request.top_k
        )
        task_id = task.id
    except Exception:
        task_id = None

    try:
        from api.database import get_engine, get_session, Job
        engine = get_engine()
        session = get_session(engine)
        job = Job(
            id=job_id,
            user_id=user_id,
            status="queued",
            n_materials=len(request.material_ids or []) + len(request.formulas or []),
        )
        session.add(job)
        session.commit()
        session.close()
    except Exception:
        pass

    return ScreeningResult(
        job_id=job_id,
        status="queued",
        created_at=datetime.utcnow().isoformat(),
    )


@app.post("/screen/upload")
async def screen_from_cif(
    file: UploadFile = File(...),
    temperature: float = 300.0,
    user_id: str = Depends(verify_token)
):
    if not file.filename.endswith(('.cif', '.poscar', '.vasp')):
        raise HTTPException(400, "Only CIF and POSCAR files accepted")

    content = await file.read()
    from pymatgen.core import Structure
    from io import StringIO

    try:
        structure = Structure.from_str(content.decode(), fmt='cif')
    except Exception:
        try:
            structure = Structure.from_str(content.decode(), fmt='poscar')
        except Exception as e:
            raise HTTPException(400, f"Invalid structure file: {e}")

    engine = get_inference_engine()
    if engine is None or not engine.is_loaded:
        return {
            "material": file.filename,
            "formula": structure.composition.reduced_formula,
            "status": "inference_unavailable",
            "message": "Model not loaded. Start inference service first."
        }

    result = engine.predict_single(structure, temperature)
    result["material"] = file.filename
    result["formula"] = structure.composition.reduced_formula
    result["spacegroup"] = structure.get_space_group_info()[1]
    return result


@app.get("/job/{job_id}")
async def get_job_status(job_id: str, user_id: str = Depends(verify_token)):
    from api.database import get_engine, get_session, Job
    from api.tasks import celery_app

    try:
        engine = get_engine()
        session = get_session(engine)
        job = session.query(Job).filter(Job.id == job_id).first()
        session.close()

        if job is None:
            raise HTTPException(404, "Job not found")

        if celery_app is not None:
            from celery.result import AsyncResult
            async_result = AsyncResult(job_id, app=celery_app)

            if async_result.ready():
                result_data = async_result.get()
                return {
                    "job_id": job_id,
                    "status": "completed",
                    "results": result_data,
                    "top_k": result_data[:10] if result_data else []
                }

        progress = 0
        if job.n_materials > 0:
            progress = job.completed_materials / job.n_materials * 100

        return {
            "job_id": job_id,
            "status": job.status,
            "progress": progress,
            "n_materials": job.n_materials,
            "completed_materials": job.completed_materials,
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "unknown",
            "error": f"Database or worker unavailable: {str(e)}"
        }


@app.get("/health")
async def health():
    engine = get_inference_engine()
    return {
        "status": "healthy",
        "model_loaded": engine.is_loaded if engine else False
    }
