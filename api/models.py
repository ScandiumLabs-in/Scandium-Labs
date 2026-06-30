from typing import Optional

from pydantic import BaseModel


class MaterialScreeningResult(BaseModel):
    material_id: str
    formula: str
    spacegroup: Optional[int] = None

    log_ionic_conductivity: Optional[float] = None
    log_ionic_conductivity_std: Optional[float] = None
    formation_energy: Optional[float] = None
    formation_energy_std: Optional[float] = None
    energy_above_hull: Optional[float] = None
    energy_above_hull_std: Optional[float] = None
    activation_energy: Optional[float] = None
    activation_energy_std: Optional[float] = None
    band_gap: Optional[float] = None

    ood_score: Optional[float] = None
    is_ood: bool = False
    recommendation: Optional[str] = None

    temperature_k: float = 300.0
    model_version: Optional[str] = None
    screening_time_ms: Optional[float] = None


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    n_materials: int = 0
    completed_materials: int = 0
    created_at: str
    completed_at: Optional[str] = None
