from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()


class Material(Base):
    __tablename__ = 'materials'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    material_id = Column(String(50), nullable=True)
    formula = Column(String(200), nullable=False)
    spacegroup = Column(Integer, nullable=True)
    structure_json = Column(JSON, nullable=True)
    source = Column(String(50), default='user_upload')
    created_at = Column(DateTime, default=datetime.utcnow)


class ScreeningResult(Base):
    __tablename__ = 'screening_results'

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    material_id = Column(String, nullable=True)
    job_id = Column(String, nullable=True)

    log_ionic_conductivity = Column(Float, nullable=True)
    log_ionic_conductivity_std = Column(Float, nullable=True)
    formation_energy = Column(Float, nullable=True)
    formation_energy_std = Column(Float, nullable=True)
    energy_above_hull = Column(Float, nullable=True)
    energy_above_hull_std = Column(Float, nullable=True)
    activation_energy = Column(Float, nullable=True)
    activation_energy_std = Column(Float, nullable=True)
    band_gap = Column(Float, nullable=True)

    temperature_k = Column(Float, default=300.0)
    model_version = Column(String(50), nullable=True)
    ood_score = Column(Float, nullable=True)
    is_ood = Column(Boolean, default=False)
    confidence_score = Column(Float, nullable=True)
    pareto_rank = Column(Integer, nullable=True)
    recommendation = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Job(Base):
    __tablename__ = 'jobs'

    id = Column(String, primary_key=True)
    user_id = Column(String(100), nullable=True)
    status = Column(String(20), default='queued')
    n_materials = Column(Integer, default=0)
    completed_materials = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


def get_engine(db_url="postgresql://user:pass@postgres:5432/scandium"):
    engine = create_engine(db_url, pool_size=10)
    Base.metadata.create_all(engine)
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
