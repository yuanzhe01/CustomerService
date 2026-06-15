from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from settings import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()

def init_db() -> None:
    # Delayed import to avoid circular dependency.
    import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
