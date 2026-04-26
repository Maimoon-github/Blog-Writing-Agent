"""
Alembic environment configuration for database migrations.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import all models so Alembic can detect them for autogenerate
from blog_agent_system.persistence.database import Base, sync_engine
from blog_agent_system.persistence.models import Task, AgentRun, ConversationTurn, Document, Feedback

# Alembic Config object
config = context.config

# Setup logging from alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata for autogenerate support
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connects to DB)."""
    connectable = sync_engine

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
