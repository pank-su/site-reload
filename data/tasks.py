import datetime
import sqlalchemy
from .db_session import SqlAlchemyBase

class Task(SqlAlchemyBase):
    __tablename__ = 'tasks'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    subject = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    info = sqlalchemy.Column(sqlalchemy.JSON, nullable=False)
    solution_path = sqlalchemy.Column(sqlalchemy.String, nullable=False)