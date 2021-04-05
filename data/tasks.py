import sqlalchemy
from sqlalchemy import orm

from .db_session import SqlAlchemyBase


class Task(SqlAlchemyBase):
    __tablename__ = 'tasks'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    subject = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("subjects.id"),
                                nullable=False)
    type = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("task_types.id"),
                             nullable=False)
    info = sqlalchemy.Column(sqlalchemy.JSON, nullable=False, unique=True)
    solution_path = sqlalchemy.Column(sqlalchemy.String, nullable=False, unique=True)
    subject_obj = orm.relation('Subject')
    type_obj = orm.relation('Task_type')
