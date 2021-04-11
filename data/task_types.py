import sqlalchemy
from sqlalchemy import orm

from .db_session import SqlAlchemyBase


class Task_type(SqlAlchemyBase):
    __tablename__ = 'task_types'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    subject_name = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    name = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    tasks = orm.relation("Task", back_populates='type_obj')
