import sqlalchemy
from sqlalchemy import orm

from .db_session import SqlAlchemyBase


class Subject(SqlAlchemyBase):
    __tablename__ = 'subjects'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    name = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    tasks = orm.relation("Task", back_populates='subject_obj')
