import sqlalchemy
from sqlalchemy import orm

from .db_session import SqlAlchemyBase


class Link(SqlAlchemyBase):
    __tablename__ = 'beautiful_links'

    id = sqlalchemy.Column(sqlalchemy.Integer,
                           primary_key=True, autoincrement=True)
    link = sqlalchemy.Column(sqlalchemy.String, nullable=False)
    task_id = sqlalchemy.Column(sqlalchemy.Integer, sqlalchemy.ForeignKey("tasks.id"),
                                nullable=False)

    task_obj = orm.relation('Task')
