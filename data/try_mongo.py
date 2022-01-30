import pymongo
from help_functions import get_db

db = get_db()

tasks = db["tasks"]
task_types = db["task_types"]
beautiful_links = db["beautiful_links"]

#tasks.insert_one({"id": 4, "type": 1, "info":"1", "solution_path": "q"})

c = tasks.find()
print(len(c.distinct("id")))
task_types.insert_many([{"id": 1, "subject_name": "phys", "name": "task_1"}, {"id": 2, "subject_name": "math", "name": "task_1"}])