import pymongo
from help_functions import get_db

db = get_db()

tasks = db["tasks"]
task_types = db["task_types"]
beautiful_links = db["beautiful_links"]

#tasks.insert_one({"id": 4, "type": 1, "info":"1", "solution_path": "q"})

c = tasks.find()
print(len(c.distinct("id")))
