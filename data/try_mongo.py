import pymongo
from help_functions import get_db
from bson.objectid import ObjectId

db = get_db()

tasks = db.tasks
task_types = db["task_types"]
beautiful_links = db["beautiful_links"]


c = tasks.find_one({"_id": ObjectId("61f70f65c4ec2bcf79dbf2ac")})