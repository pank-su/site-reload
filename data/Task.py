import json

class Task:
	_counter = 0
	def __init__(self, dict_ = None):
		if (dict_ is None):
			self.type = 1
			self.info = None
			self.solution_path = None
		else:
			self.type = dict_["type"]
			self.info = dict_["info"]
			self.solution_path = dict_["solution_path"]


	def asdict(self):
		return {"type": self.type, "info": self.info, "solution_path": self.solution_path}