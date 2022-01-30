class Task:
	_counter = 0
	def __init__(self):
		Task._counter += 1
		self.id = _counter
		self.type = 1
		self.info = None
		self.solution_path = f'static/files/{self.id}.json'

	def __init__(self, dict_):
		Task._counter += 1
		self.id = dict_["_id"]
		self.type = dict_["type"]
		self.info = dict_["info"]
		self.solution_path = dict_["solution_path"]

	def asdict(self):
		return {"id": self.id, "type": self.type, "info": self.info, "solution_path": self.solution_path}