class Link:
	_counter = 0
	def __init__(self):
		Link._counter += 1
		self.id = _counter
		self.link = 1
		self.task_id = None

	def __init__(self, dict_):
		Link._counter += 1
		self.id = dict_["_id"]
		self.link = dict_["link"]
		self.task_id = dict_["task_id"]

	def asdict(self):
		return {"id": self.id, "link": self.link, "task_id": self.task_id}