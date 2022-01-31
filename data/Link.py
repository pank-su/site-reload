class Link:
    def __init__(self, dict_=None):
        if (dict_ is None):
            self.link = 1
            self.task_id = None
        else:
            self.link = dict_["link"]
            self.task_id = dict_["task_id"]

    def asdict(self):
        return {"link": self.link, "task_id": self.task_id}
