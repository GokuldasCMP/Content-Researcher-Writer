# memory_layer.py

class MemoryLayer:
    def __init__(self):
        self.history = []  # list of memory items

    def remember(self, step_name, content):
        self.history.append({
            "step": step_name,
            "content": content
        })

    def get_history(self):
        return self.history

    def get_last(self, step_name=None):
        if step_name:
            for item in reversed(self.history):
                if item["step"] == step_name:
                    return item["content"]
        return self.history[-1]["content"] if self.history else None

    def inject_memory(self, input_data):
        input_data["memory"] = self.history
        return input_data
