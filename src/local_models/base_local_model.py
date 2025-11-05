class BaseLocalModel:
    def __init__(self, model_id, *args, **kwargs):
        self.model_id = model_id
        self.gen_params = dict(max_new_tokens=4096, do_sample=True, temperature=0.7)
        self.gen_params.update(kwargs)
        if "max_tokens" in self.gen_params:
            self.gen_params["max_new_tokens"] = self.gen_params.pop("max_tokens")

        if "temperature" in self.gen_params and self.gen_params["temperature"] == 0:
            self.gen_params["do_sample"] = False
            self.gen_params.pop("temperature")

    def batch(self, messages_batch, *args, **kwargs):
        raise NotImplementedError

    def invoke(self, messages, *args, **kwargs):
        return self.batch([messages], *args, **kwargs)[0]

