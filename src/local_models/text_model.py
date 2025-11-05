from io import BytesIO
import io, base64, torch, librosa
import time
from .base_local_model import BaseLocalModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage

class TextModel(BaseLocalModel):
    def __init__(self, model_id, *args, **kwargs):
        super().__init__(model_id, *args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side='left')

    def batch(self, messages_batch, *args, **kwargs):
        t0 = time.time()

        prompts_batch = []
        for messages in messages_batch:
            conversation = []
            human_content = ""

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    conversation.append({"role": "system", "content": msg.content})
                elif isinstance(msg, HumanMessage):
                    for item in msg.content:
                        if item.get("type") == "text":
                            human_content += item["text"]

            conversation.append({"role": "user", "content": human_content})

            prompt = self.tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            prompts_batch.append(prompt)

        t1 = time.time()
        inputs = self.tokenizer(
            prompts_batch,
            return_tensors="pt",
            padding=True
        )

        for k, v in inputs.items():
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **self.gen_params)

        t2 = time.time()

        attn = inputs["attention_mask"]  # [B, L]
        responses = []
        B = generated_ids.size(0)
        total_elapsed = t2 - t0

        for i in range(B):
            prompt_len = int(attn[i].sum().item())
            new_tokens = generated_ids[i, prompt_len:]
            text = self.tokenizer.batch_decode(
                new_tokens.unsqueeze(0),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            usage = {
                "prompt_tokens": prompt_len,
                "completion_tokens": int(new_tokens.size(0)),
                "total_tokens": int(prompt_len + new_tokens.size(0)),
            }

            responses.append(
                AIMessage(
                    content=text,
                    additional_kwargs={
                        "usage": usage,
                        "created": int(time.time()),
                        "model": self.model_id,
                        "time": {
                            "total_s": total_elapsed,
                            "preprocess_s": t1 - t0,
                            "generate_s": t2 - t1,
                        },
                    },
                )
            )

        return responses 

    def __call__(self, messages, *args, **kwargs):
        return self.batch([messages])
