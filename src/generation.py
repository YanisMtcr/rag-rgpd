import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


def _bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


class LLMGenerator:
    def __init__(self, model_name, quantize_4bit=True):
        self.model_name = model_name
        kwargs = {"device_map": "auto"}
        if quantize_4bit and torch.cuda.is_available():
            kwargs["quantization_config"] = _bnb_config()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt, max_new_tokens=512, temperature=0.2):
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)

        out = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = out[0, input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
