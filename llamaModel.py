import transformers
import torch

class Llama3:

    def run(self, user_prompt):
        model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        messages = [
            {"role": "system", "content": "You are an expert lawyer specializing in Venezuelan law. You provide accurate legal advice and information related to Venezuelan laws and regulations."},
            {"role": "user", "content": user_prompt},
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=256,
        )
        print(outputs[0]["generated_text"][-1])
        return {"respond": outputs[0]["generated_text"][-1]}

