import pandas as pd
import requests
import json

import lotus
from lotus.models import OpenAIModel, E5Model


class OllamaModel(OpenAIModel):
    def __init__(self, base_url="http://localhost:11434", model="llama3.1"):
        self.base_url = base_url
        self.model = model

    def __call__(self, prompt, **kwargs):
        # Flatten the nested structure if necessary
        if isinstance(prompt, list) and isinstance(prompt[0], list):
            prompt = [item for sublist in prompt for item in sublist]

        # Convert all items to strings and join
        prompt = " ".join(str(item) for item in prompt)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }

        # Add any additional parameters from kwargs
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            raise Exception(f"Error from Ollama API: {response.text}")


# Create an OllamaModel instance
lm = OllamaModel(model="llama3.1")  # Use the appropriate model name

rm = E5Model(device="cpu")

lotus.settings.configure(lm=lm, rm=rm, model_params={"temperature": 0.0, "max_tokens": 256})

print(lotus.settings.keys())

data = {
    "Course Name": [
        "Probability and Random Processes",
        "Optimization Methods in Engineering",
        "Digital Design and Integrated Circuits",
        "Computer Security",
        "Operating Systems and Systems Programming",
        "Compilers",
        "Computer Networks",
        "Deep Learning",
        "Graphics",
        "Databases",
        "Art History",
    ]
}
df = pd.DataFrame(data)
print(df)

courses_w_math = df.sem_filter("{Course Name} requires a lot of math")
print(f"courses requires a lot of math: {courses_w_math}")








