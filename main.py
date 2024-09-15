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

# create dataframes with course names and skills
courses_data = {
    "Course Name": [
        "History of the Atlantic World",
        "Riemannian Geometry",
        "Operating Systems",
        "Food Science",
        "Compilers",
        "Intro to computer science",
    ]
}
skills_data = {"Skill": ["Math", "Computer Science"]}
courses_df = pd.DataFrame(courses_data)
skills_df = pd.DataFrame(skills_data)

print(f"skills_data: {skills_data}  |  courses_df: {courses_df} | skills_df: {skills_df}")

# lotus sem join
res = courses_df.sem_join(skills_df, "Taking {Course Name} will help me learn {Skill}")
print(res)




