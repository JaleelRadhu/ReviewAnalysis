"""
Client adapters for different LLM backends.
Supports OpenAI, Hugging Face (including LLaMA), and Fake LLMs.
Loads API keys from .env automatically.
"""

from typing import Protocol, Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# ---------------- Protocol ---------------- #
class LLMClient(Protocol):
    """
    Minimal protocol all LLM clients must implement.
    """

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 128,
    ) -> str:
        ...


# ---------------- Fake client ---------------- #
class FakeClient:
    """A dummy client useful for testing."""

    def __init__(self, fixed_response: str = "1"):
        self.fixed_response = fixed_response

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 128) -> str:
        return self.fixed_response


# ---------------- OpenAI client ---------------- #
class OpenAIClient:
    """Client wrapper around OpenAI's API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("You must `pip install openai` to use OpenAIClient.")

        self.openai = openai
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in .env or env vars.")
        self.openai.api_key = self.api_key

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 128) -> str:
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()


# ---------------- Hugging Face client ---------------- #
class HFClient:
    """Hugging Face client for text-generation (supports LLaMA)."""

    def __init__(self, model_name: str = "gpt2", device: int = 0):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("You must `pip install transformers` to use HFClient.")

        hf_token = os.getenv("HF_API_KEY")  # Optional, for private models
        print(hf_token)
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,  # -1 = CPU, 0 = first GPU
            token=hf_token
        )

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 128) -> str:
        print(prompt)
        outputs = self.generator(
            prompt,
            max_new_tokens=max_tokens,
            # max_length=len(prompt.split()) + max_tokens,
            temperature=temperature,
            num_return_sequences=1,
            do_sample=True if temperature > 0 else False,
        )
        return outputs[0]["generated_text"][len(prompt):].strip()


# ---------------- Factory ---------------- #
def get_llm_client(config: dict) -> LLMClient:
    backend = config["llm"]["backend"]
    model = config["llm"].get("model", None)
    device = config["llm"].get("device", 0) # For HFClient

    if backend == "openai":
        return OpenAIClient(model=model)
    elif backend == "hf":
        return HFClient(model_name=model, device=device)
    elif backend == "fake":
        return FakeClient()
    else:
        raise ValueError(f"Unknown LLM backend: {backend}")


# ---------------- Test ---------------- #
if __name__ == "__main__":
    import yaml
    from src.util.config import load_config
    import sys
    sys.path.append("/home/abdullahm/jaleel/Review_analysis")
    cfg = load_config()
    print(cfg["llm"]["model"])

    client = get_llm_client(cfg)
    
    prompt = """
You are a helpful AI assistant named LLaMA 3.1 8B-Instruct. Introduce yourself in a professional yet friendly way. Include the following details:

1. Your name and version.
2. Your purpose and capabilities.
3. The types of tasks you can help with.
4. How you approach problem-solving or reasoning.
5. A short friendly note to the user.

Format your response clearly using numbered points or bullet points.

"""
    print(client.generate(prompt, temperature=0.5, max_tokens=100))
