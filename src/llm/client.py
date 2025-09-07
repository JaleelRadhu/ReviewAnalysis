"""
Client adapters for different LLM backends.
This allows us to swap OpenAI, HuggingFace, or Fake LLMs
without changing the rest of the codebase.
"""

from typing import Protocol, Optional


class LLMClient(Protocol):
    """
    A minimal protocol all LLM clients must implement.
    """

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 128,
    ) -> str:
        """
        Generate text from a given prompt.
        Must return a string (model response).
        """
        ...


# ---------------- Fake client (for testing) ---------------- #

class FakeClient:
    """
    A dummy client useful for unit tests.
    Always returns a fixed label index (e.g., "1").
    """

    def __init__(self, fixed_response: str = "1"):
        self.fixed_response = fixed_response

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 128,
    ) -> str:
        return self.fixed_response


# ---------------- OpenAI client ---------------- #

class OpenAIClient:
    """
    Client wrapper around OpenAI's API.
    Requires `openai` package and API key set in env var OPENAI_API_KEY.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            import openai
        except ImportError:
            raise ImportError("You must `pip install openai` to use OpenAIClient.")

        self.openai = openai
        self.model = model
        if api_key:
            self.openai.api_key = api_key  # else relies on env var

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 128,
    ) -> str:
        """
        Send prompt to OpenAI ChatCompletion endpoint.
        """
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"].strip()
    
    
if __name__ == "__main__":
    # Simple test of the FakeClient
    client = FakeClient(fixed_response="Test response")
    print(client.generate("Hello, world!"))  # Should print "Test response