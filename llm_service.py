import os
import llm
import logging

DEFAULT_MODEL = "elysia"  # your Ollama model name; change if needed

class LLMService:
    """LLM wrapper using Simon Willison's `llm` Python API, with tool support."""

    def __init__(self, model_id: str = DEFAULT_MODEL):
        self.model_id = model_id
        logging.info(f"Initializing LLMService via llm: model='{self.model_id}'")
        try:
            self.model = llm.get_model(self.model_id)
        except Exception as e:
            logging.error(f"llm.get_model('{self.model_id}') failed: {e}")
            raise
        # **Force-enable tool support** even if model was not flagged for it
        if hasattr(self.model, "supports_tools"):
            try:
                self.model.supports_tools = True
                logging.info(f"Overriding supports_tools for model '{self.model_id}'")
            except Exception as e:
                logging.warning(f"Could not override supports_tools: {e}")

    def prompt(self, prompt: str, system: str | None = None, tools: list | None = None) -> str:
        """
        One-shot prompt. If tools provided, model may call them; use chain() for auto-return.
        """
        resp = self.model.prompt(prompt, system=system or "", tools=tools or [])
        # If tools were requested, caller can handle resp.tool_calls() etc.
        return resp.text()

    def chain(self, prompt: str, system: str | None = None, tools: list | None = None) -> str:
        """Try tools; if model rejects them, fall back to alternate model."""
        if tools:
            try:
                ch = self.model.chain(prompt, system=system or "", tools=tools)
                return "".join(ch)
            except Exception as e:
                logging.warning(f"Primary model error during tool usage: {e}")
                # Attempt fallback model (e.g., Mistral 7B) if primary fails
                try:
                    alt_model_id = os.getenv("ELYSIA_TOOL_MODEL", "mistral:7b")
                    logging.info(f"Falling back to alternate model '{alt_model_id}' for tool usage.")
                    alt_model = llm.get_model(alt_model_id)
                    if hasattr(alt_model, "supports_tools"):
                        alt_model.supports_tools = True
                    ch = alt_model.chain(prompt, system=system or "", tools=tools)
                    return "".join(ch)
                except Exception as e2:
                    logging.error(f"Fallback model '{alt_model_id}' failed as well: {e2}")
                    # Raise the original exception to be handled by caller
                    raise e
        # Fallback: no tools (if none provided or if all attempts failed)
        resp = self.model.prompt(prompt, system=system or "")
        return resp.text()
