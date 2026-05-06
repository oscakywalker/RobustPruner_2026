# Make llava import-safe across different codebase variants.
# Some forks/versions don't expose LlavaLlamaForCausalLM from llava.model.
try:
    from .model import LlavaLlamaForCausalLM  # optional
except Exception:
    # Keep package importable even if that symbol doesn't exist in this version.
    LlavaLlamaForCausalLM = None
