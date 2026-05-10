__version__ = "0.0.3"
__author__ = "Jianbai Ye"
__url__ = "https://github.com/gusye1234/nano-graphrag"

# dp stands for data pack

__all__ = ["GraphRAG", "QueryParam"]


def __getattr__(name):
	if name in {"GraphRAG", "QueryParam"}:
		from .graphrag import GraphRAG, QueryParam

		return {"GraphRAG": GraphRAG, "QueryParam": QueryParam}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")