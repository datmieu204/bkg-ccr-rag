from typing import Iterable, List, Optional


def normalize_text(text: str) -> str:
	return " ".join((text or "").strip().split())


def truncate_texts(
	texts: Iterable[str],
	max_items: Optional[int] = None,
	max_chars: Optional[int] = None,
) -> List[str]:
	normalized = [normalize_text(t) for t in texts if t and t.strip()]

	if max_items is not None:
		normalized = normalized[:max_items]

	if max_chars is None:
		return normalized

	out: List[str] = []
	total = 0
	for text in normalized:
		if total >= max_chars:
			break
		remaining = max_chars - total
		if len(text) > remaining:
			out.append(text[:remaining])
			total = max_chars
			break
		out.append(text)
		total += len(text)
	return out
