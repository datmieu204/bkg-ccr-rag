import os
from typing import Any

import torch


class CacheManager:
	def __init__(self, cache_dir: str, enabled: bool = True) -> None:
		self.cache_dir = cache_dir
		self.enabled = enabled
		if self.enabled:
			os.makedirs(self.cache_dir, exist_ok=True)

	def _path(self, key: str) -> str:
		safe_key = key.replace("/", "_")
		return os.path.join(self.cache_dir, f"{safe_key}.pt")

	def has(self, key: str) -> bool:
		if not self.enabled:
			return False
		return os.path.exists(self._path(key))

	def load(self, key: str) -> Any:
		if not self.enabled:
			raise RuntimeError("Cache is disabled")
		return torch.load(self._path(key), weights_only=False)

	def save(self, key: str, obj: Any) -> None:
		if not self.enabled:
			return
		torch.save(obj, self._path(key))
