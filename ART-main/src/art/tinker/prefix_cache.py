from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PrefixEntry:
    rendered_len: int
    raw_prefix: tuple[int, ...]


@dataclass
class PrefixCacheStats:
    max_entries: int
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    replaced_entries: int = 0
    evictions: int = 0
    splits: int = 0
    pruned_nodes: int = 0
    merged_nodes: int = 0
    lru_repairs: int = 0


class _RadixEdge:
    __slots__ = ("label", "child")

    def __init__(self, label: tuple[int, ...], child: _RadixNode) -> None:
        self.label = label
        self.child = child


class _RadixNode:
    __slots__ = ("entry", "children", "parent", "parent_token")

    def __init__(
        self, parent: _RadixNode | None = None, parent_token: int | None = None
    ) -> None:
        self.entry: PrefixEntry | None = None
        self.children: dict[int, _RadixEdge] = {}
        self.parent = parent
        self.parent_token = parent_token


def _common_prefix_len(
    tokens: Sequence[int], start: int, label: tuple[int, ...]
) -> int:
    max_len = min(len(tokens) - start, len(label))
    i = 0
    while i < max_len and tokens[start + i] == label[i]:
        i += 1
    return i


class LRUTrieCache:
    """LRU-bounded radix trie for token sequence rewrites."""

    def __init__(self, max_entries: int = 16_384) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._root = _RadixNode()
        self._lru: OrderedDict[_RadixNode, None] = OrderedDict()
        self._max_entries = max_entries
        self.stats = PrefixCacheStats(max_entries=max_entries)

    def lookup(self, rendered_tokens: Sequence[int]) -> PrefixEntry | None:
        self.stats.lookups += 1
        node = self._root
        idx = 0
        best_node = None
        while idx < len(rendered_tokens):
            edge = node.children.get(rendered_tokens[idx])
            if edge is None:
                break
            matched = _common_prefix_len(rendered_tokens, idx, edge.label)
            if matched != len(edge.label):
                break
            idx += matched
            node = edge.child
            if node.entry is not None:
                best_node = node
        if best_node is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        try:
            self._lru.move_to_end(best_node)
        except KeyError:
            self.stats.lru_repairs += 1
            self._lru[best_node] = None
            self._lru.move_to_end(best_node)
            self._evict()
        return best_node.entry

    def insert(self, rendered_prefix: Sequence[int], raw_prefix: Sequence[int]) -> None:
        self.stats.inserts += 1
        node = self._root
        idx = 0
        while idx < len(rendered_prefix):
            token = rendered_prefix[idx]
            edge = node.children.get(token)
            if edge is None:
                child = _RadixNode(parent=node, parent_token=token)
                node.children[token] = _RadixEdge(tuple(rendered_prefix[idx:]), child)
                node = child
                idx = len(rendered_prefix)
                break

            matched = _common_prefix_len(rendered_prefix, idx, edge.label)
            if matched == len(edge.label):
                idx += matched
                node = edge.child
                continue

            mid = _RadixNode(parent=node, parent_token=token)
            self.stats.splits += 1
            old_suffix = edge.label[matched:]
            old_child = edge.child
            old_child.parent = mid
            old_child.parent_token = old_suffix[0]
            mid.children[old_suffix[0]] = _RadixEdge(old_suffix, old_child)
            edge.label = edge.label[:matched]
            edge.child = mid
            node = mid
            idx += matched
            if idx < len(rendered_prefix):
                new_token = rendered_prefix[idx]
                child = _RadixNode(parent=node, parent_token=new_token)
                node.children[new_token] = _RadixEdge(
                    tuple(rendered_prefix[idx:]), child
                )
                node = child
            break

        if node.entry is not None:
            self.stats.replaced_entries += 1
        node.entry = PrefixEntry(
            rendered_len=len(rendered_prefix), raw_prefix=tuple(raw_prefix)
        )
        self._lru[node] = None
        self._lru.move_to_end(node)
        self._evict()

    def _evict(self) -> None:
        while len(self._lru) > self._max_entries:
            old_node, _ = self._lru.popitem(last=False)
            self.stats.evictions += 1
            old_node.entry = None
            self._prune(old_node)

    def _prune(self, node: _RadixNode) -> None:
        # Collapse empty branches after eviction so the bounded cache stays bounded.
        while node.parent is not None:
            parent = node.parent
            parent_token = node.parent_token
            assert parent_token is not None

            if node.entry is None and not node.children:
                del parent.children[parent_token]
                self.stats.pruned_nodes += 1
                node = parent
                continue

            if node.entry is None and len(node.children) == 1:
                _, child_edge = next(iter(node.children.items()))
                parent_edge = parent.children[parent_token]
                parent_edge.label = parent_edge.label + child_edge.label
                parent_edge.child = child_edge.child
                child_edge.child.parent = parent
                child_edge.child.parent_token = parent_token
                self.stats.merged_nodes += 1
                node = parent
                continue

            break

    def snapshot_stats(self) -> dict[str, int | float]:
        hit_rate = self.stats.hits / self.stats.lookups if self.stats.lookups else 0.0
        return {
            "enabled": True,
            "max_entries": self.stats.max_entries,
            "current_entries": len(self._lru),
            "node_count": self._node_count(),
            "lookups": self.stats.lookups,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": hit_rate,
            "inserts": self.stats.inserts,
            "replaced_entries": self.stats.replaced_entries,
            "evictions": self.stats.evictions,
            "splits": self.stats.splits,
            "pruned_nodes": self.stats.pruned_nodes,
            "merged_nodes": self.stats.merged_nodes,
            "lru_repairs": self.stats.lru_repairs,
        }

    def _node_count(self) -> int:
        count = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(edge.child for edge in node.children.values())
        return count
