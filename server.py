"""
Memory Management MCP Server for Multi-Agent Development Framework

This MCP server provides centralized memory management for agent collaboration,
handling storage, retrieval, pruning, and cross-agent queries for the development team.

Redesigned version with streamlined tool set:
- Removed: compress_old_memories, deduplicate_memories, bulk_update_memories, 
           get_memory_health, load_memories_batch
- Kept: store_memory, load_agent_context, search_memories, promote_memory,
        demote_memory, cleanup_memories, get_memory_stats, query_cross_agent_memory
- Added: get_compression_candidates, store_extracted_facts
"""

from fastmcp import FastMCP
from typing import List, Dict, Optional, Any, Literal
from pydantic import Field, BaseModel, ConfigDict
from datetime import datetime
from pathlib import Path
import json
import hashlib
import asyncio
import aiofiles
import aiofiles.os
from collections import defaultdict
import os

# Initialize FastMCP server
mcp = FastMCP("agent_memory_mcp")

# Configuration
MEMORY_BASE_PATH = Path(os.getenv("MEMORY_BASE_PATH", ".claude/agents/memories"))
MEMORY_BASE_PATH.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_PATH = MEMORY_BASE_PATH / "audit.log"

# File lock for async safety
file_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
audit_lock = asyncio.Lock()

# Memory limits configuration
MEMORY_CONFIG = {
    "limits": {
        "recent_max_items": 50,
        "medium_term_max_items": 200,
        "long_term_max_items": 500,
        "total_max_size_kb": 150
    },
    "token_budgets": {
        "core_memory": 2000,
        "task_context": 5000
    },
    "compression": {
        "min_age_days": 7,
        "min_group_size": 3,
        "max_access_count": 2,
        "fact_min_words": 5,
        "fact_max_words": 15
    }
}

# Valid memory layers
MEMORY_LAYERS = Literal["core", "recent", "medium_term", "long_term", "compost"]
WRITABLE_LAYERS = Literal["recent", "medium_term", "long_term", "compost", "core"]
PROMOTABLE_TARGET_LAYERS = Literal["medium_term", "long_term"]


# ============================================================================
# Data Models
# ============================================================================

class MemoryItem(BaseModel):
    """Schema for a memory item."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    id: str
    layer: str
    memory_type: str
    content: str
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created: str
    last_accessed: str
    access_count: int = Field(default=0, ge=0)
    context: Dict[str, Any] = Field(default_factory=dict)
    size_estimate_tokens: int = Field(default=0, ge=0)
    version: int = Field(default=1, ge=1)
    previous_version_id: Optional[str] = None
    related_to: List[str] = Field(default_factory=list)
    content_hash: Optional[str] = None


class AgentMemory(BaseModel):
    """Schema for an agent's complete memory structure."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    agent_id: str
    role: str = "unknown"
    last_updated: str
    memory_layers: Dict[str, List[MemoryItem]] = Field(
        default_factory=lambda: {
            "core": [],
            "recent": [],
            "medium_term": [],
            "long_term": [],
            "compost": []
        }
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Input Models for Tools
# ============================================================================

class StoreMemoryInput(BaseModel):
    """Input for storing a new memory item."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent storing the memory", min_length=1, max_length=100)
    layer: WRITABLE_LAYERS = Field(..., description="Memory persistence layer: 'recent', 'medium_term', 'long_term', 'compost', or 'core'")
    memory_type: str = Field(..., description="Type of memory (e.g., 'decision', 'implementation', 'bug_fix', 'lesson_learned')", min_length=1, max_length=50)
    content: str = Field(..., description="The actual memory content to store", min_length=1)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and filtering", max_length=20)
    importance: float = Field(default=0.5, description="Importance score from 0.0 (low) to 1.0 (critical)", ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context (task_id, related_files, related_agents)")
    related_to: List[str] = Field(default_factory=list, description="IDs of related memories", max_length=50)


class LoadAgentContextInput(BaseModel):
    """Input for loading relevant memory context for a task."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    task_description: Optional[str] = Field(None, description="Description of current task for relevance filtering")
    task_tags: List[str] = Field(default_factory=list, description="Tags related to current task", max_length=20)
    max_tokens: int = Field(default=5000, description="Maximum tokens to return", ge=100, le=50000)
    include_layers: List[str] = Field(
        default=["core", "recent", "medium_term"],
        description="Which memory layers to include"
    )


class SearchMemoriesInput(BaseModel):
    """Input for searching agent memories."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    query: str = Field(..., description="Search query string", min_length=1)
    layers: Optional[List[str]] = Field(None, description="Layers to search in (defaults to all except compost)")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=10, description="Maximum results to return", ge=1, le=100)
    min_importance: float = Field(default=0.0, description="Minimum importance score filter", ge=0.0, le=1.0)


class PromoteMemoryInput(BaseModel):
    """Input for promoting a memory to a higher persistence layer."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    memory_id: str = Field(..., description="ID of memory to promote", min_length=1)
    to_layer: PROMOTABLE_TARGET_LAYERS = Field(..., description="Target layer: 'medium_term' or 'long_term'")
    reason: Optional[str] = Field(None, description="Reason for promotion")


class DemoteMemoryInput(BaseModel):
    """Input for demoting a memory to compost or deleting it."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    memory_id: str = Field(..., description="ID of memory to demote", min_length=1)
    to_compost: bool = Field(default=True, description="Move to compost (True) or delete entirely (False)")
    reason: Optional[str] = Field(None, description="Reason for demotion")


class CleanupMemoriesInput(BaseModel):
    """Input for running automatic memory cleanup."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    cleanup_type: Literal["daily", "sprint_end", "quarterly"] = Field(
        default="daily",
        description="Type of cleanup: 'daily' (recent→medium), 'sprint_end' (medium→long/compost), 'quarterly' (clean compost)"
    )
    dry_run: bool = Field(default=False, description="Preview changes without applying them")


class GetMemoryStatsInput(BaseModel):
    """Input for getting memory statistics."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    include_health_metrics: bool = Field(default=False, description="Include health diagnostics (redundancy, stale memories)")


class QueryCrossAgentMemoryInput(BaseModel):
    """Input for searching memories across multiple agents."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    query: str = Field(..., description="Search query string", min_length=1)
    agent_ids: Optional[List[str]] = Field(None, description="Specific agents to query (defaults to all)")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    limit: int = Field(default=20, description="Maximum results to return", ge=1, le=100)


class GetCompressionCandidatesInput(BaseModel):
    """Input for identifying groups of memories suitable for fact extraction."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    min_age_days: int = Field(
        default=7, 
        description="Minimum age in days for memories to be considered", 
        ge=1, 
        le=365
    )
    min_group_size: int = Field(
        default=3, 
        description="Minimum number of related memories to form a compression group", 
        ge=2, 
        le=20
    )
    max_access_count: int = Field(
        default=2, 
        description="Maximum access count - frequently accessed memories are excluded", 
        ge=0, 
        le=100
    )
    layers: List[str] = Field(
        default=["medium_term", "long_term"],
        description="Layers to search for compression candidates"
    )
    max_groups: int = Field(
        default=5, 
        description="Maximum number of candidate groups to return", 
        ge=1, 
        le=20
    )


class ExtractedFact(BaseModel):
    """Schema for an atomic extracted fact."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    fact: str = Field(
        ...,
        description="Atomic fact statement (5-15 words)",
        min_length=10,
        max_length=150
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for this specific fact",
        max_items=10
    )
    importance: float = Field(
        default=0.6,
        description="Importance score for this fact",
        ge=0.0,
        le=1.0
    )


class StoreExtractedFactsInput(BaseModel):
    """Input for storing extracted facts and removing original memories."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    original_memory_ids: List[str] = Field(
        ...,
        description="IDs of original memories being compressed",
        min_items=1,
        max_items=50
    )
    facts: List[ExtractedFact] = Field(
        ...,
        description="List of extracted atomic facts (5-15 words each)",
        min_items=1,
        max_items=100
    )
    context_summary: Optional[str] = Field(
        None,
        description="Optional 1-2 sentence summary preserving essential reasoning context",
        max_length=500
    )
    memory_type: str = Field(
        default="compressed_facts",
        description="Type for the new compressed memory",
        min_length=1,
        max_length=50
    )


# ============================================================================
# Helper Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """Estimate token count based on content type.
    
    Code typically has more tokens due to symbols, while regular text is more efficient.
    """
    if '{' in text or 'def ' in text or 'class ' in text:
        return int(len(text) / 4)
    return int(len(text) / 3.5)


def generate_memory_id() -> str:
    """Generate a unique memory ID using timestamp and hash."""
    timestamp = datetime.now().isoformat()
    random_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"mem_{random_hash}"


def generate_content_hash(content: str) -> str:
    """Generate hash of content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()


def get_memory_file_path(agent_id: str) -> Path:
    """Get the file path for an agent's memory storage."""
    return MEMORY_BASE_PATH / f"{agent_id}-memory.json"


async def load_agent_memory(agent_id: str) -> AgentMemory:
    """Load an agent's memory from disk with async safety.
    
    Creates a new memory structure if the agent doesn't exist.
    """
    memory_file = get_memory_file_path(agent_id)

    async with file_locks[agent_id]:
        try:
            file_exists = await aiofiles.os.path.exists(memory_file)
        except Exception:
            file_exists = False

        if not file_exists:
            memory = AgentMemory(
                agent_id=agent_id,
                role="unknown",
                last_updated=datetime.now().isoformat()
            )
            await _save_agent_memory_unlocked(memory)
            return memory

        try:
            async with aiofiles.open(memory_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                return AgentMemory(**data)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Corrupted memory file for {agent_id}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load memory for {agent_id}: {e}")


async def _save_agent_memory_unlocked(memory: AgentMemory) -> None:
    """Save agent memory without acquiring lock (caller must hold lock)."""
    memory.last_updated = datetime.now().isoformat()
    memory_file = get_memory_file_path(memory.agent_id)

    try:
        temp_file = memory_file.with_suffix('.tmp')
        async with aiofiles.open(temp_file, 'w') as f:
            await f.write(json.dumps(memory.model_dump(), indent=2))
        await aiofiles.os.replace(temp_file, memory_file)
    except Exception as e:
        raise RuntimeError(f"Failed to save memory for {memory.agent_id}: {e}")


async def save_agent_memory(memory: AgentMemory) -> None:
    """Save an agent's memory to disk with async safety."""
    async with file_locks[memory.agent_id]:
        await _save_agent_memory_unlocked(memory)


def filter_memories_by_tags(memories: List[MemoryItem], tags: List[str]) -> List[MemoryItem]:
    """Filter memories that match any of the provided tags."""
    if not tags:
        return memories
    return [mem for mem in memories if any(tag in mem.tags for tag in tags)]


def search_memories_by_query(memories: List[MemoryItem], query: str) -> List[MemoryItem]:
    """Search memories by keyword matching in content and tags."""
    query_lower = query.lower()
    results = []
    
    for mem in memories:
        if (query_lower in mem.content.lower() or
            any(query_lower in tag.lower() for tag in mem.tags)):
            results.append(mem)
    
    return results


def calculate_relevance_score(memory: MemoryItem) -> float:
    """Calculate relevance score for memory ranking.
    
    Combines importance, recency, and access frequency.
    """
    score = memory.importance
    
    try:
        last_accessed = datetime.fromisoformat(memory.last_accessed)
        days_since_access = (datetime.now() - last_accessed).days
        recency_boost = max(0, 0.2 * (30 - days_since_access) / 30)
    except ValueError:
        recency_boost = 0
    
    access_boost = min(0.1 * memory.access_count, 0.3)
    
    return min(1.0, score + recency_boost + access_boost)


def find_memory_by_id(memory: AgentMemory, memory_id: str) -> tuple[Optional[MemoryItem], Optional[str]]:
    """Find a memory item by ID across all layers.
    
    Returns (memory_item, layer_name) or (None, None) if not found.
    """
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id == memory_id:
                return item, layer_name
    return None, None


def calculate_tag_overlap(tags1: List[str], tags2: List[str]) -> float:
    """Calculate Jaccard similarity between two tag sets."""
    if not tags1 or not tags2:
        return 0.0
    set1, set2 = set(tags1), set(tags2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def group_memories_by_similarity(
    memories: List[MemoryItem],
    min_group_size: int
) -> List[List[MemoryItem]]:
    """Group memories by memory_type and tag overlap.

    Uses a simple clustering approach based on memory_type match and
    tag overlap threshold of 0.3 (Jaccard similarity).
    """
    if not memories:
        return []

    type_groups: Dict[str, List[MemoryItem]] = defaultdict(list)
    for mem in memories:
        type_groups[mem.memory_type].append(mem)

    result_groups = []

    for memory_type, type_memories in type_groups.items():
        if len(type_memories) < min_group_size:
            continue

        # Within each type, cluster by tag similarity
        used = set()
        for i, mem in enumerate(type_memories):
            if i in used:
                continue

            cluster = [mem]
            used.add(i)

            for j, other in enumerate(type_memories):
                if j in used:
                    continue

                # Check tag overlap with any member of the cluster
                for cluster_mem in cluster:
                    if calculate_tag_overlap(cluster_mem.tags, other.tags) >= 0.3:
                        cluster.append(other)
                        used.add(j)
                        break

            if len(cluster) >= min_group_size:
                result_groups.append(cluster)

    return result_groups


async def audit(agent_id: str, action: str, details: str = "") -> None:
    """Log all memory operations for debugging and accountability.

    Writes audit entries to a persistent log file with format:
    timestamp | agent_id | action | details

    Args:
        agent_id: ID of the agent performing the operation
        action: Type of action (e.g., 'store_memory', 'promote_memory')
        details: Additional context about the operation
    """
    timestamp = datetime.now().isoformat()
    entry = f"{timestamp} | {agent_id} | {action} | {details}\n"

    async with audit_lock:
        try:
            async with aiofiles.open(AUDIT_LOG_PATH, 'a') as f:
                await f.write(entry)
        except Exception as e:
            # Don't fail the operation if audit logging fails
            # In production, consider logging this to a separate error log
            print(f"Audit log write failed: {e}", flush=True)


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="store_memory",
    annotations={
        "title": "Store Agent Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def store_memory(params: StoreMemoryInput) -> Dict[str, Any]:
    """Store a new memory item for an agent.
    
    Automatically detects and updates duplicate content (same hash) instead of 
    creating new entries. Enforces layer-specific item limits with automatic 
    pruning of oldest items.
    
    Args:
        params: StoreMemoryInput containing agent_id, layer, memory_type, content,
                tags, importance, context, and related_to fields.
    
    Returns:
        Dict with success status, memory_id, agent_id, layer, action taken 
        ('created_new' or 'updated_existing'), and size_tokens.
    """
    memory = await load_agent_memory(params.agent_id)
    
    content_hash = generate_content_hash(params.content)

    for item in memory.memory_layers[params.layer]:
        if item.content_hash == content_hash:
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            await save_agent_memory(memory)

            # Audit log
            await audit(
                params.agent_id,
                "store_memory",
                f"updated_existing | layer={params.layer} | type={params.memory_type} | mem_id={item.id}"
            )

            return {
                "success": True,
                "memory_id": item.id,
                "agent_id": params.agent_id,
                "layer": params.layer,
                "action": "updated_existing",
                "size_tokens": item.size_estimate_tokens
            }

    memory_item = MemoryItem(
        id=generate_memory_id(),
        layer=params.layer,
        memory_type=params.memory_type,
        content=params.content,
        tags=params.tags,
        importance=params.importance,
        created=datetime.now().isoformat(),
        last_accessed=datetime.now().isoformat(),
        access_count=0,
        context=params.context,
        size_estimate_tokens=estimate_tokens(params.content),
        content_hash=content_hash,
        related_to=params.related_to
    )
    
    memory.memory_layers[params.layer].append(memory_item)
    
    # Enforce layer limits
    max_items = MEMORY_CONFIG["limits"].get(f"{params.layer}_max_items")
    if max_items and len(memory.memory_layers[params.layer]) > max_items:
        memory.memory_layers[params.layer] = memory.memory_layers[params.layer][-max_items:]

    await save_agent_memory(memory)

    # Audit log
    await audit(
        params.agent_id,
        "store_memory",
        f"created_new | layer={params.layer} | type={params.memory_type} | mem_id={memory_item.id} | tokens={memory_item.size_estimate_tokens}"
    )

    return {
        "success": True,
        "memory_id": memory_item.id,
        "agent_id": params.agent_id,
        "layer": params.layer,
        "action": "created_new",
        "size_tokens": memory_item.size_estimate_tokens
    }


@mcp.tool(
    name="load_agent_context",
    annotations={
        "title": "Load Agent Context",
        "readOnlyHint": False,  # Updates access counts
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def load_agent_context(params: LoadAgentContextInput) -> Dict[str, Any]:
    """Load relevant memory context for an agent's current task.
    
    Retrieves memories from specified layers, prioritized by relevance to the 
    task. Stays within token budget and updates access counts.
    
    Args:
        params: LoadAgentContextInput with agent_id, task_description, task_tags,
                max_tokens, and include_layers.
    
    Returns:
        Dict with agent_id, context list (layer, type, content, tags, importance),
        tokens_used, items_loaded count, and memory_ids list.
    """
    memory = await load_agent_memory(params.agent_id)
    
    context_parts = []
    tokens_used = 0
    items_loaded = []
    
    # 1. Always include core memory first
    if "core" in params.include_layers:
        for item in memory.memory_layers["core"]:
            if tokens_used + item.size_estimate_tokens <= params.max_tokens:
                context_parts.append({
                    "layer": "core",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
    
    # 2. Load recent memories with task relevance filtering
    if "recent" in params.include_layers:
        recent_memories = memory.memory_layers["recent"]
        if params.task_tags:
            recent_memories = filter_memories_by_tags(recent_memories, params.task_tags)
        
        recent_memories = sorted(recent_memories, key=calculate_relevance_score, reverse=True)
        
        for item in recent_memories:
            if tokens_used + item.size_estimate_tokens <= params.max_tokens - 500:
                context_parts.append({
                    "layer": "recent",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    # 3. Load medium-term memories if space available
    if "medium_term" in params.include_layers and tokens_used < params.max_tokens - 1000:
        medium_memories = memory.memory_layers["medium_term"]
        
        if params.task_description:
            medium_memories = search_memories_by_query(medium_memories, params.task_description)
        if params.task_tags:
            medium_memories = filter_memories_by_tags(medium_memories, params.task_tags)
        
        medium_memories = sorted(medium_memories, key=calculate_relevance_score, reverse=True)
        
        for item in medium_memories[:5]:
            if tokens_used + item.size_estimate_tokens <= params.max_tokens - 500:
                context_parts.append({
                    "layer": "medium_term",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    # 4. Load long-term memories if space available
    if "long_term" in params.include_layers and tokens_used < params.max_tokens - 500:
        long_term_memories = memory.memory_layers["long_term"]
        
        if params.task_description:
            long_term_memories = search_memories_by_query(long_term_memories, params.task_description)
        if params.task_tags:
            long_term_memories = filter_memories_by_tags(long_term_memories, params.task_tags)
        
        long_term_memories = sorted(
            long_term_memories,
            key=lambda x: (x.importance, calculate_relevance_score(x)),
            reverse=True
        )
        
        for item in long_term_memories[:3]:
            if tokens_used + item.size_estimate_tokens <= params.max_tokens - 500:
                context_parts.append({
                    "layer": "long_term",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    await save_agent_memory(memory)

    return {
        "agent_id": params.agent_id,
        "context": context_parts,
        "tokens_used": tokens_used,
        "items_loaded": len(items_loaded),
        "memory_ids": items_loaded
    }


@mcp.tool(
    name="search_memories",
    annotations={
        "title": "Search Agent Memories",
        "readOnlyHint": False,  # Updates access counts
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def search_memories(params: SearchMemoriesInput) -> Dict[str, Any]:
    """Search across agent memories using keywords and filters.
    
    Performs keyword matching on content and tags, with optional filtering 
    by layer, tags, and importance threshold.
    
    Args:
        params: SearchMemoriesInput with agent_id, query, layers, tags, limit,
                and min_importance.
    
    Returns:
        Dict with agent_id, query, results_count, and memories list containing
        memory_id, layer, type, content, tags, importance, relevance_score,
        created, and context for each match.
    """
    memory = await load_agent_memory(params.agent_id)
    
    search_layers = params.layers if params.layers else ["recent", "medium_term", "long_term"]
    
    all_memories = []
    for layer in search_layers:
        if layer in memory.memory_layers:
            all_memories.extend(memory.memory_layers[layer])
    
    results = search_memories_by_query(all_memories, params.query)
    
    if params.tags:
        results = filter_memories_by_tags(results, params.tags)
    
    results = [m for m in results if m.importance >= params.min_importance]
    results = sorted(results, key=calculate_relevance_score, reverse=True)
    results = results[:params.limit]
    
    # Update access counts
    for mem in results:
        mem.access_count += 1
        mem.last_accessed = datetime.now().isoformat()
    await save_agent_memory(memory)
    
    formatted_results = [
        {
            "memory_id": mem.id,
            "layer": mem.layer,
            "type": mem.memory_type,
            "content": mem.content,
            "tags": mem.tags,
            "importance": mem.importance,
            "relevance_score": round(calculate_relevance_score(mem), 3),
            "created": mem.created,
            "context": mem.context
        }
        for mem in results
    ]
    
    return {
        "agent_id": params.agent_id,
        "query": params.query,
        "results_count": len(formatted_results),
        "memories": formatted_results
    }


@mcp.tool(
    name="promote_memory",
    annotations={
        "title": "Promote Memory Layer",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def promote_memory(params: PromoteMemoryInput) -> Dict[str, Any]:
    """Promote a memory item to a higher persistence layer.
    
    Use when a memory proves valuable and should be retained longer. 
    Cannot promote from long_term or core layers.
    
    Args:
        params: PromoteMemoryInput with agent_id, memory_id, to_layer, and reason.
    
    Returns:
        Dict with success status, memory_id, from_layer, to_layer, and reason.
    
    Raises:
        RuntimeError: If memory not found or cannot be promoted from current layer.
    """
    memory = await load_agent_memory(params.agent_id)
    
    found_memory, source_layer = find_memory_by_id(memory, params.memory_id)
    
    if not found_memory:
        raise RuntimeError(f"Memory {params.memory_id} not found for agent {params.agent_id}")
    
    if source_layer in ["long_term", "core"]:
        raise RuntimeError(f"Cannot promote from {source_layer} layer")
    
    memory.memory_layers[source_layer].remove(found_memory)
    
    found_memory.layer = params.to_layer
    if params.reason:
        found_memory.context["promotion_reason"] = params.reason
        found_memory.context["promoted_from"] = source_layer
        found_memory.context["promoted_at"] = datetime.now().isoformat()
    
    memory.memory_layers[params.to_layer].append(found_memory)
    await save_agent_memory(memory)

    return {
        "success": True,
        "memory_id": params.memory_id,
        "from_layer": source_layer,
        "to_layer": params.to_layer,
        "reason": params.reason
    }


@mcp.tool(
    name="demote_memory",
    annotations={
        "title": "Demote or Delete Memory",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def demote_memory(params: DemoteMemoryInput) -> Dict[str, Any]:
    """Demote a memory item to compost or delete it entirely.
    
    Use for outdated, incorrect, or low-value memories that should no longer 
    influence agent behavior.
    
    Args:
        params: DemoteMemoryInput with agent_id, memory_id, to_compost, and reason.
    
    Returns:
        Dict with success status, memory_id, from_layer, action 
        ('moved_to_compost' or 'deleted'), and reason.
    
    Raises:
        RuntimeError: If memory not found.
    """
    memory = await load_agent_memory(params.agent_id)
    
    found_memory, source_layer = find_memory_by_id(memory, params.memory_id)
    
    if not found_memory:
        raise RuntimeError(f"Memory {params.memory_id} not found for agent {params.agent_id}")
    
    memory.memory_layers[source_layer].remove(found_memory)
    
    action_taken = "deleted"
    if params.to_compost:
        found_memory.layer = "compost"
        if params.reason:
            found_memory.context["demotion_reason"] = params.reason
            found_memory.context["demoted_from"] = source_layer
            found_memory.context["demoted_at"] = datetime.now().isoformat()
        memory.memory_layers["compost"].append(found_memory)
        action_taken = "moved_to_compost"

    await save_agent_memory(memory)

    return {
        "success": True,
        "memory_id": params.memory_id,
        "from_layer": source_layer,
        "action": action_taken,
        "reason": params.reason
    }


@mcp.tool(
    name="cleanup_memories",
    annotations={
        "title": "Cleanup Agent Memories",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def cleanup_memories(params: CleanupMemoriesInput) -> Dict[str, Any]:
    """Run automatic memory pruning and layer transitions.
    
    Cleanup types:
    - daily: Move recent → medium_term (items >24h old with importance >0.3)
    - sprint_end: Move medium → long_term (high value) or compost (low value)
    - quarterly: Delete old compost items (>90 days with no access)
    
    Args:
        params: CleanupMemoriesInput with agent_id, cleanup_type, and dry_run.
    
    Returns:
        Dict with agent_id, cleanup_type, dry_run status, changes dict
        (promoted, demoted, deleted, kept lists), and total_changes count.
    """
    memory = await load_agent_memory(params.agent_id)
    
    changes: Dict[str, List[Dict[str, Any]]] = {
        "promoted": [],
        "demoted": [],
        "deleted": [],
        "kept": []
    }
    
    now = datetime.now()
    
    if params.cleanup_type == "daily":
        for item in list(memory.memory_layers["recent"]):
            try:
                created = datetime.fromisoformat(item.created)
                age_hours = (now - created).total_seconds() / 3600
            except ValueError:
                continue
            
            if age_hours > 24 and item.importance > 0.3:
                if not params.dry_run:
                    memory.memory_layers["recent"].remove(item)
                    item.layer = "medium_term"
                    memory.memory_layers["medium_term"].append(item)
                changes["promoted"].append({
                    "memory_id": item.id,
                    "from": "recent",
                    "to": "medium_term",
                    "age_hours": round(age_hours, 1)
                })
    
    elif params.cleanup_type == "sprint_end":
        for item in list(memory.memory_layers["medium_term"]):
            try:
                created = datetime.fromisoformat(item.created)
                age_days = (now - created).days
            except ValueError:
                continue
            
            if age_days > 14:
                if item.importance > 0.6 or item.access_count > 3:
                    if not params.dry_run:
                        memory.memory_layers["medium_term"].remove(item)
                        item.layer = "long_term"
                        memory.memory_layers["long_term"].append(item)
                    changes["promoted"].append({
                        "memory_id": item.id,
                        "from": "medium_term",
                        "to": "long_term",
                        "age_days": age_days
                    })
                else:
                    if not params.dry_run:
                        memory.memory_layers["medium_term"].remove(item)
                        item.layer = "compost"
                        memory.memory_layers["compost"].append(item)
                    changes["demoted"].append({
                        "memory_id": item.id,
                        "from": "medium_term",
                        "to": "compost",
                        "age_days": age_days
                    })
    
    elif params.cleanup_type == "quarterly":
        for item in list(memory.memory_layers["compost"]):
            try:
                created = datetime.fromisoformat(item.created)
                age_days = (now - created).days
            except ValueError:
                continue
            
            if age_days > 90 and item.access_count == 0:
                if not params.dry_run:
                    memory.memory_layers["compost"].remove(item)
                changes["deleted"].append({
                    "memory_id": item.id,
                    "from": "compost",
                    "age_days": age_days
                })
    
    if not params.dry_run:
        await save_agent_memory(memory)

    return {
        "agent_id": params.agent_id,
        "cleanup_type": params.cleanup_type,
        "dry_run": params.dry_run,
        "changes": changes,
        "total_changes": sum(len(v) for v in changes.values())
    }


@mcp.tool(
    name="get_memory_stats",
    annotations={
        "title": "Get Memory Statistics",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_memory_stats(params: GetMemoryStatsInput) -> Dict[str, Any]:
    """Get memory usage statistics for an agent.
    
    Returns counts, sizes, and other metrics per layer. Optionally includes
    health diagnostics (redundancy score, stale memory count).
    
    Args:
        params: GetMemoryStatsInput with agent_id and include_health_metrics flag.
    
    Returns:
        Dict with agent_id, role, last_updated, layers dict (count, tokens, 
        avg_importance, avg_access_count per layer), total (items, tokens, 
        size_kb_estimate), limits info, and optionally health_metrics.
    """
    memory = await load_agent_memory(params.agent_id)
    
    stats: Dict[str, Any] = {
        "agent_id": params.agent_id,
        "role": memory.role,
        "last_updated": memory.last_updated,
        "layers": {}
    }
    
    total_items = 0
    total_tokens = 0
    
    for layer_name, items in memory.memory_layers.items():
        layer_tokens = sum(item.size_estimate_tokens for item in items)
        total_items += len(items)
        total_tokens += layer_tokens
        
        stats["layers"][layer_name] = {
            "count": len(items),
            "tokens": layer_tokens,
            "avg_importance": round(
                sum(item.importance for item in items) / len(items) if items else 0, 2
            ),
            "avg_access_count": round(
                sum(item.access_count for item in items) / len(items) if items else 0, 1
            )
        }
    
    stats["total"] = {
        "items": total_items,
        "tokens": total_tokens,
        "size_kb_estimate": round(total_tokens * 4 / 1024, 2)
    }
    
    stats["limits"] = {
        "within_limits": total_tokens < MEMORY_CONFIG["token_budgets"]["task_context"],
        "max_tokens": MEMORY_CONFIG["token_budgets"]["task_context"],
        "utilization_percent": round(
            total_tokens / MEMORY_CONFIG["token_budgets"]["task_context"] * 100, 1
        )
    }
    
    # Optional health metrics (folded from removed get_memory_health tool)
    if params.include_health_metrics:
        health_metrics: Dict[str, Any] = {}
        
        # Check for content hash duplicates (redundancy)
        all_hashes = []
        duplicate_count = 0
        for layer_name, items in memory.memory_layers.items():
            if layer_name == "core":
                continue
            hashes = [item.content_hash for item in items if item.content_hash]
            duplicate_count += len(hashes) - len(set(hashes))
            all_hashes.extend(hashes)
        
        health_metrics["redundancy_score"] = round(
            duplicate_count / max(1, len(all_hashes)), 3
        )
        health_metrics["duplicate_count"] = duplicate_count
        
        # Check for stale memories (old, rarely accessed)
        now = datetime.now()
        stale_count = 0
        for layer_name, items in memory.memory_layers.items():
            if layer_name in ["long_term", "medium_term"]:
                for item in items:
                    try:
                        last_accessed = datetime.fromisoformat(item.last_accessed)
                        if (now - last_accessed).days > 60 and item.access_count < 2:
                            stale_count += 1
                    except ValueError:
                        continue
        
        health_metrics["stale_memory_count"] = stale_count
        
        # Overall health score
        health_score = 1.0
        health_score -= health_metrics["redundancy_score"] * 0.3
        health_score -= min(0.2, stale_count * 0.005)
        health_metrics["health_score"] = round(max(0, health_score), 2)
        
        stats["health_metrics"] = health_metrics
    
    return stats


@mcp.tool(
    name="query_cross_agent_memory",
    annotations={
        "title": "Query Cross-Agent Memories",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def query_cross_agent_memory(params: QueryCrossAgentMemoryInput) -> Dict[str, Any]:
    """Search memories across multiple agents (primarily for orchestrator).
    
    Searches all non-compost layers of specified agents (or all agents) for
    memories matching the query.
    
    Args:
        params: QueryCrossAgentMemoryInput with query, agent_ids, memory_types, 
                and limit.
    
    Returns:
        Dict with query, agents_queried count, results_count, and memories list
        containing agent_id, agent_role, memory_id, layer, type, content, tags,
        importance, relevance_score, and created for each match.
    """
    if params.agent_ids:
        agents_to_query = params.agent_ids
    else:
        def get_agent_files():
            return [
                f.stem.replace("-memory", "")
                for f in MEMORY_BASE_PATH.glob("*-memory.json")
            ]
        agents_to_query = await asyncio.to_thread(get_agent_files)
    
    all_results = []
    
    for agent_id in agents_to_query:
        try:
            memory = await load_agent_memory(agent_id)
            
            for layer in ["recent", "medium_term", "long_term"]:
                memories = memory.memory_layers[layer]
                
                if params.memory_types:
                    memories = [m for m in memories if m.memory_type in params.memory_types]
                
                results = search_memories_by_query(memories, params.query)
                
                for mem in results:
                    all_results.append({
                        "agent_id": agent_id,
                        "agent_role": memory.role,
                        "memory_id": mem.id,
                        "layer": mem.layer,
                        "type": mem.memory_type,
                        "content": mem.content,
                        "tags": mem.tags,
                        "importance": mem.importance,
                        "relevance_score": round(calculate_relevance_score(mem), 3),
                        "created": mem.created
                    })
        except Exception:
            continue
    
    all_results = sorted(all_results, key=lambda x: x["relevance_score"], reverse=True)
    all_results = all_results[:params.limit]
    
    return {
        "query": params.query,
        "agents_queried": len(agents_to_query),
        "results_count": len(all_results),
        "memories": all_results
    }


@mcp.tool(
    name="get_compression_candidates",
    annotations={
        "title": "Get Compression Candidates",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_compression_candidates(params: GetCompressionCandidatesInput) -> Dict[str, Any]:
    """Identify groups of old, low-access, related memories suitable for fact extraction.
    
    Groups memories by memory_type and tag overlap to find compression candidates.
    Returns full memory content for agent/LLM processing. Does NOT perform 
    compression itself - use store_extracted_facts to complete the compression.
    
    Compression workflow:
    1. Call get_compression_candidates to identify groups
    2. Agent/LLM extracts atomic facts from each group's content
    3. Call store_extracted_facts with the facts and original memory IDs
    
    Args:
        params: GetCompressionCandidatesInput with agent_id, min_age_days, 
                min_group_size, max_access_count, layers, and max_groups.
    
    Returns:
        Dict with agent_id, candidate_groups list (each containing group_id, 
        memory_type, shared_tags, memories list with full content, total_tokens, 
        and memory_count), total_groups found, and compression_potential 
        (estimated token savings).
    """
    memory = await load_agent_memory(params.agent_id)
    
    now = datetime.now()
    eligible_memories: List[MemoryItem] = []
    
    # Collect eligible memories based on age and access count
    for layer in params.layers:
        if layer not in memory.memory_layers:
            continue
        
        for item in memory.memory_layers[layer]:
            try:
                created = datetime.fromisoformat(item.created)
                age_days = (now - created).days
            except ValueError:
                continue
            
            if age_days >= params.min_age_days and item.access_count <= params.max_access_count:
                eligible_memories.append(item)

    groups = group_memories_by_similarity(eligible_memories, params.min_group_size)
    groups = groups[:params.max_groups]
    
    # Format results with full content for LLM processing
    candidate_groups = []
    total_compression_tokens = 0
    
    for idx, group in enumerate(groups):
        # Find shared tags across group
        all_tags = [set(mem.tags) for mem in group]
        shared_tags = list(set.intersection(*all_tags)) if all_tags else []
        
        group_tokens = sum(mem.size_estimate_tokens for mem in group)
        total_compression_tokens += group_tokens
        
        candidate_groups.append({
            "group_id": f"group_{idx}",
            "memory_type": group[0].memory_type,
            "shared_tags": shared_tags,
            "memories": [
                {
                    "memory_id": mem.id,
                    "layer": mem.layer,
                    "content": mem.content,
                    "tags": mem.tags,
                    "importance": mem.importance,
                    "created": mem.created,
                    "access_count": mem.access_count,
                    "size_tokens": mem.size_estimate_tokens
                }
                for mem in group
            ],
            "total_tokens": group_tokens,
            "memory_count": len(group)
        })
    
    # Estimate compression potential (facts typically achieve 80-90% reduction)
    estimated_savings = int(total_compression_tokens * 0.85)
    
    return {
        "agent_id": params.agent_id,
        "candidate_groups": candidate_groups,
        "total_groups": len(candidate_groups),
        "compression_potential": {
            "current_tokens": total_compression_tokens,
            "estimated_after_compression": total_compression_tokens - estimated_savings,
            "estimated_savings": estimated_savings,
            "estimated_reduction_percent": 85
        }
    }


@mcp.tool(
    name="store_extracted_facts",
    annotations={
        "title": "Store Extracted Facts",
        "readOnlyHint": False,
        "destructiveHint": True,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def store_extracted_facts(params: StoreExtractedFactsInput) -> Dict[str, Any]:
    """Store extracted atomic facts and remove original memories.
    
    Completes the compression workflow by storing compressed facts in long_term 
    layer and removing the original verbose memories. Tracks compression metadata
    for audit and potential restoration.
    
    Fact guidelines for calling agent:
    - Each fact should be 5-15 words (atomic, self-contained)
    - Facts should be keyword-rich for good retrieval
    - Use context_summary only when reasoning/decision context is essential
    - Aim for 80-90% token reduction vs original content
    
    Args:
        params: StoreExtractedFactsInput with agent_id, original_memory_ids,
                facts list (each with fact text, tags, importance), 
                optional context_summary, and memory_type.
    
    Returns:
        Dict with success status, compressed_memory_id, facts_stored count,
        original_memories_removed count, token_reduction (before, after, saved,
        reduction_percent), and compression_metadata.
    
    Raises:
        RuntimeError: If any original memory ID is not found.
    """
    memory = await load_agent_memory(params.agent_id)
    
    # Validate all original memories exist and collect them
    original_memories: List[tuple[MemoryItem, str]] = []
    original_tokens = 0
    all_original_tags: set = set()
    
    for mem_id in params.original_memory_ids:
        found_memory, layer = find_memory_by_id(memory, mem_id)
        if not found_memory:
            raise RuntimeError(
                f"Memory {mem_id} not found for agent {params.agent_id}. "
                f"Ensure all original_memory_ids are valid before calling store_extracted_facts."
            )
        original_memories.append((found_memory, layer))
        original_tokens += found_memory.size_estimate_tokens
        all_original_tags.update(found_memory.tags)
    
    # Build compressed content from facts
    fact_lines = []
    combined_tags = set()
    max_importance = 0.0
    
    for extracted_fact in params.facts:
        fact_lines.append(f"• {extracted_fact.fact}")
        combined_tags.update(extracted_fact.tags)
        max_importance = max(max_importance, extracted_fact.importance)

    if params.context_summary:
        compressed_content = f"[Context: {params.context_summary}]\n\n" + "\n".join(fact_lines)
    else:
        compressed_content = "\n".join(fact_lines)
    
    # Merge original tags with fact-specific tags
    combined_tags.update(all_original_tags)

    compressed_tokens = estimate_tokens(compressed_content)
    content_hash = generate_content_hash(compressed_content)
    
    compressed_memory = MemoryItem(
        id=generate_memory_id(),
        layer="long_term",
        memory_type=params.memory_type,
        content=compressed_content,
        tags=list(combined_tags)[:20],  # Enforce tag limit
        importance=max_importance,
        created=datetime.now().isoformat(),
        last_accessed=datetime.now().isoformat(),
        access_count=0,
        context={
            "compression_metadata": {
                "original_memory_ids": params.original_memory_ids,
                "original_token_count": original_tokens,
                "compressed_token_count": compressed_tokens,
                "fact_count": len(params.facts),
                "has_context_summary": params.context_summary is not None,
                "compressed_at": datetime.now().isoformat()
            }
        },
        size_estimate_tokens=compressed_tokens,
        content_hash=content_hash,
        related_to=params.original_memory_ids
    )

    removed_count = 0
    for found_memory, layer in original_memories:
        try:
            memory.memory_layers[layer].remove(found_memory)
            removed_count += 1
        except ValueError:
            # Memory already removed (shouldn't happen, but handle gracefully)
            pass

    memory.memory_layers["long_term"].append(compressed_memory)
    
    # Enforce layer limits
    max_items = MEMORY_CONFIG["limits"].get("long_term_max_items")
    if max_items and len(memory.memory_layers["long_term"]) > max_items:
        memory.memory_layers["long_term"] = memory.memory_layers["long_term"][-max_items:]
    
    await save_agent_memory(memory)
    
    tokens_saved = original_tokens - compressed_tokens
    reduction_percent = round((tokens_saved / original_tokens * 100) if original_tokens > 0 else 0, 1)
    
    return {
        "success": True,
        "compressed_memory_id": compressed_memory.id,
        "facts_stored": len(params.facts),
        "original_memories_removed": removed_count,
        "token_reduction": {
            "before": original_tokens,
            "after": compressed_tokens,
            "saved": tokens_saved,
            "reduction_percent": reduction_percent
        },
        "compression_metadata": {
            "original_memory_ids": params.original_memory_ids,
            "fact_count": len(params.facts),
            "has_context_summary": params.context_summary is not None,
            "tags_preserved": len(combined_tags)
        }
    }


# ============================================================================
# MCP Resources
# ============================================================================

@mcp.resource("memory://{agent_id}/summary")
async def agent_memory_summary(agent_id: str) -> str:
    """Provides a summary of an agent's memory state.
    
    Use this to quickly check memory usage and status without loading full context.
    """
    try:
        stats = await get_memory_stats(GetMemoryStatsInput(
            agent_id=agent_id, 
            include_health_metrics=True
        ))
        
        summary = f"""# Memory Summary: {agent_id}

**Role:** {stats['role']}
**Last Updated:** {stats['last_updated']}

## Layer Statistics
"""
        for layer_name, layer_stats in stats['layers'].items():
            summary += f"""
### {layer_name.replace('_', ' ').title()}
- Items: {layer_stats['count']}
- Tokens: {layer_stats['tokens']}
- Avg Importance: {layer_stats['avg_importance']}
- Avg Access Count: {layer_stats['avg_access_count']}
"""
        
        summary += f"""
## Total Usage
- Total Items: {stats['total']['items']}
- Total Tokens: {stats['total']['tokens']}
- Size Estimate: {stats['total']['size_kb_estimate']} KB
- Within Limits: {stats['limits']['within_limits']}
- Utilization: {stats['limits']['utilization_percent']}%
"""
        
        if 'health_metrics' in stats:
            hm = stats['health_metrics']
            summary += f"""
## Health Metrics
- Health Score: {hm['health_score']}
- Redundancy Score: {hm['redundancy_score']}
- Duplicate Count: {hm['duplicate_count']}
- Stale Memories: {hm['stale_memory_count']}
"""
        
        return summary
    
    except Exception as e:
        return f"Error generating summary for {agent_id}: {e}"


@mcp.resource("memory://config")
def memory_config() -> str:
    """Provides the current memory management configuration.
    
    Shows limits, token budgets, and cleanup schedules.
    """
    return f"""# Memory Management Configuration

## Memory Limits

- Recent Layer: Max {MEMORY_CONFIG['limits']['recent_max_items']} items
- Medium-Term Layer: Max {MEMORY_CONFIG['limits']['medium_term_max_items']} items
- Long-Term Layer: Max {MEMORY_CONFIG['limits']['long_term_max_items']} items
- Total Size Limit: {MEMORY_CONFIG['limits']['total_max_size_kb']} KB per agent

## Token Budgets

- Core Memory: {MEMORY_CONFIG['token_budgets']['core_memory']} tokens
- Task Context: {MEMORY_CONFIG['token_budgets']['task_context']} tokens

## Compression Settings

- Minimum Age for Compression: {MEMORY_CONFIG['compression']['min_age_days']} days
- Minimum Group Size: {MEMORY_CONFIG['compression']['min_group_size']} memories
- Maximum Access Count: {MEMORY_CONFIG['compression']['max_access_count']}
- Fact Length: {MEMORY_CONFIG['compression']['fact_min_words']}-{MEMORY_CONFIG['compression']['fact_max_words']} words

## Cleanup Schedule

- **Daily (00:00):** Recent → Medium-Term (items > 24h old with importance > 0.3)
- **Sprint End (Manual):** Medium-Term → Long-Term (importance > 0.6 or access_count > 3) or Compost
- **Quarterly:** Compost cleanup (delete items > 90 days with no access)

## Memory Layers

1. **Core:** Role definition, permanent agent identity
2. **Recent:** Last 24 hours, auto-promoted to medium-term
3. **Medium-Term:** Current sprint context, 2-week lifecycle
4. **Long-Term:** Valuable memories retained across sprints (including compressed facts)
5. **Compost:** Demoted memories, cleared quarterly

## Fact Extraction Compression

The compression system uses atomic fact extraction for efficient long-term storage:

1. **get_compression_candidates:** Identifies groups of old, related memories
2. **store_extracted_facts:** Stores atomic facts, removes originals

Benefits:
- 80-90% token reduction (vs 50-60% with prose summaries)
- Better keyword matching for retrieval
- No degradation through re-compression
- Optional context preservation for decisions
"""


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run(
        transport="sse",
        host="0.0.0.0",
        port=8080
    )
