"""
Memory Management MCP Server for Multi-Agent Development Framework

This MCP server provides centralized memory management for agent collaboration,
handling storage, retrieval, pruning, and cross-agent queries for the development team.
"""

from fastmcp import FastMCP, Context
from fastmcp.exceptions import ToolError
from typing import List, Dict, Optional, Any, Literal, Tuple
from pydantic import Field, BaseModel
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
import asyncio
import threading
from collections import defaultdict

# Initialize FastMCP server
mcp = FastMCP(
    name="Agent Memory Manager",
    dependencies=[]
)

# Configuration
MEMORY_BASE_PATH = Path(".claude/agents/memories")
MEMORY_BASE_PATH.mkdir(parents=True, exist_ok=True)

# File lock for thread safety
file_locks = defaultdict(threading.Lock)

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
    }
}


# ============================================================================
# Data Models
# ============================================================================

class MemoryItem(BaseModel):
    """Schema for a memory item"""
    id: str
    layer: Literal["core", "recent", "medium_term", "long_term", "compost"]
    memory_type: str
    content: str
    tags: List[str] = Field(default_factory=list)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    created: str
    last_accessed: str
    access_count: int = 0
    context: Dict[str, Any] = Field(default_factory=dict)
    size_estimate_tokens: int = 0
    # New fields for better management
    version: int = 1
    previous_version_id: Optional[str] = None
    related_to: List[str] = Field(default_factory=list)  # Simple ID references
    content_hash: Optional[str] = None  # For deduplication


class AgentMemory(BaseModel):
    """Schema for an agent's complete memory structure"""
    agent_id: str
    role: str
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
# Helper Functions
# ============================================================================

def estimate_tokens(text: str) -> int:
    """More accurate token estimation based on content type"""
    # Code typically has more tokens due to symbols
    if '{' in text or 'def ' in text or 'class ' in text:
        return int(len(text) / 4)
    # Regular text is slightly more efficient
    return int(len(text) / 3.5)


def generate_memory_id() -> str:
    """Generate a unique memory ID"""
    timestamp = datetime.now().isoformat()
    random_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"mem_{random_hash}"


def generate_content_hash(content: str) -> str:
    """Generate hash of content for deduplication"""
    return hashlib.md5(content.encode()).hexdigest()


def get_memory_file_path(agent_id: str) -> Path:
    """Get the file path for an agent's memory"""
    return MEMORY_BASE_PATH / f"{agent_id}-memory.json"


def load_agent_memory(agent_id: str) -> AgentMemory:
    """Load an agent's memory from disk with thread safety"""
    memory_file = get_memory_file_path(agent_id)
    
    with file_locks[agent_id]:
        if not memory_file.exists():
            # Initialize new memory structure
            memory = AgentMemory(
                agent_id=agent_id,
                role="unknown",
                last_updated=datetime.now().isoformat()
            )
            save_agent_memory(memory)
            return memory
        
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
                return AgentMemory(**data)
        except Exception as e:
            raise ToolError(f"Failed to load memory for {agent_id}: {str(e)}")


def save_agent_memory(memory: AgentMemory) -> None:
    """Save an agent's memory to disk with thread safety"""
    memory.last_updated = datetime.now().isoformat()
    memory_file = get_memory_file_path(memory.agent_id)
    
    with file_locks[memory.agent_id]:
        try:
            # Write to temp file first, then rename for atomicity
            temp_file = memory_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(memory.model_dump(), f, indent=2)
            temp_file.replace(memory_file)
        except Exception as e:
            raise ToolError(f"Failed to save memory for {memory.agent_id}: {str(e)}")


def filter_memories_by_tags(
    memories: List[MemoryItem], 
    tags: List[str]
) -> List[MemoryItem]:
    """Filter memories that match any of the provided tags"""
    if not tags:
        return memories
    
    return [
        mem for mem in memories
        if any(tag in mem.tags for tag in tags)
    ]


def search_memories_by_query(
    memories: List[MemoryItem],
    query: str
) -> List[MemoryItem]:
    """Simple keyword search in memory content"""
    query_lower = query.lower()
    results = []
    
    for mem in memories:
        # Search in content and tags
        if (query_lower in mem.content.lower() or
            any(query_lower in tag.lower() for tag in mem.tags)):
            results.append(mem)
    
    return results


def calculate_relevance_score(memory: MemoryItem) -> float:
    """Calculate relevance score for memory ranking"""
    # Base score is importance
    score = memory.importance
    
    # Boost for recent access
    last_accessed = datetime.fromisoformat(memory.last_accessed)
    days_since_access = (datetime.now() - last_accessed).days
    recency_boost = max(0, 0.2 * (30 - days_since_access) / 30)
    
    # Boost for access frequency
    access_boost = min(0.1 * memory.access_count, 0.3)
    
    return min(1.0, score + recency_boost + access_boost)


def calculate_similarity(content1: str, content2: str) -> float:
    """Simple similarity calculation for deduplication"""
    # Basic Jaccard similarity on words
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def summarize_content(contents: List[str], max_tokens: int = 500) -> str:
    """Create a summary of multiple content pieces"""
    # Simple summarization: take key sentences up to token limit
    all_sentences = []
    for content in contents:
        sentences = content.split('. ')
        all_sentences.extend(sentences)
    
    # Sort by length (prefer more informative sentences)
    all_sentences.sort(key=len, reverse=True)
    
    summary = []
    tokens_used = 0
    
    for sentence in all_sentences:
        sentence_tokens = estimate_tokens(sentence)
        if tokens_used + sentence_tokens <= max_tokens:
            summary.append(sentence)
            tokens_used += sentence_tokens
    
    return '. '.join(summary)


# ============================================================================
# MCP Tools - Memory CRUD Operations
# ============================================================================

@mcp.tool
def store_memory(
    agent_id: str = Field(description="ID of the agent storing the memory"),
    layer: Literal["recent", "medium_term", "long_term", "compost", "core"] = Field(
        description="Memory persistence layer"
    ),
    memory_type: str = Field(description="Type of memory (decision, implementation, bug_fix, etc.)"),
    content: str = Field(description="The actual memory content to store"),
    tags: List[str] = Field(default_factory=list, description="Tags for categorization"),
    importance: float = Field(default=0.5, description="Importance score (0.0 to 1.0)"),
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context (task_id, related_files, related_agents)"
    ),
    related_to: List[str] = Field(default_factory=list, description="Related memory IDs")
) -> Dict[str, Any]:
    """
    Store a new memory item for an agent.
    
    Returns the memory ID and confirmation of storage.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    # Generate content hash for deduplication
    content_hash = generate_content_hash(content)
    
    # Check for duplicates in the same layer
    for item in memory.memory_layers[layer]:
        if item.content_hash == content_hash:
            # Update existing instead of creating duplicate
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            save_agent_memory(memory)
            return {
                "success": True,
                "memory_id": item.id,
                "agent_id": agent_id,
                "layer": layer,
                "action": "updated_existing",
                "size_tokens": item.size_estimate_tokens
            }
    
    # Create new memory item
    memory_item = MemoryItem(
        id=generate_memory_id(),
        layer=layer,
        memory_type=memory_type,
        content=content,
        tags=tags,
        importance=importance,
        created=datetime.now().isoformat(),
        last_accessed=datetime.now().isoformat(),
        access_count=0,
        context=context,
        size_estimate_tokens=estimate_tokens(content),
        content_hash=content_hash,
        related_to=related_to
    )
    
    # Add to appropriate layer
    memory.memory_layers[layer].append(memory_item)
    
    # Check if we need to prune
    max_items = MEMORY_CONFIG["limits"].get(f"{layer}_max_items")
    if max_items and len(memory.memory_layers[layer]) > max_items:
        # Keep most recent items
        memory.memory_layers[layer] = memory.memory_layers[layer][-max_items:]
    
    # Save memory
    save_agent_memory(memory)
    
    return {
        "success": True,
        "memory_id": memory_item.id,
        "agent_id": agent_id,
        "layer": layer,
        "action": "created_new",
        "size_tokens": memory_item.size_estimate_tokens
    }


@mcp.tool
def load_agent_context(
    agent_id: str = Field(description="ID of the agent"),
    task_description: Optional[str] = Field(None, description="Description of current task"),
    task_tags: List[str] = Field(default_factory=list, description="Tags related to current task"),
    max_tokens: int = Field(default=5000, description="Maximum tokens to return"),
    include_layers: List[str] = Field(
        default=["core", "recent", "medium_term"],
        description="Which memory layers to include"
    )
) -> Dict[str, Any]:
    """
    Load relevant memory context for an agent's current task.
    
    Returns optimized context within token budget.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    context_parts = []
    tokens_used = 0
    items_loaded = []
    
    # 1. Always include core memory (if requested)
    if "core" in include_layers:
        for item in memory.memory_layers["core"]:
            if tokens_used + item.size_estimate_tokens <= max_tokens:
                context_parts.append({
                    "layer": "core",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
    
    # 2. Load task-relevant recent memories
    if "recent" in include_layers:
        recent_memories = memory.memory_layers["recent"]
        if task_tags:
            recent_memories = filter_memories_by_tags(recent_memories, task_tags)
        
        # Sort by relevance
        recent_memories.sort(
            key=lambda x: calculate_relevance_score(x),
            reverse=True
        )
        
        for item in recent_memories:
            if tokens_used + item.size_estimate_tokens <= max_tokens - 500:
                context_parts.append({
                    "layer": "recent",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                
                # Update access info
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    # 3. Load relevant medium-term memories if space available
    if "medium_term" in include_layers and tokens_used < max_tokens - 1000:
        medium_memories = memory.memory_layers["medium_term"]
        
        # Filter by task relevance
        if task_description:
            medium_memories = search_memories_by_query(medium_memories, task_description)
        if task_tags:
            medium_memories = filter_memories_by_tags(medium_memories, task_tags)
        
        # Sort by relevance
        medium_memories.sort(
            key=lambda x: calculate_relevance_score(x),
            reverse=True
        )
        
        for item in medium_memories[:5]:  # Top 5 relevant items
            if tokens_used + item.size_estimate_tokens <= max_tokens - 500:
                context_parts.append({
                    "layer": "medium_term",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                
                # Update access info
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    # 4. Load long-term memories if space available
    if "long_term" in include_layers and tokens_used < max_tokens - 500:
        long_term_memories = memory.memory_layers["long_term"]
        
        # Filter by task relevance
        if task_description:
            long_term_memories = search_memories_by_query(long_term_memories, task_description)
        if task_tags:
            long_term_memories = filter_memories_by_tags(long_term_memories, task_tags)
        
        # Sort by importance and relevance
        long_term_memories.sort(
            key=lambda x: (x.importance, calculate_relevance_score(x)),
            reverse=True
        )
        
        for item in long_term_memories[:3]:  # Top 3 most important
            if tokens_used + item.size_estimate_tokens <= max_tokens - 500:
                context_parts.append({
                    "layer": "long_term",
                    "type": item.memory_type,
                    "content": item.content,
                    "tags": item.tags,
                    "importance": item.importance
                })
                tokens_used += item.size_estimate_tokens
                items_loaded.append(item.id)
                
                # Update access info
                item.access_count += 1
                item.last_accessed = datetime.now().isoformat()
    
    # Save updated access counts
    save_agent_memory(memory)
    
    return {
        "agent_id": agent_id,
        "context": context_parts,
        "tokens_used": tokens_used,
        "items_loaded": len(items_loaded),
        "memory_ids": items_loaded
    }


@mcp.tool
def search_memories(
    agent_id: str = Field(description="ID of the agent"),
    query: str = Field(description="Search query string"),
    layers: Optional[List[str]] = Field(None, description="Layers to search in"),
    tags: Optional[List[str]] = Field(None, description="Filter by tags"),
    limit: int = Field(default=10, description="Maximum results to return"),
    min_importance: float = Field(default=0.0, description="Minimum importance score")
) -> Dict[str, Any]:
    """
    Search across agent memories using keywords and filters.
    
    Returns matching memories ranked by relevance.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    # Determine which layers to search
    search_layers = layers if layers else ["recent", "medium_term", "long_term"]
    
    # Collect all memories from searched layers
    all_memories = []
    for layer in search_layers:
        if layer in memory.memory_layers:
            all_memories.extend(memory.memory_layers[layer])
    
    # Apply search query
    results = search_memories_by_query(all_memories, query)
    
    # Apply tag filter
    if tags:
        results = filter_memories_by_tags(results, tags)
    
    # Apply importance filter
    results = [m for m in results if m.importance >= min_importance]
    
    # Sort by relevance score
    results.sort(key=lambda x: calculate_relevance_score(x), reverse=True)
    
    # Limit results
    results = results[:limit]
    
    # Update access counts
    for mem in results:
        mem.access_count += 1
        mem.last_accessed = datetime.now().isoformat()
    save_agent_memory(memory)
    
    # Format results
    formatted_results = [
        {
            "memory_id": mem.id,
            "layer": mem.layer,
            "type": mem.memory_type,
            "content": mem.content,
            "tags": mem.tags,
            "importance": mem.importance,
            "relevance_score": calculate_relevance_score(mem),
            "created": mem.created,
            "context": mem.context
        }
        for mem in results
    ]
    
    return {
        "agent_id": agent_id,
        "query": query,
        "results_count": len(formatted_results),
        "memories": formatted_results
    }


@mcp.tool
def promote_memory(
    agent_id: str = Field(description="ID of the agent"),
    memory_id: str = Field(description="ID of memory to promote"),
    to_layer: Literal["medium_term", "long_term"] = Field(
        description="Target layer for promotion"
    ),
    reason: Optional[str] = Field(None, description="Reason for promotion")
) -> Dict[str, Any]:
    """
    Promote a memory item to a higher persistence layer.
    
    Use when a memory proves valuable and should be retained longer.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    # Find the memory item
    found_memory = None
    source_layer = None
    
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id == memory_id:
                found_memory = item
                source_layer = layer_name
                break
        if found_memory:
            break
    
    if not found_memory:
        raise ToolError(f"Memory {memory_id} not found for agent {agent_id}")
    
    # Can't promote from long_term or core
    if source_layer in ["long_term", "core"]:
        raise ToolError(f"Cannot promote from {source_layer} layer")
    
    # Remove from source layer
    memory.memory_layers[source_layer].remove(found_memory)
    
    # Update layer and add to target
    found_memory.layer = to_layer
    if reason:
        found_memory.context["promotion_reason"] = reason
        found_memory.context["promoted_from"] = source_layer
        found_memory.context["promoted_at"] = datetime.now().isoformat()
    
    memory.memory_layers[to_layer].append(found_memory)
    
    # Save memory
    save_agent_memory(memory)
    
    return {
        "success": True,
        "memory_id": memory_id,
        "from_layer": source_layer,
        "to_layer": to_layer,
        "reason": reason
    }


@mcp.tool
def demote_memory(
    agent_id: str = Field(description="ID of the agent"),
    memory_id: str = Field(description="ID of memory to demote"),
    to_compost: bool = Field(default=True, description="Move to compost or delete"),
    reason: Optional[str] = Field(None, description="Reason for demotion")
) -> Dict[str, Any]:
    """
    Demote a memory item to compost or delete it entirely.
    
    Use for outdated or low-value memories.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    # Find the memory item
    found_memory = None
    source_layer = None
    
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id == memory_id:
                found_memory = item
                source_layer = layer_name
                break
        if found_memory:
            break
    
    if not found_memory:
        raise ToolError(f"Memory {memory_id} not found for agent {agent_id}")
    
    # Remove from source layer
    memory.memory_layers[source_layer].remove(found_memory)
    
    action_taken = "deleted"
    if to_compost:
        # Move to compost
        found_memory.layer = "compost"
        if reason:
            found_memory.context["demotion_reason"] = reason
            found_memory.context["demoted_from"] = source_layer
            found_memory.context["demoted_at"] = datetime.now().isoformat()
        memory.memory_layers["compost"].append(found_memory)
        action_taken = "moved_to_compost"
    
    # Save memory
    save_agent_memory(memory)
    
    return {
        "success": True,
        "memory_id": memory_id,
        "from_layer": source_layer,
        "action": action_taken,
        "reason": reason
    }


@mcp.tool
def cleanup_memories(
    agent_id: str = Field(description="ID of the agent"),
    cleanup_type: Literal["daily", "sprint_end", "quarterly"] = Field(
        default="daily",
        description="Type of cleanup to perform"
    ),
    dry_run: bool = Field(default=False, description="Preview changes without applying")
) -> Dict[str, Any]:
    """
    Run automatic memory pruning and layer transitions.
    
    Cleanup types:
    - daily: Move recent → medium_term (>24h old)
    - sprint_end: Move medium → long_term or compost (>14 days)
    - quarterly: Clean compost, deduplicate long_term
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    changes = {
        "promoted": [],
        "demoted": [],
        "deleted": [],
        "kept": []
    }
    
    now = datetime.now()
    
    if cleanup_type == "daily":
        # Move recent → medium_term (>24h old)
        for item in list(memory.memory_layers["recent"]):
            created = datetime.fromisoformat(item.created)
            age_hours = (now - created).total_seconds() / 3600
            
            if age_hours > 24 and item.importance > 0.3:
                if not dry_run:
                    memory.memory_layers["recent"].remove(item)
                    item.layer = "medium_term"
                    memory.memory_layers["medium_term"].append(item)
                changes["promoted"].append({
                    "memory_id": item.id,
                    "from": "recent",
                    "to": "medium_term",
                    "age_hours": age_hours
                })
    
    elif cleanup_type == "sprint_end":
        # Move medium_term → long_term or compost (>14 days)
        for item in list(memory.memory_layers["medium_term"]):
            created = datetime.fromisoformat(item.created)
            age_days = (now - created).days
            
            if age_days > 14:
                if item.importance > 0.6 or item.access_count > 3:
                    # Promote to long_term
                    if not dry_run:
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
                    # Move to compost
                    if not dry_run:
                        memory.memory_layers["medium_term"].remove(item)
                        item.layer = "compost"
                        memory.memory_layers["compost"].append(item)
                    changes["demoted"].append({
                        "memory_id": item.id,
                        "from": "medium_term",
                        "to": "compost",
                        "age_days": age_days
                    })
    
    elif cleanup_type == "quarterly":
        # Clean old compost items (>90 days with no access)
        for item in list(memory.memory_layers["compost"]):
            created = datetime.fromisoformat(item.created)
            age_days = (now - created).days
            
            if age_days > 90 and item.access_count == 0:
                if not dry_run:
                    memory.memory_layers["compost"].remove(item)
                changes["deleted"].append({
                    "memory_id": item.id,
                    "from": "compost",
                    "age_days": age_days
                })
    
    # Save if not dry run
    if not dry_run:
        save_agent_memory(memory)
    
    return {
        "agent_id": agent_id,
        "cleanup_type": cleanup_type,
        "dry_run": dry_run,
        "changes": changes,
        "total_changes": sum(len(v) for v in changes.values())
    }


@mcp.tool
def get_memory_stats(
    agent_id: str = Field(description="ID of the agent")
) -> Dict[str, Any]:
    """
    Get memory usage statistics for an agent.
    
    Returns counts, sizes, and other metrics per layer.
    """
    # Load agent memory
    memory = load_agent_memory(agent_id)
    
    stats = {
        "agent_id": agent_id,
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
            "avg_importance": (
                sum(item.importance for item in items) / len(items)
                if items else 0
            ),
            "avg_access_count": (
                sum(item.access_count for item in items) / len(items)
                if items else 0
            )
        }
    
    stats["total"] = {
        "items": total_items,
        "tokens": total_tokens,
        "size_kb_estimate": total_tokens * 4 / 1024  # Rough estimate
    }
    
    # Check limits
    stats["limits"] = {
        "within_limits": total_tokens < MEMORY_CONFIG["token_budgets"]["task_context"],
        "max_tokens": MEMORY_CONFIG["token_budgets"]["task_context"],
        "utilization_percent": (
            total_tokens / MEMORY_CONFIG["token_budgets"]["task_context"] * 100
        )
    }
    
    return stats


@mcp.tool
def query_cross_agent_memory(
    query: str = Field(description="Search query string"),
    agent_ids: Optional[List[str]] = Field(None, description="Specific agents to query"),
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types"),
    limit: int = Field(default=20, description="Maximum results to return")
) -> Dict[str, Any]:
    """
    Search memories across multiple agents (primarily for orchestrator).
    
    Returns relevant memories from specified agents or all agents.
    """
    # Determine which agents to query
    if agent_ids:
        agents_to_query = agent_ids
    else:
        # Get all agent memory files
        agents_to_query = [
            f.stem.replace("-memory", "")
            for f in MEMORY_BASE_PATH.glob("*-memory.json")
        ]
    
    all_results = []
    
    for agent_id in agents_to_query:
        try:
            memory = load_agent_memory(agent_id)
            
            # Search across all non-compost layers
            for layer in ["recent", "medium_term", "long_term"]:
                memories = memory.memory_layers[layer]
                
                # Apply memory type filter
                if memory_types:
                    memories = [m for m in memories if m.memory_type in memory_types]
                
                # Search
                results = search_memories_by_query(memories, query)
                
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
                        "relevance_score": calculate_relevance_score(mem),
                        "created": mem.created
                    })
        except Exception as e:
            # Skip agents with errors
            continue
    
    # Sort by relevance
    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    # Limit results
    all_results = all_results[:limit]
    
    return {
        "query": query,
        "agents_queried": len(agents_to_query),
        "results_count": len(all_results),
        "memories": all_results
    }


# ============================================================================
# NEW CRITICAL TOOLS
# ============================================================================

@mcp.tool
def deduplicate_memories(
    agent_id: str = Field(description="ID of the agent"),
    similarity_threshold: float = Field(default=0.85, description="Similarity threshold for deduplication"),
    dry_run: bool = Field(default=False, description="Preview changes without applying")
) -> Dict[str, Any]:
    """
    Find and merge duplicate or highly similar memories.
    
    Consolidates duplicate memories within each layer, keeping the most accessed version.
    """
    memory = load_agent_memory(agent_id)
    duplicates_found = []
    
    for layer_name, items in memory.memory_layers.items():
        if layer_name == "core":  # Don't deduplicate core memories
            continue
            
        # Group by content hash for exact duplicates
        hash_groups = defaultdict(list)
        for item in items:
            if item.content_hash:
                hash_groups[item.content_hash].append(item)
        
        # Find exact duplicates
        for hash_val, group in hash_groups.items():
            if len(group) > 1:
                # Sort by access count and keep the most accessed
                group.sort(key=lambda x: (x.access_count, x.importance), reverse=True)
                keeper = group[0]
                
                for duplicate in group[1:]:
                    # Merge metadata
                    keeper.access_count += duplicate.access_count
                    keeper.tags = list(set(keeper.tags + duplicate.tags))
                    keeper.related_to = list(set(keeper.related_to + duplicate.related_to))
                    
                    duplicates_found.append({
                        "layer": layer_name,
                        "kept_id": keeper.id,
                        "removed_id": duplicate.id,
                        "type": "exact_match"
                    })
                    
                    if not dry_run:
                        items.remove(duplicate)
        
        # Find similar content (fuzzy matching)
        remaining_items = list(items)
        for i, item1 in enumerate(remaining_items):
            for item2 in remaining_items[i+1:]:
                similarity = calculate_similarity(item1.content, item2.content)
                
                if similarity >= similarity_threshold:
                    # Keep the more important/accessed one
                    if (item1.importance > item2.importance or 
                        item1.access_count > item2.access_count):
                        keeper, duplicate = item1, item2
                    else:
                        keeper, duplicate = item2, item1
                    
                    # Merge metadata
                    keeper.access_count += duplicate.access_count
                    keeper.tags = list(set(keeper.tags + duplicate.tags))
                    keeper.related_to = list(set(keeper.related_to + duplicate.related_to))
                    
                    duplicates_found.append({
                        "layer": layer_name,
                        "kept_id": keeper.id,
                        "removed_id": duplicate.id,
                        "type": "similar_content",
                        "similarity": similarity
                    })
                    
                    if not dry_run:
                        if duplicate in items:
                            items.remove(duplicate)
    
    if not dry_run:
        save_agent_memory(memory)
    
    return {
        "agent_id": agent_id,
        "duplicates_found": len(duplicates_found),
        "duplicates": duplicates_found,
        "dry_run": dry_run
    }


@mcp.tool
def compress_old_memories(
    agent_id: str = Field(description="ID of the agent"),
    older_than_days: int = Field(default=30, description="Compress memories older than this"),
    target_reduction: float = Field(default=0.5, description="Target size reduction (0.5 = 50% smaller)"),
    dry_run: bool = Field(default=False, description="Preview changes without applying")
) -> Dict[str, Any]:
    """
    Summarize old memories to reduce token usage while preserving key information.
    
    Groups related old memories and creates compressed summaries.
    """
    memory = load_agent_memory(agent_id)
    now = datetime.now()
    compressed = []
    
    for layer_name in ["medium_term", "long_term"]:
        if layer_name not in memory.memory_layers:
            continue
            
        items = memory.memory_layers[layer_name]
        old_items = []
        
        # Find old items
        for item in items:
            created = datetime.fromisoformat(item.created)
            age_days = (now - created).days
            
            if age_days > older_than_days and item.access_count < 2:
                old_items.append(item)
        
        if not old_items:
            continue
        
        # Group by memory type and tags
        groups = defaultdict(list)
        for item in old_items:
            group_key = (item.memory_type, tuple(sorted(item.tags)))
            groups[group_key].append(item)
        
        # Compress each group
        for (memory_type, tags), group_items in groups.items():
            if len(group_items) < 2:  # Don't compress single items
                continue
            
            # Calculate target token count
            total_tokens = sum(item.size_estimate_tokens for item in group_items)
            target_tokens = int(total_tokens * target_reduction)
            
            # Create summary
            contents = [item.content for item in group_items]
            summary_content = summarize_content(contents, target_tokens)
            
            if not dry_run:
                # Create new compressed memory
                compressed_memory = MemoryItem(
                    id=generate_memory_id(),
                    layer=layer_name,
                    memory_type=f"compressed_{memory_type}",
                    content=summary_content,
                    tags=list(tags) + ["compressed"],
                    importance=max(item.importance for item in group_items),
                    created=min(item.created for item in group_items),
                    last_accessed=datetime.now().isoformat(),
                    access_count=sum(item.access_count for item in group_items),
                    context={
                        "compression_date": datetime.now().isoformat(),
                        "original_count": len(group_items),
                        "original_ids": [item.id for item in group_items]
                    },
                    size_estimate_tokens=estimate_tokens(summary_content),
                    content_hash=generate_content_hash(summary_content)
                )
                
                # Remove old items and add compressed
                for item in group_items:
                    items.remove(item)
                items.append(compressed_memory)
            
            compressed.append({
                "layer": layer_name,
                "original_count": len(group_items),
                "original_tokens": total_tokens,
                "compressed_tokens": estimate_tokens(summary_content),
                "reduction_percent": (1 - estimate_tokens(summary_content) / total_tokens) * 100
            })
    
    if not dry_run:
        save_agent_memory(memory)
    
    return {
        "agent_id": agent_id,
        "groups_compressed": len(compressed),
        "compressions": compressed,
        "dry_run": dry_run
    }


@mcp.tool
def bulk_update_memories(
    agent_id: str = Field(description="ID of the agent"),
    memory_ids: List[str] = Field(description="List of memory IDs to update"),
    updates: Dict[str, Any] = Field(
        description="Updates to apply (tags, importance, etc.)"
    )
) -> Dict[str, Any]:
    """
    Update multiple memories at once.
    
    Efficiently updates tags, importance, or other metadata for multiple memories.
    """
    memory = load_agent_memory(agent_id)
    updated_count = 0
    not_found = []
    
    # Find and update memories
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id in memory_ids:
                # Apply updates
                if "tags" in updates:
                    item.tags = updates["tags"]
                if "importance" in updates:
                    item.importance = min(1.0, max(0.0, updates["importance"]))
                if "add_tags" in updates:
                    item.tags = list(set(item.tags + updates["add_tags"]))
                if "remove_tags" in updates:
                    item.tags = [t for t in item.tags if t not in updates["remove_tags"]]
                
                item.last_accessed = datetime.now().isoformat()
                updated_count += 1
    
    # Find which IDs weren't found
    found_ids = set()
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id in memory_ids:
                found_ids.add(item.id)
    
    not_found = [mid for mid in memory_ids if mid not in found_ids]
    
    # Save
    save_agent_memory(memory)
    
    return {
        "agent_id": agent_id,
        "requested": len(memory_ids),
        "updated": updated_count,
        "not_found": not_found,
        "updates_applied": updates
    }


@mcp.tool
def get_memory_health(
    agent_id: str = Field(description="ID of the agent")
) -> Dict[str, Any]:
    """
    Check for memory issues like fragmentation, orphaned references, or corruption.
    
    Returns health metrics and recommendations for optimization.
    """
    memory = load_agent_memory(agent_id)
    
    health_report = {
        "agent_id": agent_id,
        "issues": [],
        "metrics": {},
        "recommendations": []
    }
    
    # Check for duplicates
    all_hashes = []
    duplicate_count = 0
    for layer_name, items in memory.memory_layers.items():
        hashes = [item.content_hash for item in items if item.content_hash]
        duplicate_count += len(hashes) - len(set(hashes))
        all_hashes.extend(hashes)
    
    redundancy_score = duplicate_count / max(1, len(all_hashes))
    health_report["metrics"]["redundancy_score"] = round(redundancy_score, 3)
    
    if redundancy_score > 0.1:
        health_report["issues"].append(f"High redundancy: {duplicate_count} duplicates found")
        health_report["recommendations"].append("Run deduplicate_memories to clean up")
    
    # Check for orphaned references
    all_memory_ids = set()
    for layer_name, items in memory.memory_layers.items():
        all_memory_ids.update(item.id for item in items)
    
    orphaned_refs = []
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            for ref_id in item.related_to:
                if ref_id not in all_memory_ids:
                    orphaned_refs.append({
                        "memory_id": item.id,
                        "orphaned_ref": ref_id
                    })
    
    health_report["metrics"]["orphaned_references"] = len(orphaned_refs)
    if orphaned_refs:
        health_report["issues"].append(f"Found {len(orphaned_refs)} orphaned references")
        health_report["recommendations"].append("Clean up related_to fields")
    
    # Check fragmentation (memories scattered across time)
    for layer_name in ["medium_term", "long_term"]:
        items = memory.memory_layers.get(layer_name, [])
        if items:
            dates = [datetime.fromisoformat(item.created) for item in items]
            if len(dates) > 1:
                time_span = (max(dates) - min(dates)).days
                if time_span > 0:
                    fragmentation = len(set(d.date() for d in dates)) / time_span
                    health_report["metrics"][f"{layer_name}_fragmentation"] = round(fragmentation, 3)
    
    # Check for old unaccessed memories
    now = datetime.now()
    stale_memories = 0
    for layer_name, items in memory.memory_layers.items():
        if layer_name in ["long_term", "medium_term"]:
            for item in items:
                last_accessed = datetime.fromisoformat(item.last_accessed)
                if (now - last_accessed).days > 60 and item.access_count < 2:
                    stale_memories += 1
    
    health_report["metrics"]["stale_memories"] = stale_memories
    if stale_memories > 20:
        health_report["issues"].append(f"Found {stale_memories} stale memories")
        health_report["recommendations"].append("Consider compressing old memories")
    
    # Check layer balance
    layer_counts = {layer: len(items) for layer, items in memory.memory_layers.items()}
    health_report["metrics"]["layer_distribution"] = layer_counts
    
    # Overall health score
    health_score = 1.0
    health_score -= redundancy_score * 0.3
    health_score -= min(0.2, len(orphaned_refs) * 0.02)
    health_score -= min(0.2, stale_memories * 0.005)
    
    health_report["health_score"] = round(max(0, health_score), 2)
    
    return health_report


@mcp.tool
def load_memories_batch(
    memory_requests: List[Dict[str, str]] = Field(
        description="List of {agent_id: str, memory_id: str} dicts"
    )
) -> Dict[str, Any]:
    """
    Efficiently load multiple specific memories from different agents.
    
    Optimized for loading specific memories rather than searching.
    """
    results = []
    not_found = []
    
    # Group by agent for efficiency
    agent_groups = defaultdict(list)
    for request in memory_requests:
        agent_groups[request["agent_id"]].append(request["memory_id"])
    
    # Load memories by agent
    for agent_id, memory_ids in agent_groups.items():
        try:
            memory = load_agent_memory(agent_id)
            
            # Find requested memories
            for layer_name, items in memory.memory_layers.items():
                for item in items:
                    if item.id in memory_ids:
                        results.append({
                            "agent_id": agent_id,
                            "memory_id": item.id,
                            "layer": item.layer,
                            "type": item.memory_type,
                            "content": item.content,
                            "tags": item.tags,
                            "importance": item.importance,
                            "created": item.created,
                            "context": item.context
                        })
                        
                        # Update access
                        item.access_count += 1
                        item.last_accessed = datetime.now().isoformat()
            
            # Save updated access counts
            save_agent_memory(memory)
            
            # Track not found
            found_ids = [r["memory_id"] for r in results if r["agent_id"] == agent_id]
            for mid in memory_ids:
                if mid not in found_ids:
                    not_found.append({"agent_id": agent_id, "memory_id": mid})
                    
        except Exception as e:
            # Skip agents with errors
            for mid in memory_ids:
                not_found.append({"agent_id": agent_id, "memory_id": mid, "error": str(e)})
    
    return {
        "requested": len(memory_requests),
        "found": len(results),
        "not_found": len(not_found),
        "memories": results,
        "missing": not_found
    }


# ============================================================================
# MCP Resources - Agent Memory Access
# ============================================================================

@mcp.resource("memory://{agent_id}/summary")
def agent_memory_summary(agent_id: str) -> str:
    """
    Provides a summary of an agent's memory state.
    
    Use this to quickly check memory usage and status.
    """
    try:
        stats = get_memory_stats(agent_id=agent_id)
        
        summary = f"""
# Memory Summary: {agent_id}

**Role:** {stats['role']}
**Last Updated:** {stats['last_updated']}

## Layer Statistics

"""
        for layer_name, layer_stats in stats['layers'].items():
            summary += f"""
### {layer_name.replace('_', ' ').title()}
- Items: {layer_stats['count']}
- Tokens: {layer_stats['tokens']}
- Avg Importance: {layer_stats['avg_importance']:.2f}
- Avg Access Count: {layer_stats['avg_access_count']:.1f}
"""
        
        summary += f"""
## Total Usage
- Total Items: {stats['total']['items']}
- Total Tokens: {stats['total']['tokens']}
- Size Estimate: {stats['total']['size_kb_estimate']:.2f} KB
- Within Limits: {stats['limits']['within_limits']}
- Utilization: {stats['limits']['utilization_percent']:.1f}%
"""
        
        return summary
    
    except Exception as e:
        return f"Error generating summary for {agent_id}: {str(e)}"


@mcp.resource("memory://config")
def memory_config() -> str:
    """
    Provides the current memory management configuration.
    
    Shows limits, token budgets, and cleanup schedules.
    """
    config_text = f"""
# Memory Management Configuration

## Memory Limits

- Recent Layer: Max {MEMORY_CONFIG['limits']['recent_max_items']} items
- Medium-Term Layer: Max {MEMORY_CONFIG['limits']['medium_term_max_items']} items
- Long-Term Layer: Max {MEMORY_CONFIG['limits']['long_term_max_items']} items
- Total Size Limit: {MEMORY_CONFIG['limits']['total_max_size_kb']} KB per agent

## Token Budgets

- Core Memory: {MEMORY_CONFIG['token_budgets']['core_memory']} tokens
- Task Context: {MEMORY_CONFIG['token_budgets']['task_context']} tokens

## Cleanup Schedule

- **Daily (00:00):** Recent → Medium-Term (items > 24h old)
- **Sprint End (Manual):** Medium-Term → Long-Term or Compost (items > 14 days)
- **Quarterly:** Compost cleanup (delete items > 90 days with no access)

## Memory Layers

1. **Core:** Role definition, never changes
2. **Recent:** Last 24 hours, cleared nightly
3. **Medium-Term:** Current sprint, cleared at sprint end
4. **Long-Term:** Permanent important memories
5. **Compost:** Failed approaches, cleared quarterly
"""
    return config_text


# ============================================================================
# Server Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
