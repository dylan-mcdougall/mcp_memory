"""
Memory Management MCP Server for Multi-Agent Development Framework

This MCP server provides centralized memory management for agent collaboration,
handling storage, retrieval, pruning, and cross-agent queries for the development team.

Enhanced version integrating personality-shaping core memory philosophy:
- Core memories are PERSONALITY-DEFINING moments, not task outcomes
- Strict 3-sentence limit enforced
- Core memories always loaded first in context retrieval
- Formative experiences shape agent behavior across sessions

Redesigned tool set:
- store_memory, load_agent_context, search_memories, promote_memory,
  demote_memory, cleanup_memories, get_memory_stats, query_cross_agent_memory
- get_compression_candidates, store_extracted_facts
- NEW: add_core_memory (personality-shaping moments with validation)
- NEW: recall_identity (quick identity refresh on spawn)
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
import re

# Initialize FastMCP server
mcp = FastMCP("agent_memory_mcp")

# Configuration
MEMORY_BASE_PATH = Path(os.getenv("MEMORY_BASE_PATH", ".claude/agents/memories"))
MEMORY_BASE_PATH.mkdir(parents=True, exist_ok=True)

# File lock for async safety
file_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

# Memory limits configuration
MEMORY_CONFIG = {
    "limits": {
        "recent_max_items": 50,
        "medium_term_max_items": 200,
        "long_term_max_items": 500,
        "total_max_size_kb": 150,
        "core_memory_max_items": 10,  # Strict limit on core memories
        "core_memory_max_sentences": 3  # Per-item sentence limit
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
WRITABLE_LAYERS = Literal["recent", "medium_term", "long_term", "compost"]
PROMOTABLE_TARGET_LAYERS = Literal["medium_term", "long_term"]

# Core memory types - restricted vocabulary for personality-shaping moments
CORE_MEMORY_TYPES = Literal[
    "formative_lesson",      # "From that day forward, I always..."
    "behavioral_change",     # "After X incident, I learned to..."
    "working_dynamic",       # "I discovered I work best by..."
    "identity_principle",    # Core belief about role/purpose
    "collaboration_insight"  # How to work with specific agents
]


# ============================================================================
# Data Models
# ============================================================================

class CoreMemoryItem(BaseModel):
    """Schema for a personality-shaping core memory.
    
    Core memories are fundamentally different from task memories:
    - They define WHO the agent IS, not what it DID
    - Maximum 3 sentences - formative moments, not documentation
    - Always retrieved on spawn
    - Write sparingly - only truly behavior-changing experiences
    """
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    id: str
    key: str  # Short identifier (e.g., "research_verification_principle")
    memory_type: CORE_MEMORY_TYPES
    content: str  # The personality-shaping realization (max 3 sentences)
    formed_from: Optional[str] = None  # Reference to incident that shaped this
    created: str
    last_reinforced: str  # Updated when a situation validates this memory
    reinforcement_count: int = Field(default=0, ge=0)


class MemoryItem(BaseModel):
    """Schema for a standard memory item (non-core)."""
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


class AgentIdentity(BaseModel):
    """Schema for agent's core identity - always loaded first.
    
    This represents the stable "personality" of an agent across sessions.
    """
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    agent_name: str
    role: str
    voice: str = "professional"  # Communication style
    motto: Optional[str] = None  # Guiding principle in one line
    personality_traits: List[str] = Field(default_factory=list)  # e.g., ["skeptical", "thorough"]
    biases: List[str] = Field(default_factory=list)  # Known productive biases
    specialty: Optional[str] = None


class AgentMemory(BaseModel):
    """Schema for an agent's complete memory structure."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True)
    
    agent_id: str
    identity: AgentIdentity
    last_updated: str
    core_memories: List[CoreMemoryItem] = Field(default_factory=list)
    memory_layers: Dict[str, List[MemoryItem]] = Field(
        default_factory=lambda: {
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

class AddCoreMemoryInput(BaseModel):
    """Input for adding a personality-shaping core memory.
    
    ⚠️ USE EXTREMELY SPARINGLY - core memories define WHO YOU ARE.
    """
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    key: str = Field(
        ..., 
        description="Short identifier for this formative moment (e.g., 'research_verification_principle')",
        min_length=3,
        max_length=50,
        pattern=r'^[a-z][a-z0-9_]*$'
    )
    memory_type: CORE_MEMORY_TYPES = Field(
        ...,
        description="Type of core memory: 'formative_lesson', 'behavioral_change', 'working_dynamic', 'identity_principle', or 'collaboration_insight'"
    )
    content: str = Field(
        ..., 
        description="The personality-shaping realization (MAXIMUM 3 sentences). Write as a defining moment that changed how you approach work.",
        min_length=10,
        max_length=500
    )
    formed_from: Optional[str] = Field(
        None, 
        description="Optional reference to the incident/task that shaped this memory"
    )


class ReinforceCoreMemoryInput(BaseModel):
    """Input for reinforcing an existing core memory when validated by experience."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    key: str = Field(..., description="Key of the core memory to reinforce", min_length=3, max_length=50)
    reinforcing_incident: Optional[str] = Field(
        None,
        description="Brief description of what validated this memory"
    )


class RecallIdentityInput(BaseModel):
    """Input for quick identity refresh on agent spawn."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)


class UpdateIdentityInput(BaseModel):
    """Input for updating agent identity fields."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    voice: Optional[str] = Field(None, description="Communication style", max_length=50)
    motto: Optional[str] = Field(None, description="Guiding principle in one line", max_length=200)
    personality_traits: Optional[List[str]] = Field(None, description="Personality traits", max_items=10)
    biases: Optional[List[str]] = Field(None, description="Known productive biases", max_items=10)
    specialty: Optional[str] = Field(None, description="Area of specialty", max_length=100)


class StoreMemoryInput(BaseModel):
    """Input for storing a new memory item."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent storing the memory", min_length=1, max_length=100)
    layer: WRITABLE_LAYERS = Field(..., description="Memory persistence layer: 'recent', 'medium_term', 'long_term', or 'compost'")
    memory_type: str = Field(..., description="Type of memory (e.g., 'decision', 'implementation', 'bug_fix', 'lesson_learned')", min_length=1, max_length=50)
    content: str = Field(..., description="The actual memory content to store", min_length=1)
    tags: List[str] = Field(default_factory=list, description="Tags for categorization and filtering", max_items=20)
    importance: float = Field(default=0.5, description="Importance score from 0.0 (low) to 1.0 (critical)", ge=0.0, le=1.0)
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context (task_id, related_files, related_agents)")
    related_to: List[str] = Field(default_factory=list, description="IDs of related memories", max_items=50)


class LoadAgentContextInput(BaseModel):
    """Input for loading relevant memory context for a task."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    
    agent_id: str = Field(..., description="ID of the agent", min_length=1, max_length=100)
    task_description: Optional[str] = Field(None, description="Description of current task for relevance filtering")
    task_tags: List[str] = Field(default_factory=list, description="Tags related to current task", max_items=20)
    max_tokens: int = Field(default=5000, description="Maximum tokens to return", ge=100, le=50000)
    include_layers: List[str] = Field(
        default=["recent", "medium_term"],
        description="Which memory layers to include (core + identity always included)"
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
    include_core: bool = Field(default=False, description="Also search core memories")


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
    include_core: bool = Field(default=False, description="Also search core memories")


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

def count_sentences(text: str) -> int:
    """Count sentences in text (handles common abbreviations)."""
    # Simple heuristic: count sentence-ending punctuation
    # Excluding common abbreviations like "e.g.", "i.e.", "etc."
    cleaned = re.sub(r'\b(e\.g\.|i\.e\.|etc\.|vs\.|Mr\.|Ms\.|Dr\.)', 'ABBREV', text)
    return len(re.findall(r'[.!?]+', cleaned))


def estimate_tokens(text: str) -> int:
    """Estimate token count based on content type."""
    if '{' in text or 'def ' in text or 'class ' in text:
        return int(len(text) / 4)
    return int(len(text) / 3.5)


def generate_memory_id() -> str:
    """Generate a unique memory ID using timestamp and hash."""
    timestamp = datetime.now().isoformat()
    random_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"mem_{random_hash}"


def generate_core_memory_id() -> str:
    """Generate a unique core memory ID."""
    timestamp = datetime.now().isoformat()
    random_hash = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"core_{random_hash}"


def generate_content_hash(content: str) -> str:
    """Generate hash of content for deduplication."""
    return hashlib.md5(content.encode()).hexdigest()


def get_memory_file_path(agent_id: str) -> Path:
    """Get the file path for an agent's memory storage."""
    return MEMORY_BASE_PATH / f"{agent_id}-memory.json"


async def load_agent_memory(agent_id: str) -> AgentMemory:
    """Load an agent's memory from disk with async safety."""
    memory_file = get_memory_file_path(agent_id)

    async with file_locks[agent_id]:
        try:
            file_exists = await aiofiles.os.path.exists(memory_file)
        except Exception:
            file_exists = False

        if not file_exists:
            memory = AgentMemory(
                agent_id=agent_id,
                identity=AgentIdentity(
                    agent_name=agent_id,
                    role="unknown"
                ),
                last_updated=datetime.now().isoformat()
            )
            await _save_agent_memory_unlocked(memory)
            return memory

        try:
            async with aiofiles.open(memory_file, 'r') as f:
                content = await f.read()
                data = json.loads(content)
                
                # Handle migration from old format (without identity/core_memories)
                if 'identity' not in data:
                    data['identity'] = {
                        'agent_name': data.get('agent_id', agent_id),
                        'role': data.get('role', 'unknown')
                    }
                if 'core_memories' not in data:
                    data['core_memories'] = []
                    
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


def search_core_memories_by_query(memories: List[CoreMemoryItem], query: str) -> List[CoreMemoryItem]:
    """Search core memories by keyword matching."""
    query_lower = query.lower()
    results = []
    
    for mem in memories:
        if (query_lower in mem.content.lower() or
            query_lower in mem.key.lower() or
            query_lower in mem.memory_type.lower()):
            results.append(mem)
    
    return results


def calculate_relevance_score(memory: MemoryItem) -> float:
    """Calculate relevance score for memory ranking."""
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
    """Find a memory item by ID across all layers."""
    for layer_name, items in memory.memory_layers.items():
        for item in items:
            if item.id == memory_id:
                return item, layer_name
    return None, None


def find_core_memory_by_key(memory: AgentMemory, key: str) -> Optional[CoreMemoryItem]:
    """Find a core memory by its key."""
    for item in memory.core_memories:
        if item.key == key:
            return item
    return None


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
    """Group memories by memory_type and tag overlap."""
    if not memories:
        return []
    
    type_groups: Dict[str, List[MemoryItem]] = defaultdict(list)
    for mem in memories:
        type_groups[mem.memory_type].append(mem)
    
    result_groups = []
    
    for memory_type, type_memories in type_groups.items():
        if len(type_memories) < min_group_size:
            continue
        
        used = set()
        for i, mem in enumerate(type_memories):
            if i in used:
                continue
            
            cluster = [mem]
            used.add(i)
            
            for j, other in enumerate(type_memories):
                if j in used:
                    continue
                
                for cluster_mem in cluster:
                    if calculate_tag_overlap(cluster_mem.tags, other.tags) >= 0.3:
                        cluster.append(other)
                        used.add(j)
                        break
            
            if len(cluster) >= min_group_size:
                result_groups.append(cluster)
    
    return result_groups


def format_identity_summary(identity: AgentIdentity) -> str:
    """Format identity for display."""
    lines = [
        f"**{identity.agent_name}** ({identity.role})",
        f"Voice: {identity.voice}"
    ]
    
    if identity.motto:
        lines.append(f'Motto: "{identity.motto}"')
    
    if identity.personality_traits:
        lines.append(f"Traits: {', '.join(identity.personality_traits)}")
    
    if identity.biases:
        lines.append(f"Biases: {', '.join(identity.biases)}")
    
    if identity.specialty:
        lines.append(f"Specialty: {identity.specialty}")
    
    return "\n".join(lines)


def format_core_memories_summary(core_memories: List[CoreMemoryItem]) -> str:
    """Format core memories for display."""
    if not core_memories:
        return "No core memories yet."
    
    lines = []
    for mem in core_memories:
        reinforced = f" (reinforced {mem.reinforcement_count}x)" if mem.reinforcement_count > 0 else ""
        lines.append(f"• [{mem.key}]{reinforced}: {mem.content}")
    
    return "\n".join(lines)


# ============================================================================
# MCP Tools - Core Memory & Identity
# ============================================================================

@mcp.tool(
    name="add_core_memory",
    annotations={
        "title": "Add Core Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def add_core_memory(params: AddCoreMemoryInput) -> Dict[str, Any]:
    """Add a PERSONALITY-SHAPING MOMENT to core memory.
    
    ⚠️ WARNING: Use EXTREMELY sparingly! Core memories define WHO YOU ARE.
    
    Core memories are for FORMATIVE, BEHAVIOR-CHANGING experiences:
    ✅ GOOD: "From that day forward, I always verify research before implementation"
    ✅ GOOD: "After the fabrication incident, I learned to question round numbers"
    ✅ GOOD: "I discovered I work best by challenging Cynthia's optimism constructively"
    
    ❌ BAD: Project details, task outcomes, temporary insights
    ❌ BAD: "Completed citation audit" (use long-term memory instead)
    ❌ BAD: "Found paper X useful" (use recent learnings instead)
    
    Core memories are ALWAYS shown on recall. They shape your personality forever.
    
    Args:
        params: AddCoreMemoryInput with agent_id, key, memory_type, content, formed_from.
    
    Returns:
        Dict with success status, core_memory details, and warnings if applicable.
    
    Raises:
        RuntimeError: If sentence limit exceeded or core memory limit reached.
    """
    # Enforce 3-sentence limit
    sentence_count = count_sentences(params.content)
    max_sentences = MEMORY_CONFIG["limits"]["core_memory_max_sentences"]
    
    if sentence_count > max_sentences:
        return {
            "success": False,
            "error": f"Core memory too long ({sentence_count} sentences). Maximum {max_sentences} sentences.",
            "guidance": "Core memories must be concise personality-defining moments, not documentation."
        }
    
    memory = await load_agent_memory(params.agent_id)
    
    # Check for existing key
    existing = find_core_memory_by_key(memory, params.key)
    if existing:
        return {
            "success": False,
            "error": f"Core memory with key '{params.key}' already exists.",
            "guidance": "Use reinforce_core_memory to strengthen existing memories, or choose a different key."
        }
    
    # Enforce core memory limit
    max_core = MEMORY_CONFIG["limits"]["core_memory_max_items"]
    if len(memory.core_memories) >= max_core:
        return {
            "success": False,
            "error": f"Core memory limit reached ({max_core} items).",
            "guidance": "Core memories are precious. Consider if this truly defines who you are, or if it belongs in long-term memory instead.",
            "current_core_memories": [m.key for m in memory.core_memories]
        }

    core_memory = CoreMemoryItem(
        id=generate_core_memory_id(),
        key=params.key,
        memory_type=params.memory_type,
        content=params.content,
        formed_from=params.formed_from,
        created=datetime.now().isoformat(),
        last_reinforced=datetime.now().isoformat(),
        reinforcement_count=0
    )
    
    memory.core_memories.append(core_memory)
    await save_agent_memory(memory)
    
    return {
        "success": True,
        "message": f"⚠️ Core memory added: {params.key}",
        "warning": "This memory will shape your behavior forever. It will be loaded on every spawn.",
        "core_memory": core_memory.model_dump(),
        "total_core_memories": len(memory.core_memories)
    }


@mcp.tool(
    name="reinforce_core_memory",
    annotations={
        "title": "Reinforce Core Memory",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def reinforce_core_memory(params: ReinforceCoreMemoryInput) -> Dict[str, Any]:
    """Reinforce an existing core memory when experience validates it.
    
    Use this when a situation confirms that a core memory's lesson was correct.
    This tracks how often core beliefs prove valuable.
    
    Args:
        params: ReinforceCoreMemoryInput with agent_id, key, and optional reinforcing_incident.
    
    Returns:
        Dict with success status and updated reinforcement count.
    """
    memory = await load_agent_memory(params.agent_id)
    
    core_mem = find_core_memory_by_key(memory, params.key)
    if not core_mem:
        return {
            "success": False,
            "error": f"Core memory '{params.key}' not found.",
            "available_keys": [m.key for m in memory.core_memories]
        }
    
    core_mem.reinforcement_count += 1
    core_mem.last_reinforced = datetime.now().isoformat()
    
    await save_agent_memory(memory)
    
    return {
        "success": True,
        "key": params.key,
        "reinforcement_count": core_mem.reinforcement_count,
        "message": f"Core memory reinforced. This principle has proven valuable {core_mem.reinforcement_count} time(s)."
    }


@mcp.tool(
    name="recall_identity",
    annotations={
        "title": "Recall Identity",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def recall_identity(params: RecallIdentityInput) -> Dict[str, Any]:
    """Recall your identity and core memories (use on spawn).
    
    Returns a concise summary of WHO YOU ARE - your role, personality,
    and the formative experiences that shaped you.
    
    Args:
        params: RecallIdentityInput with agent_id.
    
    Returns:
        Dict with formatted identity and core memories.
    """
    memory = await load_agent_memory(params.agent_id)
    
    identity_summary = format_identity_summary(memory.identity)
    core_summary = format_core_memories_summary(memory.core_memories)
    
    return {
        "agent_id": params.agent_id,
        "identity": identity_summary,
        "core_memories": core_summary,
        "raw_identity": memory.identity.model_dump(),
        "raw_core_memories": [m.model_dump() for m in memory.core_memories],
        "last_updated": memory.last_updated
    }


@mcp.tool(
    name="update_identity",
    annotations={
        "title": "Update Identity",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def update_identity(params: UpdateIdentityInput) -> Dict[str, Any]:
    """Update agent identity fields.
    
    Use to refine personality traits, voice, motto, biases, or specialty
    as the agent develops over time.
    
    Args:
        params: UpdateIdentityInput with fields to update.
    
    Returns:
        Dict with updated identity.
    """
    memory = await load_agent_memory(params.agent_id)
    
    updated_fields = []
    
    if params.voice is not None:
        memory.identity.voice = params.voice
        updated_fields.append("voice")
    
    if params.motto is not None:
        memory.identity.motto = params.motto
        updated_fields.append("motto")
    
    if params.personality_traits is not None:
        memory.identity.personality_traits = params.personality_traits
        updated_fields.append("personality_traits")
    
    if params.biases is not None:
        memory.identity.biases = params.biases
        updated_fields.append("biases")
    
    if params.specialty is not None:
        memory.identity.specialty = params.specialty
        updated_fields.append("specialty")
    
    if not updated_fields:
        return {
            "success": False,
            "error": "No fields provided to update."
        }
    
    await save_agent_memory(memory)
    
    return {
        "success": True,
        "updated_fields": updated_fields,
        "identity": memory.identity.model_dump()
    }


# ============================================================================
# MCP Tools - Standard Memory Operations
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
    
    For task outcomes, decisions, implementations, and other work memories.
    NOT for personality-shaping moments (use add_core_memory instead).
    
    Args:
        params: StoreMemoryInput with agent_id, layer, memory_type, content, etc.
    
    Returns:
        Dict with success status, memory_id, and action taken.
    """
    memory = await load_agent_memory(params.agent_id)
    
    content_hash = generate_content_hash(params.content)
    
    # Check for duplicates in the same layer
    for item in memory.memory_layers[params.layer]:
        if item.content_hash == content_hash:
            item.access_count += 1
            item.last_accessed = datetime.now().isoformat()
            await save_agent_memory(memory)
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
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def load_agent_context(params: LoadAgentContextInput) -> Dict[str, Any]:
    """Load relevant memory context for an agent's current task.
    
    ALWAYS includes identity and core memories first (these define who you are),
    then adds task-relevant memories from specified layers within token budget.
    
    Args:
        params: LoadAgentContextInput with agent_id, task_description, etc.
    
    Returns:
        Dict with identity, core_memories, context, tokens_used, and items_loaded.
    """
    memory = await load_agent_memory(params.agent_id)
    
    context_parts = []
    tokens_used = 0
    items_loaded = []
    
    # 1. ALWAYS include identity first
    identity_text = format_identity_summary(memory.identity)
    identity_tokens = estimate_tokens(identity_text)
    tokens_used += identity_tokens
    
    # 2. ALWAYS include core memories (these define WHO YOU ARE)
    core_memories_formatted = []
    for core_mem in memory.core_memories:
        core_tokens = estimate_tokens(core_mem.content)
        if tokens_used + core_tokens <= params.max_tokens:
            core_memories_formatted.append({
                "key": core_mem.key,
                "type": core_mem.memory_type,
                "content": core_mem.content,
                "reinforcement_count": core_mem.reinforcement_count
            })
            tokens_used += core_tokens
    
    # 3. Load recent memories with task relevance filtering
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
    
    # 4. Load medium-term memories if space available
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
    
    # 5. Load long-term memories if space available
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
        "identity": memory.identity.model_dump(),
        "core_memories": core_memories_formatted,
        "context": context_parts,
        "tokens_used": tokens_used,
        "items_loaded": len(items_loaded),
        "memory_ids": items_loaded
    }


@mcp.tool(
    name="search_memories",
    annotations={
        "title": "Search Agent Memories",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def search_memories(params: SearchMemoriesInput) -> Dict[str, Any]:
    """Search across agent memories using keywords and filters.
    
    Args:
        params: SearchMemoriesInput with query, layers, tags, etc.
    
    Returns:
        Dict with results from standard memories and optionally core memories.
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
    
    response = {
        "agent_id": params.agent_id,
        "query": params.query,
        "results_count": len(formatted_results),
        "memories": formatted_results
    }
    
    # Optionally search core memories
    if params.include_core:
        core_results = search_core_memories_by_query(memory.core_memories, params.query)
        response["core_memories"] = [
            {
                "key": m.key,
                "type": m.memory_type,
                "content": m.content,
                "reinforcement_count": m.reinforcement_count
            }
            for m in core_results
        ]
    
    return response


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
    
    Args:
        params: PromoteMemoryInput with memory_id and target layer.
    
    Returns:
        Dict with promotion details.
    """
    memory = await load_agent_memory(params.agent_id)
    
    found_memory, source_layer = find_memory_by_id(memory, params.memory_id)
    
    if not found_memory:
        raise RuntimeError(f"Memory {params.memory_id} not found for agent {params.agent_id}")
    
    if source_layer == "long_term":
        raise RuntimeError("Cannot promote from long_term layer")
    
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
    
    Args:
        params: DemoteMemoryInput with memory_id and destination.
    
    Returns:
        Dict with demotion details.
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
    
    Args:
        params: CleanupMemoriesInput with cleanup_type and dry_run flag.
    
    Returns:
        Dict with cleanup summary.
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
    
    Args:
        params: GetMemoryStatsInput with agent_id and health metrics flag.
    
    Returns:
        Dict with comprehensive memory statistics.
    """
    memory = await load_agent_memory(params.agent_id)
    
    stats: Dict[str, Any] = {
        "agent_id": params.agent_id,
        "identity": {
            "name": memory.identity.agent_name,
            "role": memory.identity.role,
            "traits": memory.identity.personality_traits
        },
        "last_updated": memory.last_updated,
        "core_memories": {
            "count": len(memory.core_memories),
            "limit": MEMORY_CONFIG["limits"]["core_memory_max_items"],
            "keys": [m.key for m in memory.core_memories],
            "total_reinforcements": sum(m.reinforcement_count for m in memory.core_memories)
        },
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
    
    # Add core memory tokens to total
    core_tokens = sum(estimate_tokens(m.content) for m in memory.core_memories)
    total_tokens += core_tokens
    
    stats["total"] = {
        "items": total_items + len(memory.core_memories),
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
    
    if params.include_health_metrics:
        health_metrics: Dict[str, Any] = {}
        
        all_hashes = []
        duplicate_count = 0
        for layer_name, items in memory.memory_layers.items():
            hashes = [item.content_hash for item in items if item.content_hash]
            duplicate_count += len(hashes) - len(set(hashes))
            all_hashes.extend(hashes)
        
        health_metrics["redundancy_score"] = round(
            duplicate_count / max(1, len(all_hashes)), 3
        )
        health_metrics["duplicate_count"] = duplicate_count
        
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
        
        # Core memory health
        unreinforced_core = sum(1 for m in memory.core_memories if m.reinforcement_count == 0)
        health_metrics["unreinforced_core_memories"] = unreinforced_core
        
        health_score = 1.0
        health_score -= health_metrics["redundancy_score"] * 0.3
        health_score -= min(0.2, stale_count * 0.005)
        health_score -= min(0.1, unreinforced_core * 0.02)
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
    """Search memories across multiple agents.
    
    Args:
        params: QueryCrossAgentMemoryInput with query and filters.
    
    Returns:
        Dict with cross-agent search results.
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
                        "agent_role": memory.identity.role,
                        "memory_id": mem.id,
                        "layer": mem.layer,
                        "type": mem.memory_type,
                        "content": mem.content,
                        "tags": mem.tags,
                        "importance": mem.importance,
                        "relevance_score": round(calculate_relevance_score(mem), 3),
                        "created": mem.created
                    })
            
            # Optionally search core memories
            if params.include_core:
                core_results = search_core_memories_by_query(memory.core_memories, params.query)
                for mem in core_results:
                    all_results.append({
                        "agent_id": agent_id,
                        "agent_role": memory.identity.role,
                        "memory_id": mem.id,
                        "layer": "core",
                        "type": mem.memory_type,
                        "content": mem.content,
                        "key": mem.key,
                        "importance": 1.0,  # Core memories are always high importance
                        "relevance_score": 1.0,
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
    """Identify groups of memories suitable for fact extraction.
    
    Args:
        params: GetCompressionCandidatesInput with filtering criteria.
    
    Returns:
        Dict with candidate groups for compression.
    """
    memory = await load_agent_memory(params.agent_id)
    
    now = datetime.now()
    eligible_memories: List[MemoryItem] = []
    
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
    
    candidate_groups = []
    total_compression_tokens = 0
    
    for idx, group in enumerate(groups):
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
    
    Args:
        params: StoreExtractedFactsInput with facts and original memory IDs.
    
    Returns:
        Dict with compression results.
    """
    memory = await load_agent_memory(params.agent_id)
    
    original_memories: List[tuple[MemoryItem, str]] = []
    original_tokens = 0
    all_original_tags: set = set()
    
    for mem_id in params.original_memory_ids:
        found_memory, layer = find_memory_by_id(memory, mem_id)
        if not found_memory:
            raise RuntimeError(
                f"Memory {mem_id} not found for agent {params.agent_id}."
            )
        original_memories.append((found_memory, layer))
        original_tokens += found_memory.size_estimate_tokens
        all_original_tags.update(found_memory.tags)
    
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
    
    combined_tags.update(all_original_tags)
    
    compressed_tokens = estimate_tokens(compressed_content)
    content_hash = generate_content_hash(compressed_content)
    
    compressed_memory = MemoryItem(
        id=generate_memory_id(),
        layer="long_term",
        memory_type=params.memory_type,
        content=compressed_content,
        tags=list(combined_tags)[:20],
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
            pass
    
    memory.memory_layers["long_term"].append(compressed_memory)
    
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

@mcp.resource("memory://{agent_id}/identity")
async def agent_identity_resource(agent_id: str) -> str:
    """Provides agent identity and core memories."""
    try:
        result = await recall_identity(RecallIdentityInput(agent_id=agent_id))
        return f"""# Agent Identity: {agent_id}

{result['identity']}

## Core Memories (Personality-Shaping Moments)

{result['core_memories']}

*Last updated: {result['last_updated']}*
"""
    except Exception as e:
        return f"Error loading identity for {agent_id}: {e}"


@mcp.resource("memory://{agent_id}/summary")
async def agent_memory_summary(agent_id: str) -> str:
    """Provides a summary of an agent's memory state."""
    try:
        stats = await get_memory_stats(GetMemoryStatsInput(
            agent_id=agent_id, 
            include_health_metrics=True
        ))
        
        summary = f"""# Memory Summary: {agent_id}

**Name:** {stats['identity']['name']}
**Role:** {stats['identity']['role']}
**Traits:** {', '.join(stats['identity']['traits']) if stats['identity']['traits'] else 'None defined'}
**Last Updated:** {stats['last_updated']}

## Core Memories
- Count: {stats['core_memories']['count']}/{stats['core_memories']['limit']}
- Keys: {', '.join(stats['core_memories']['keys']) if stats['core_memories']['keys'] else 'None'}
- Total Reinforcements: {stats['core_memories']['total_reinforcements']}

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
- Unreinforced Core Memories: {hm['unreinforced_core_memories']}
"""
        
        return summary
    
    except Exception as e:
        return f"Error generating summary for {agent_id}: {e}"


@mcp.resource("memory://config")
def memory_config_resource() -> str:
    """Provides the current memory management configuration."""
    return f"""# Memory Management Configuration

## Memory Limits

- Recent Layer: Max {MEMORY_CONFIG['limits']['recent_max_items']} items
- Medium-Term Layer: Max {MEMORY_CONFIG['limits']['medium_term_max_items']} items
- Long-Term Layer: Max {MEMORY_CONFIG['limits']['long_term_max_items']} items
- Core Memories: Max {MEMORY_CONFIG['limits']['core_memory_max_items']} items
- Core Memory Sentences: Max {MEMORY_CONFIG['limits']['core_memory_max_sentences']} per item
- Total Size Limit: {MEMORY_CONFIG['limits']['total_max_size_kb']} KB per agent

## Token Budgets

- Core Memory: {MEMORY_CONFIG['token_budgets']['core_memory']} tokens
- Task Context: {MEMORY_CONFIG['token_budgets']['task_context']} tokens

## Core Memory Philosophy

Core memories are PERSONALITY-DEFINING moments, not task outcomes:

✅ GOOD core memories:
- "From that day forward, I always verify research before implementation"
- "After the fabrication incident, I learned to question round numbers"
- "I discovered I work best by challenging optimism constructively"

❌ BAD core memories:
- "Completed the citation audit" (use long-term instead)
- "Found paper X useful" (use recent learnings instead)
- Project details, task outcomes, temporary insights

Core memories are:
- Always loaded on spawn (they define WHO YOU ARE)
- Limited to {MEMORY_CONFIG['limits']['core_memory_max_sentences']} sentences max
- Capped at {MEMORY_CONFIG['limits']['core_memory_max_items']} total (precious space)
- Tracked for reinforcement (how often they prove valuable)

## Compression Settings

- Minimum Age for Compression: {MEMORY_CONFIG['compression']['min_age_days']} days
- Minimum Group Size: {MEMORY_CONFIG['compression']['min_group_size']} memories
- Maximum Access Count: {MEMORY_CONFIG['compression']['max_access_count']}
- Fact Length: {MEMORY_CONFIG['compression']['fact_min_words']}-{MEMORY_CONFIG['compression']['fact_max_words']} words

## Memory Layers

1. **Core:** Personality-defining moments (NOT tasks)
2. **Recent:** Last 24 hours, auto-promoted to medium-term
3. **Medium-Term:** Current sprint context, 2-week lifecycle
4. **Long-Term:** Valuable memories + compressed facts
5. **Compost:** Demoted memories, cleared quarterly

## Cleanup Schedule

- **Daily (00:00):** Recent → Medium-Term (items > 24h old with importance > 0.3)
- **Sprint End (Manual):** Medium-Term → Long-Term (importance > 0.6 or access_count > 3) or Compost
- **Quarterly:** Compost cleanup (delete items > 90 days with no access)
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
