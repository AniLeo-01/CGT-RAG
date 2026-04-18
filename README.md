# GRACE: Graph Reasoning via Adaptive Context Expansion

## Motivation

Modern Retrieval-Augmented Generation (RAG) systems rely on **vector similarity search** to retrieve relevant context. While effective for many use cases, this approach has fundamental limitations:

- **Shallow Retrieval**: Top-k vector matches capture surface-level similarity but miss deeper, multi-hop relational structure.
- **Static Context**: Traditional RAG retrieves a fixed context window without iterative refinement based on missing information.
- **Context Incompleteness**: Complex queries require connecting information across documents, entities, and concepts — something flat retrieval cannot handle.
- **Fixed Retrieval Depth**: A predefined `k` is insufficient for queries that require adaptive, query-conditioned exploration.

### Key Insight

Answering complex queries is not purely a retrieval problem — it is a **reasoning problem over structured knowledge**.

GRACE addresses this by combining vector search with **LLM-guided graph traversal**, enabling iterative context expansion until sufficient information is gathered to answer a query.

---

## Theoretical Framework

GRACE reframes retrieval as a **sequential decision-making process over a knowledge graph**.

---

### 1. Problem Formulation

Let:

- $`q`$ — user query
- $`\mathcal{G} = (\mathcal{V}, \mathcal{E})`$ — knowledge graph with nodes $`\mathcal{V}`$ and typed edges $`\mathcal{E}`$
- $`\phi: \mathcal{V} \to \mathbb{R}^d`$ — node embedding function
- $`C_t \subseteq \mathcal{V}`$ — context set at step $`t`$
- $`\pi`$ — traversal policy

The objective is to construct a final answer:

```math
\text{Answer} = f(q,\ C_T)
```

where $`C_T`$ is the accumulated context:

```math
C_T = \bigcup_{t=0}^{T} \Delta C_t
```

and each context increment is produced by the traversal policy:

```math
\Delta C_t = \pi(q,\ C_t,\ \mathcal{G})
```

---

### 2. System Architecture

GRACE operates in three stages:

#### (a) Semantic Entry — Vector Retrieval

Initial graph entry points are selected via approximate nearest-neighbor search over node embeddings:

```math
N_0 = \text{TopK}_{n \in \mathcal{V}} \left( \text{sim}(q,\ \phi(n)) \right)
```

These seed nodes anchor subsequent graph traversal.

---

#### (b) Iterative Graph Expansion

Rather than issuing additional vector queries, GRACE expands context by traversing graph neighborhoods:

```math
C_{t+1} = C_t \cup \text{Neighbors}(C_t,\ a_t)
```

where the traversal action is:

```math
a_t = \pi(q,\ C_t)
```

The policy $`\pi`$ determines:
- which nodes to expand
- which edge types (relations) to follow
- how much context to admit per step

---

#### (c) Context Sufficiency Check

At each step, a sufficiency estimator evaluates accumulated context:

```math
S(C_t,\ q) \in [0,\ 1]
```

Traversal terminates when:

```math
S(C_t,\ q) \geq \tau
```

or when resource constraints (token budget $`B`$, maximum hops $`H`$) are reached.

---

### 3. Traversal as a Learned Policy

Unlike deterministic graph search (BFS/DFS), GRACE employs a **policy-driven traversal**:

```math
a_t = \pi_{\text{LLM}}(q,\ C_t)
```

This enables:
- dynamic, query-conditioned exploration
- relation-aware multi-hop reasoning
- context-sensitive pruning of irrelevant subgraphs

---

### 4. Graph Representation

The knowledge graph is organized across three semantic levels:

| Level | Unit | Role |
|-------|------|------|
| Surface | Chunk | Retrieval unit (text passage) |
| Structural | Entity | Reasoning unit (named entity, concept instance) |
| Abstract | Concept | Abstraction layer (category, topic) |

Traversal follows the pattern:

```math
\text{Chunk} \xrightarrow{\text{extract}} \text{Entity} \xrightarrow{\text{relate}} \text{Entity} \xrightarrow{\text{ground}} \text{Chunk}
```

This bidirectional text-structure-text traversal enables semantic expansion while limiting noise accumulation.

---

### 5. Objective Function

GRACE optimizes for answer quality over the accumulated context $`C`$:

```math
\max_{C}\ \text{AnswerQuality}(C)
```

subject to:

```math
|C| \cdot \bar{l} \leq B \qquad \text{(token budget)}
```

```math
T \leq H \qquad \text{(max traversal hops)}
```

```math
\frac{d}{dC}\text{AnswerQuality}(C) \geq \epsilon \qquad \text{(diminishing returns threshold)}
```

This contrasts with standard ANN-based retrieval (e.g., HNSW), which minimizes:

```math
\min_{x \in \mathcal{V}}\ d(q,\ x)
```

GRACE prioritizes **information completeness** over embedding proximity.

---

### 6. MDP Formulation

GRACE has a natural Markov Decision Process (MDP) interpretation:

| MDP Component | GRACE Instantiation |
|---------------|---------------------|
| State $`s_t`$ | $`(q,\ C_t)`$ |
| Action $`a_t`$ | traversal decision (node/edge selection) |
| Transition $`T`$ | graph expansion: $`C_t \to C_{t+1}`$ |
| Reward $`r`$ | $`\text{AnswerAccuracy}(C_T) - \lambda \cdot \text{Cost}(T)`$ |

This formulation opens pathways to:
- reinforcement learning for traversal policy optimization
- imitation learning from expert retrieval traces
- cost-aware, token-budget-sensitive reasoning

---

### 7. Key Properties

| Property | Description |
|----------|-------------|
| Adaptive Depth | No fixed hop count; traversal depth conditioned on query and context state |
| Semantic Navigation | Expansion guided by relational semantics, not embedding distance |
| Closed-loop Retrieval | Retrieval and generation reasoning are interleaved, not sequential |
| Structured Expansion | Exploits graph topology to constrain and direct context growth |

---

## Summary

GRACE generalizes RAG into a **graph-based reasoning system** with four functional components:

| Component | Role |
|-----------|------|
| Vector search | Semantic entry — identifies initial graph nodes |
| Graph traversal | Structured expansion — follows relational paths |
| LLM policy $`\pi`$ | Directed exploration — decides where to traverse |
| Sufficiency check $`S`$ | Termination condition — determines when to stop |

This transforms retrieval from a one-shot operation into an **iterative, goal-directed process**, enabling more accurate and explainable answers for complex, multi-hop queries.

---

> GRACE retrieves not just relevant documents, but the **reasoning path required to answer the query**.

---

## Plausibility Assessment

### What Works Well

The **inflection-point termination** is the novel contribution. Rather than a fixed hop count or a binary "sufficient/not-sufficient" LLM call, watching for when marginal context relevancy *decreases* provides a natural, query-adaptive stopping boundary. This avoids both under-retrieval (too few hops) and noise accumulation (too many hops). The combination of vector-seeded entry + graph expansion is well-validated by multiple production systems (Microsoft GraphRAG, LightRAG, Neo4j GenAI).

### Critical Bottleneck — Context Relevancy Scoring

The main challenge is the **context relevancy scorer**. If implemented as an LLM call at every traversal step for every neighbor of every seed node, costs and latency become prohibitive. With $`k=5`$ seed nodes, average degree 6, and 3 hops of expansion, this yields up to $`5 \times 6 \times 3 = 90`$ LLM calls per query. At ~500ms and ~\$0.003 each (using a fast model), that's **45 seconds and \$0.27 per query** — not viable for production.

**Mitigation — Hybrid Scoring:**

- **Primary scorer:** A cross-encoder model (e.g., `bge-reranker-v2-m3`) that scores `(query, node_text)` pairs in ~5ms per pair on GPU. Scoring 100 neighbors costs ~500ms total with no LLM call.
- **LLM fallback:** Use an LLM call only when the cross-encoder score falls within an ambiguous confidence band, or as a periodic calibration check. This keeps LLM calls to ~2–5 per query.

### Differentiation from Existing Systems

| System | Traversal | Termination | Scoring |
|--------|-----------|-------------|---------|
| Microsoft GraphRAG | Community summarization (pre-computed) | Fixed community levels | Map-reduce over summaries |
| LightRAG | Dual-mode (low-level + high-level) | Fixed retrieval | Embedding similarity |
| ReMindRAG | Node explore/exploit with memory replay | Heuristic | Edge embeddings |
| **GRACE** | **Inflection-point adaptive traversal** | **Context relevancy inflection** | **Hybrid cross-encoder + LLM** |

---

## Retrieval Algorithm

1. User provides the query; the graph is preloaded in Neo4j with dynamic labels and embeddings.
2. Retrieve the top-$`k`$ nodes with highest embedding similarity and compute the context relevancy score for each against the query.
3. For each of the $`k`$ seed nodes, fetch adjacent connected $`n`$ nodes (default $`n=1`$) and their relationships. Score the expanded path for context relevancy.
4. If relevancy increases, continue traversal until the **inflection point** — the node at which context relevancy begins decreasing. The accumulated path becomes the context/document for that seed. Each of the $`k`$ seeds produces one or more context paths.

### Traversal Pseudocode

```python
def traverse_from_seed(seed_node, query, scorer):
    context_path = [seed_node]
    current_score = scorer.score(query, seed_node)

    frontier = get_neighbors(seed_node)  # 1-hop Cypher query

    while frontier:
        # Batch score all neighbors with cross-encoder
        scored = [(n, scorer.score(query, n)) for n in frontier]
        best_neighbor, best_score = max(scored, key=lambda x: x[1])

        if best_score < current_score:
            # Inflection point — relevancy is decreasing
            break

        context_path.append(best_neighbor)
        current_score = best_score
        frontier = get_neighbors(best_neighbor) - set(context_path)

    return context_path
```

---

## Generation Algorithm

1. Collect all context paths from the $`k`$ seed traversals and deduplicate overlapping nodes.
2. Flatten into passages and run a **reranker** (e.g., `bge-reranker-v2-m3`, Cohere Rerank, or Jina Reranker) to rank by query relevance.
3. Select top-$`N`$ passages within the token budget $`B`$.
4. Construct the generation prompt with reranked context and the original query.
5. Generate the final answer using a high-capability LLM (Sonnet/Opus-class).

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Graph Database | Neo4j (with GDS plugin) | Native graph traversal, vector index support (5.11+), Cypher for flexible queries |
| Embeddings | `bge-m3` or OpenAI `text-embedding-3-small` | Strong multilingual performance, 1024-dim, cost-effective |
| Context Scorer | Cross-encoder (`bge-reranker-v2-m3`) + LLM fallback | 100x faster and 1000x cheaper than LLM-only scoring |
| Reranker | `bge-reranker-v2-m3` or Cohere Rerank | Battle-tested, fast, high accuracy |
| Generation LLM | Sonnet/Opus-class model | Final answer quality is the priority |
| Orchestration | Python, LangGraph or custom async pipeline | State management for iterative traversal |

---

## Neo4j Graph Schema

```
(:Chunk {id, text, embedding, source, metadata})
(:Entity {id, name, type, embedding, description})
(:Concept {id, name, embedding, description})

(:Chunk)-[:CONTAINS_ENTITY]->(:Entity)
(:Entity)-[:RELATES_TO {type, weight}]->(:Entity)
(:Entity)-[:INSTANCE_OF]->(:Concept)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)
(:Chunk)-[:FROM_DOCUMENT]->(:Document)
```

Vector indices are created on `Chunk.embedding` and `Entity.embedding` for seed retrieval.

---

## Project Structure

```
CGT-RAG/
├── grace/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters (k, max_hops, thresholds, token budget)
│   ├── ingestion/
│   │   ├── chunker.py             # Document → chunks (semantic or fixed-window)
│   │   ├── extractor.py           # Chunk → entities + relationships (LLM structured output)
│   │   ├── embedder.py            # Embedding generation for chunks, entities, concepts
│   │   └── graph_builder.py       # Neo4j graph construction and index creation
│   ├── retrieval/
│   │   ├── seed_retriever.py      # Vector similarity top-k via Neo4j vector index
│   │   ├── scorer.py              # Hybrid context relevancy (cross-encoder + LLM fallback)
│   │   ├── traverser.py           # Inflection-point graph traversal engine
│   │   └── context_assembler.py   # Deduplicate and assemble context across seed paths
│   ├── generation/
│   │   ├── reranker.py            # Rerank retrieved contexts by query relevance
│   │   └── generator.py           # Final LLM answer generation
│   └── evaluation/
│       ├── metrics.py             # RAGAS metrics (context recall, faithfulness, relevance)
│       └── benchmarks.py          # HotpotQA, MuSiQue multi-hop QA runners
├── tests/
├── examples/
│   └── demo_query.py
├── pyproject.toml
└── README.md
```

---

## Implementation Roadmap

### Phase 1 — Graph Ingestion & Schema (Week 1–2)

- Implement document chunker with configurable strategy (semantic / fixed-window with overlap).
- Build LLM-based entity and relationship extractor using structured output.
- Implement entity deduplication (fuzzy string matching + embedding similarity).
- Create Neo4j graph builder with schema setup and vector index creation.
- Write integration tests against a local Neo4j instance.

### Phase 2 — Retrieval Engine (Week 2–4)

- Implement seed retriever using Neo4j vector index queries.
- Build the hybrid context relevancy scorer (cross-encoder primary, LLM fallback).
- Implement the inflection-point traversal algorithm with configurable max hops.
- Build context assembler with deduplication across seed paths.
- Parallelize the $`k`$ independent seed traversals using async execution.

### Phase 3 — Generation Pipeline (Week 4–5)

- Integrate reranker for post-retrieval context ordering.
- Build the generation agent with token-budget-aware prompt construction.
- Implement end-to-end query pipeline (query → seeds → traversal → rerank → generate).

### Phase 4 — Evaluation & Tuning (Week 5–6)

- Integrate RAGAS evaluation framework for automated metrics.
- Benchmark against vanilla vector RAG and Microsoft GraphRAG on multi-hop QA datasets (HotpotQA, MuSiQue).
- Tune hyperparameters: $`k`$, max hops $`H`$, cross-encoder confidence thresholds, token budget $`B`$.

---

## Open Design Decisions

1. **Cross-encoder vs. LLM for context scoring.** Start with the cross-encoder — it's 100x faster and 1000x cheaper. Add LLM scoring later for edge cases. The inflection-point logic works identically with either scorer.

2. **Greedy vs. beam traversal.** The current algorithm is greedy (follow the single best neighbor per hop). Beam search (keep top-$`b`$ paths) finds better context at the cost of more scoring calls. Start greedy, benchmark, then decide.

3. **Edge-type constraints.** Should traversal follow all edge types or only the Chunk → Entity → Entity → Chunk pattern? Constraining to this pattern reduces branching factor and keeps context grounded in source text.

4. **Concurrency model.** The $`k`$ seed traversals are independent and should run in parallel. Per-hop neighbor scoring should be batched for throughput.

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM scoring latency per hop | Query takes 30–60s | Use cross-encoder as primary scorer |
| Sparse entity extraction → disconnected graph | Traversal dead-ends at seed nodes | Add concept-level bridging edges; fall back to vector search for additional seeds |
| Noise accumulation over long traversal paths | Answer quality degrades | Inflection-point termination handles this naturally; enforce max hop limit ($`H=5`$) |
| Entity deduplication errors | Redundant or missing edges in graph | Use embedding similarity + fuzzy string matching; periodic graph cleanup |
| Cost at scale with LLM-only scoring | \$0.10–\$0.50 per query | Hybrid scoring brings cost to ~\$0.001–\$0.01 per query |
