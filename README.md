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
N_0 = \operatorname{TopK}_{n \in \mathcal{V}} \left( \operatorname{sim}(q,\ \phi(n)) \right)
```

These seed nodes anchor subsequent graph traversal.

---

#### (b) Iterative Graph Expansion

Rather than issuing additional vector queries, GRACE expands context by traversing graph neighborhoods:

```math
C_{t+1} = C_t \cup \operatorname{Neighbors}(C_t,\ a_t)
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
\max_{C}\ \operatorname{AnswerQuality}(C)
```

subject to:

```math
|C| \cdot \bar{l} \leq B \qquad \text{(token budget)}
```

```math
T \leq H \qquad \text{(max traversal hops)}
```

```math
\frac{d}{dC}\operatorname{AnswerQuality}(C) \geq \epsilon \qquad \text{(diminishing returns threshold)}
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
| Reward $`r`$ | $`\operatorname{AnswerAccuracy}(C_T) - \lambda \cdot \operatorname{Cost}(T)`$ |

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
