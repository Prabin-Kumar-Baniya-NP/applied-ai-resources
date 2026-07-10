# Applied AI Learning Roadmap (2026)

A categorized list of topics to learn as a software engineer moving from beginner to advanced applied AI, focused on what matters in production-grade applications. Compiled from web research on the current (2026) landscape. Topics only — no explanations.

Legend: 🟢 Beginner · 🟡 Intermediate · 🔴 Advanced · ✅ Already covered in this repo

---

## 1. LLM Fundamentals & Working with Model APIs
- 🟢 How LLMs work at a high level (tokens, embeddings, attention, sampling)
- 🟢 Tokenization and token counting
- 🟢 Model parameters (temperature, top-p, max tokens, stop sequences)
- 🟢 Chat completions vs. responses APIs (OpenAI, Anthropic, Gemini)
- 🟢 Streaming responses (SSE, chunked output)
- 🟢 Structured outputs (JSON mode, schema-constrained generation)
- 🟢 Function calling / tool use
- 🟡 Multimodal inputs (vision, audio, documents/PDFs)
- 🟡 Model selection and routing (frontier vs. small models, cost/latency/quality trade-offs)
- 🟡 Reasoning models and extended thinking (test-time compute, thinking budgets)
- 🟡 Prompt caching (provider-side)
- 🟡 Batch APIs for offline workloads
- 🔴 Test-time compute scaling strategies (best-of-N, self-consistency, verifier-guided)

## 2. Prompt Engineering & Context Engineering
- 🟢 Prompt structure (system prompts, few-shot examples, delimiters)
- 🟢 Chain-of-thought and step-by-step prompting
- 🟡 Prompt versioning and management in production
- 🟡 Context window management strategies
- ✅ 🟡 Token budgeting
- ✅ 🟡 History compression and folding
- ✅ 🟡 Semantic caching
- 🟡 Context pruning / relevance filtering
- 🟡 Dynamic prompt assembly (templates, variables, per-tenant customization)
- 🔴 Context engineering for agents (tool results, scratchpads, sub-agent contexts)
- 🔴 Prompt optimization frameworks (DSPy, automatic prompt optimization)

## 3. Retrieval-Augmented Generation (RAG)
- 🟢 Embeddings and semantic similarity
- 🟢 Chunking strategies (fixed, recursive, semantic, document-aware)
- 🟢 Vector databases (pgvector, Qdrant, Pinecone, Weaviate, Milvus)
- 🟢 Naive RAG pipeline (ingest → embed → retrieve → generate)
- 🟡 Hybrid search (BM25 + dense vectors, reciprocal rank fusion)
- 🟡 Reranking (cross-encoders, Cohere Rerank, LLM rerankers)
- 🟡 Query transformation (rewriting, decomposition, HyDE, multi-query)
- ✅ 🟡 Advanced RAG patterns (parent-document, sentence-window, auto-merging)
- ✅ 🟡 RAG evaluation (RAG triad, retrieval metrics — precision/recall/MRR/NDCG, RAGAS)
- ✅ 🔴 GraphRAG and knowledge-graph-backed retrieval
- 🟡 Metadata filtering and multi-tenancy in retrieval
- 🟡 Document parsing and ETL for unstructured data (OCR, layout-aware parsing, tables)
- 🔴 Agentic RAG (retrieval as a tool, iterative retrieval, self-correcting RAG)
- 🔴 Multimodal RAG (image/table/chart retrieval)
- 🔴 Embedding model selection, fine-tuning embeddings, Matryoshka embeddings
- 🔴 Freshness, incremental indexing, and index maintenance at scale

## 4. AI Agents & Agentic Systems
- 🟢 Agent fundamentals (the agent loop: reason → act → observe)
- 🟢 Tool design for agents (schemas, descriptions, error handling)
- 🟡 Workflow patterns vs. autonomous agents (prompt chaining, routing, parallelization, orchestrator-worker, evaluator-optimizer)
- 🟡 Agent frameworks (LangGraph, Claude Agent SDK, OpenAI Agents SDK, CrewAI, Pydantic AI, LlamaIndex Workflows, Microsoft Agent Framework)
- 🟡 Planning and task decomposition
- 🟡 Human-in-the-loop patterns (approvals, interrupts, escalation)
- 🟡 Agent state management and checkpointing (durable execution, resumability)
- 🔴 Multi-agent architectures (supervisor, hierarchical, swarm, handoffs)
- 🔴 Agent memory: short-term, episodic, semantic, procedural (Letta/MemGPT, Mem0, LangMem, Zep)
- 🔴 Long-horizon agents (running for hours, self-verification, recovery from failure)
- 🔴 Computer-use and browser-use agents
- 🔴 Coding agents and agentic software development (Claude Code, Codex, agent teams)
- 🔴 Sandboxing and isolation for agent code execution

## 5. Agent Interoperability & Protocols
- 🟡 Model Context Protocol (MCP): servers, clients, tools, resources, prompts
- 🟡 Building and deploying MCP servers (auth, remote MCP, security)
- 🔴 Agent-to-Agent protocol (A2A) and agent cards
- 🔴 The emerging protocol stack (MCP vs. A2A vs. ACP/ANP, WebMCP)
- 🔴 Agent discovery, identity, and permissioning across systems

## 6. Evaluation & Testing (Evals)
- 🟢 Why evals matter; eval-driven development
- 🟢 Building golden datasets and test cases
- 🟡 Deterministic/code-based metrics vs. model-graded evals
- 🟡 LLM-as-judge (pairwise, rubric-based, G-Eval; judge calibration and bias)
- 🟡 Offline evals vs. online evals (A/B tests, canary releases)
- 🟡 Eval tooling (Braintrust, LangSmith, Langfuse, DeepEval, Promptfoo, Phoenix/Arize)
- 🔴 Agent evals (trajectory/trace-based evaluation, tool-call correctness, task completion)
- 🔴 Multi-turn conversation evaluation
- 🔴 Regression testing for prompts and model upgrades
- 🔴 Benchmark literacy (what MMLU/SWE-bench/etc. do and don't tell you)

## 7. LLMOps, Observability & Production Operations
- 🟢 Logging LLM requests/responses with metadata
- 🟡 Tracing (spans across chains/agents; OpenTelemetry GenAI conventions)
- 🟡 Observability platforms (Langfuse, LangSmith, LangWatch, Helicone, Datadog LLM Observability)
- 🟡 Cost tracking, token budgets, and alerting
- 🟡 Latency optimization at the application layer (streaming, parallel calls, speculative UI)
- 🟡 Rate limits, retries, timeouts, fallbacks, and multi-provider failover
- 🟡 LLM gateways / AI gateways (LiteLLM, OpenRouter, portkey-style routing)
- 🟡 Prompt/config release management (versioning, rollbacks, feature flags)
- 🔴 Drift detection and quality monitoring in production
- 🔴 Feedback loops (user feedback capture → eval sets → improvement)
- 🔴 CI/CD for AI systems (eval gates in pipelines)
- 🔴 Incident response and debugging for AI features

## 8. Safety, Security & Guardrails
- 🟢 OWASP Top 10 for LLM applications
- 🟡 Prompt injection (direct and indirect) and defenses
- 🟡 Jailbreak patterns and mitigation
- 🟡 Input/output guardrails (Llama Guard, LLM-Guard, NeMo Guardrails, moderation APIs)
- 🟡 PII detection, redaction, and data privacy in prompts
- 🟡 Least-privilege tool design and human approval for high-risk actions
- 🔴 Agent security (confused deputy, tool poisoning, MCP supply-chain risks)
- 🔴 Red teaming and adversarial testing (automated red teaming, AgentHarm-style benchmarks)
- 🔴 Trust boundaries in agentic pipelines
- 🔴 Hallucination detection and grounding enforcement (citations, faithfulness checks)

## 9. Model Customization & Fine-Tuning
- 🟡 When to fine-tune vs. RAG vs. prompt engineering (decision framework)
- 🟡 Dataset preparation and curation for fine-tuning
- 🟡 Parameter-efficient fine-tuning: LoRA, QLoRA, PEFT
- 🟡 Hosted fine-tuning APIs vs. self-managed training
- 🔴 Preference optimization (DPO, RLHF, GRPO/RLVR — conceptual literacy)
- 🔴 Distillation (teaching small models from large-model outputs)
- 🔴 Small language models (SLMs) as production specialists
- 🔴 Continued pretraining / domain adaptation
- 🔴 Fine-tuning evaluation and avoiding catastrophic forgetting

## 10. Inference, Serving & Cost Optimization
- 🟡 Self-hosting open-weight models (Llama, Qwen, Mistral, DeepSeek; Ollama for local dev)
- 🟡 Inference servers (vLLM, SGLang, TensorRT-LLM, TGI)
- 🟡 Quantization (INT8/INT4, AWQ, GPTQ, FP8) and quality trade-offs
- 🔴 Continuous batching, PagedAttention, KV-cache management
- 🔴 Speculative decoding
- 🔴 GPU sizing, throughput vs. latency tuning (TTFT, tokens/sec)
- 🔴 Serving economics (cost per task, model cascades/routing to cut spend)
- 🔴 Edge and on-device inference

## 11. Multimodal & Voice AI
- 🟡 Vision-language models in applications (document understanding, screenshots, charts)
- 🟡 Speech-to-text (Whisper and successors) and text-to-speech
- 🟡 Image generation in product features (and prompt-to-image control)
- 🔴 Real-time voice agents (speech-to-speech models, sub-second latency, interruption handling)
- 🔴 Realtime APIs and telephony/WebRTC integration (LiveKit, Pipecat, SIP)
- 🔴 Multimodal agents (voice + vision + tools in one session)
- 🔴 Video understanding and generation basics

## 12. AI Product & System Design
- 🟢 UX patterns for AI features (streaming UI, citations, confidence, undo)
- 🟡 Designing for non-determinism (graceful degradation, fallbacks)
- 🟡 Build vs. buy decisions (APIs vs. open models vs. platforms)
- 🟡 Architecture patterns for AI features in existing systems (async queues, event-driven AI)
- 🟡 Latency/cost/quality triangle and product trade-offs
- 🔴 Personalization and per-user memory in products
- 🔴 Data flywheels (capturing usage data to improve the system)
- 🔴 Scaling AI features (caching layers, precomputation, tiered model usage)
- 🔴 AI governance, compliance, and audit trails (EU AI Act awareness, usage policies)

## 13. Supporting Foundations (learn as needed)
- 🟢 Python for AI work (async, typing, Pydantic)
- 🟢 API design and webhooks for AI services
- 🟡 Data pipelines for AI (ingestion, ETL for unstructured data, orchestration)
- 🟡 Postgres + pgvector as a pragmatic default stack
- 🟡 Queues and background jobs for long-running AI tasks
- 🟡 Docker/deployment basics for model-adjacent services
- 🔴 Classical ML literacy (when a classifier beats an LLM; scikit-learn level)
- 🔴 GPU/accelerator fundamentals (VRAM math, what fits where)

---

## Suggested learning order

1. **Foundations first (1–2):** LLM APIs, tool use, structured outputs, prompt/context engineering.
2. **The core production pattern (3):** RAG end-to-end, including evaluation — the most in-demand production skill.
3. **Agents (4–5):** workflow patterns before autonomous agents; MCP early since it's now the de facto tool-integration standard.
4. **Make it production-grade (6–8):** evals, observability, and security — this is what separates demos from products.
5. **Specialize (9–11):** fine-tuning, inference/serving, voice/multimodal, based on your job's needs.
6. **Zoom out (12):** AI product and system design ties everything together.

## Sources

- [dataskew.io — AI Engineer Roadmap 2026](https://dataskew.io/roadmaps/ai-engineering/)
- [Codebasics — Software Engineer to AI Engineer Roadmap](https://codebasics.io/blog/software-engineer-to-ai-engineer-the-most-effective-path-with-roadmap)
- [roadmap.sh — AI Engineer Roadmap](https://roadmap.sh/ai-engineer)
- [KDnuggets — How to Become an AI Engineer in 2026](https://www.kdnuggets.com/how-to-become-an-ai-engineer-in-2026-a-self-study-roadmap)
- [MachineLearningMastery — 7 Agentic AI Trends to Watch in 2026](https://machinelearningmastery.com/7-agentic-ai-trends-to-watch-in-2026/)
- [Firecrawl — Top 13 Agentic AI Trends](https://www.firecrawl.dev/blog/agentic-ai-trends)
- [FutureAGI — Multi-Agent AI Systems in 2026](https://futureagi.com/blog/multi-agent-systems-2025/)
- [TrueFoundry — Best LLMOps Tools in 2026](https://www.truefoundry.com/blog/llmops-tools)
- [OpenObserve — LLM Monitoring Best Practices](https://openobserve.ai/blog/llm-monitoring-best-practices/)
- [Braintrust — Best LLMOps platforms](https://www.braintrust.dev/articles/best-llmops-platforms-2025)
- [Digital Applied — AI Agent Protocol Ecosystem Map 2026 (MCP, A2A, ACP, UCP)](https://www.digitalapplied.com/blog/ai-agent-protocol-ecosystem-map-2026-mcp-a2a-acp-ucp)
- [arXiv — Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1)
- [Aembit — OWASP Top 10 for LLM Applications](https://aembit.io/blog/owasp-top-10-llm-risks-explained/)
- [Kili Technology — LLM Red Teaming in 2026](https://kili-technology.com/blog/llm-red-teaming-in-2026)
- [Confident AI — LLM Agent Evaluation Metrics in 2026](https://www.confident-ai.com/blog/llm-agent-evaluation-complete-guide)
- [Hakia — LLM Inference Optimization Techniques 2026](https://www.hakia.com/tech-insights/llm-inference-optimization/)
- [Zylos Research — LLM Inference Optimization and Quantization 2026](https://zylos.ai/research/2026-01-15-llm-inference-optimization/)
- [BigDataBoutique — Fine-Tuning LLMs in 2026: When RAG Isn't Enough](https://bigdataboutique.com/blog/fine-tuning-llms-when-rag-isnt-enough)
- [FutureAGI — LLM Fine-Tuning Guide: LoRA, QLoRA, DPO, GRPO, RLHF](https://futureagi.com/blog/llm-fine-tuning-guide-2025/)
- [MachineLearningMastery — 6 Best AI Agent Memory Frameworks 2026](https://machinelearningmastery.com/the-6-best-ai-agent-memory-frameworks-you-should-try-in-2026/)
- [Vectorize — Best AI Agent Memory Systems in 2026](https://vectorize.io/articles/best-ai-agent-memory-systems)
- [Zylos Research — Voice AI and Speech Technology 2026](https://zylos.ai/research/2026-01-25-voice-ai-speech-technology)
- [Famulor — State of Voice AI 2026](https://www.famulor.io/blog/state-of-voice-ai-2026-voice-ai-moves-to-production)
- [Hugging Face — AI Trends 2026: Test-Time Reasoning and Reflective Agents](https://huggingface.co/blog/aufklarer/ai-trends-2026-test-time-reasoning-reflective-agen)
- [Taskade — AI Reasoning Models Explained: Test-Time Compute](https://www.taskade.com/blog/reasoning-models)
- [Anthropic — 2026 Agentic Coding Trends Report](https://resources.anthropic.com/2026-agentic-coding-trends-report)
- [Claude — Eight trends defining how software gets built in 2026](https://claude.com/blog/eight-trends-defining-how-software-gets-built-in-2026)
- [The Claude Codex — 10 AI coding trends in 2026](https://claude-codex.fr/en/future/trends-2026/)
