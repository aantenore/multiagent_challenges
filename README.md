# 🪞 Project Antigravity: The Universal Adaptive Triage Engine

Project Antigravity is a high-performance, **zero-hardcode**, multi-agent classification framework. It is designed to be perfectly domain-agnostic, treating every industry as a set of abstract analytical roles.

## 🚀 Architectural Pillars

1.  **Pillar 1: Mathematical Filter**: High-speed behavioral frequency detection via Autocorrelation (ACF) and dynamic sliding windows.
2.  **Pillar 2: Memory Store**: Hierarchical RAG (Identity, Contextual, Global) with **Strict Stage Isolation**.
3.  **Pillar 3: Analytical Squads**: Specialized domain-expert agent swarms (Profile, Spatial, Temporal, Health/Behavioral).
4.  **Pillar 4: Global Orchestrator**: Final behavioral recovery assessment and economic risk arbitration.

## 🛠️ Key Features

- **N-Stage Continuous Streaming**: Cumulative indexing through infinite data streams with complete memory partitioning.
- **Domain Agnosticity**: Map any data field (e.g., 'HeartRate', 'TxAmount') to abstract roles in `manifest.json`.
- **Multi-LLM Native**: Switch seamlessly between Google Gemini and OpenAI models via LangChain integrations.
- **Deep Observability**: Tracing of internal swarm reasoning, token costs, and API calls via Langfuse.
- **Economic Decider**: Final decisions are weighted by the cost of False Negatives (FN) vs False Positives (FP).
- **Asymmetric Intelligence**: Optimized to minimize critical regret in high-stakes classification tasks.

## 📁 System Documentation

For deep dives into the framework, see the [docs/](file:///c:/PROJECTS/multiagent_challenge/docs/) directory:
- [Theory of Operation](file:///c:/PROJECTS/multiagent_challenge/docs/Theory_of_Operation.md)
- [Architecture Deep Dive](file:///c:/PROJECTS/multiagent_challenge/docs/Architecture_Deep_Dive.md)
- [Domain Adaptation Guide](file:///c:/PROJECTS/multiagent_challenge/docs/Domain_Adaptation_Guide.md)
- [Configuration Matrix](file:///c:/PROJECTS/multiagent_challenge/docs/Configuration_Matrix.md)

## 🏁 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (API Keys, Provider, Observability)
cp .env.example .env

# Run the pipeline for a specific stage
python main.py --manifest manifest.json --stage stage_1
```
