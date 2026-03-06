# 📔 Project Antigravity: Volume VI — Configuration Matrix

Project Antigravity is a study in **Extensibility through Abstraction**. This volume provides a complete reference for every tunable parameter in `settings.py` and `.env`.

## 1. Pillar Activation (Core)
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `pillar_filter_enabled` | `True` | Activates Pillar 1 (Mathematical Filter). |
| `pillar_memory_enabled` | `True` | Activates Pillar 2 (Hierarchical Knowledge Memory). |
| `pillar_squads_enabled` | `True` | Activates Pillar 3 (Specialized Analytical Squads). |
| `pillar_orchestrator_enabled` | `True` | Activates Pillar 4 (Global Recovery Judge). |

## 2. Pillar 1: Mathematical Filtering
| Parameter | Default | Description |
| :--- | :--- | :--- |
| `filter_primary_col` | `"BehavioralIndex"` | Primary column for autocorrelation analysis. |
| `filter_secondary_col` | `"TrendIndex"` | Fallback column for behavioral triggers. |
| `filter_upper_threshold` | `0.85` | Confidence above which Pillar 1 can short-circuit the pipeline if the case is stable. |
| `filter_lower_threshold` | `0.15` | Confidence below which Pillar 1 forces escalation despite its own prediction. |

## 3. Pillar 3: Swarm Scaling
- **`squad_base_agents`** (Default: 1): Minimum agents per squad.
- **`squad_scaling_factor`** (Default: 0.001): Rate at which the swarm grows based on data complexity.
- **`swarm_max_agents`** (Default: 5): Hard cap on agents per squad.
- **`L1_SKIP_L2_CONFIDENCE_THRESHOLD`** (Default: 0.98): Confidence threshold required by L1 to skip the L2 Orchestrator.

---
*Volume VII: Deep Troubleshooting & Observability coming next.*
