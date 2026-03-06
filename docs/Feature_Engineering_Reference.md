# 📒 Project Antigravity: Volume V — Feature Engineering Reference

Project Antigravity treats "Time" as a dynamic dimension. This volume documents the math and logic behind our automated feature extraction engine.

## 1. Sliding Window Extraction
Pillar 3 agents don't see the entire history; they see **Incident Slices** cropped by Pillar 1.

### 1.1 Trend Signals
For any numeric column `X` in any role `R`, the system generates:
- **`mean`**: Average value in the window.
- **`std`**: Volatility (standard deviation).
- **`delta`**: The net change from start to end of the window.
- **`velocity`**: Rate of change across the window.

## 2. Spatial Intelligence
If columns are explicitly mapped to `spatial_coordinates` in the manifest `feature_mapping` block, the system pairs generic coordinates with operations like the Haversine formula and Shannon Entropy to dynamically synthesize:
- **Mobility Radius**: The standard deviation of distances from the centroid.
- **Location Entropy**: The diversity of visited locations.
*(These features are synthesized dynamically before the Z-Score is applied).*

## 3. The Composite Z-Score (Vitality Index)
Instead of monitoring single columns, Antigravity synthesizes holistic metrics.
- Computes a synthesized **Composite Z-Score** (e.g., `Vitality_Index` combining Physical Activity, Sleep Quality, and Location Entropy).
- The Autocorrelation and Z-Score are applied directly to this composite metric. This aggressively reduces noise and catches holistic behavioral collapses that single-column thresholds would miss.

## 4. Temporal Rhythms
Using Autocorrelation (ACF), the system calculates a `natural_frequency_score` for each entity, allowing the squads to distinguish between expected periodic behavior and true disruptions.
