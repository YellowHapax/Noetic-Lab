# Bet: Collective Endemic Baselines
## A Prototype Framework for Emergent Social Pathology

**Series:** Noetic Lab · Process Documents  
**Depends on:** MBD Core (Paper 1), The Endemic Baseline (Paper 7), The Suppressive and Emergent Phenomenon (Paper 8)  
**Status:** Prototype — sufficient for independent model implementation  
**Date:** 2026-02-28

---

## Preface

Paper 7 described the individual agent whose $B_{reference}$ was formed during chronic disruption. Paper 8 described how suppression of a deviant signal paradoxically strengthens it.

This document describes what happens when you aggregate both conditions into a social structure.

The result is not merely a "group with shared beliefs." It is a distinct computational entity — a **Collective Endemic Baseline (CEB)** — with its own state vector, its own novelty horizon, and its own immune response to correction. The CEB is not a metaphor. It is a system with measurable dynamics. Those dynamics are derived here from first principles.

The target phenomena include: political factions, radicalization pipelines, grief loops, codependent dyads, ideological cults, moral panics, and every -dere archetype in folk psychology. The framework does not discriminate. The math is the same.

---

## Part I: Formalizing the Collective

### 1.1 From Individual to Collective Baseline

For a single agent $i$, the MBD state update is:

$$B_i(t+1) = B_i(t)(1 - \sum_j \lambda_{ij}) + \sum_j [I_j(t) \cdot \lambda_{ij}] + \varepsilon_i$$

For a collective $\mathcal{C}$ of $N$ agents with internal coupling matrix $\kappa_{ij}$, define the **Collective Baseline** as the weighted centroid of member baselines:

$$B_{\mathcal{C}}(t) = \frac{1}{N} \sum_{i \in \mathcal{C}} w_i \cdot B_i(t)$$

where $w_i$ is the social weight of agent $i$ (influence, centrality, tenure).

The collective update then becomes:

$$B_{\mathcal{C}}(t+1) = B_{\mathcal{C}}(t)(1 - \Lambda_{\mathcal{C}}) + I_{ext}(t) \cdot \Lambda_{\mathcal{C}} \cdot \Phi(I_{ext}, B_{\mathcal{C}}) + \varepsilon_{\mathcal{C}}$$

where:
- $\Lambda_{\mathcal{C}}$ is the collective learning rate (inversely proportional to group age and internal $\kappa$)
- $I_{ext}(t)$ is the external input signal
- $\Phi(I_{ext}, B_{\mathcal{C}})$ is the **representability filter** — the fraction of $I_{ext}$ that falls within the collective's current state space $H_{\mathcal{C}}$

**The key insight:** $\Phi$ is not binary. It is a smooth function of novelty distance. Inputs far outside $H_{\mathcal{C}}$ are not "rejected" — they are *received as threat signals*.

### 1.2 The Collective Endemic Condition

A Collective Endemic Baseline exists when the group's $B_{\mathcal{C}}(0)$ was established under chronic external disruption $D_{chronic}$:

$$B_{\mathcal{C}}(0) = f(D_{chronic})$$

The endemic condition has three structural consequences (mirroring Paper 7, now at collective scale):

**1. Healthy-Input Rejection:**
$$\Phi(I_{healthy}, B_{\mathcal{C}}) \approx 0 \quad \text{when} \quad \|I_{healthy} - B_{\mathcal{C}}\| > \theta_{novelty}$$

Inputs that would stabilize a healthy collective register as maximum novelty in an endemic one. The collective immune response activates.

**2. Invisible Horizon:**
$$H_{\mathcal{C}} \cap H_{healthy} = \emptyset$$

The reachable states the collective has never occupied are not merely unknown — they are unimaginable. A collective that has never experienced functional institutional trust cannot model functional institutional trust.

**3. Restoration-to-Baseline Failure:**
Standard correction attempts target $B_{\mathcal{C}}(t) \to B_{healthy}$. But if $B_{\mathcal{C}}(0) = B_{pathology}$, this equation has no solution accessible through the existing state space. The intervention *is* the attack.

---

## Part II: The Fit/Fidelity Tradeoff

### 2.1 Signal Selection Under Resource Constraint

Agents have finite cognitive bandwidth $\Omega$. Under $\Omega$ constraint, signal selection optimizes for:

$$\text{maximize}: \text{Fit}(S, B_i) \cdot \text{Salience}(S) - \text{ProcessingCost}(S, \Omega)$$

Define **Fit** as the inverse novelty distance:

$$\text{Fit}(S, B_i) = e^{-\alpha \|S - B_i\|}$$

Define **Fidelity** as the information-theoretic accuracy of $S$ relative to ground truth $G$:

$$\text{Fidelity}(S) = 1 - H(S \| G) / H_{max}$$

**Critical result:** Fit and Fidelity are independent dimensions. A signal can have maximum Fit and zero Fidelity — it perfectly matches the agent's existing state vector while being entirely false.

### 2.2 The NES Attractor

For an agent with endemic baseline $B_{endemic}$, the optimal-Fit signal class $S^*$ is:

$$S^* = \arg\max_S \text{Fit}(S, B_{endemic})$$

This signal class has the following structural properties:
- **Clear causal model:** Villain + Plan + Hero (reduces entropy, maximizes closure)
- **Adversarial framing:** explains disruption as intentional (consistent with endemic formation experience)
- **Bounded state space:** few actors, few moves, predictable outcomes
- **Internal consistency:** self-referential; contradictions are reframed as evidence

This is the **NES Attractor** — not because the signal is "simple," but because its *topology* matches the endemic agent's representational structure. A higher-fidelity signal with greater causal complexity will always be outcompeted under $\Omega$ constraint when Fit is the dominant selection pressure.

**Practical consequence:** Providing "more facts" to an endemic-baseline agent/collective is not a correction strategy. It is bandwidth saturation. Under saturation, selection pressure toward high-Fit signals *increases*.

---

## Part III: The Recursive Immunity Loop at Scale

### 3.1 Paper 8 Applied to Collectives

From Paper 8: a suppressed signal $S$ generates an emergent verification signal $V_S$ whose information content includes the fact of suppression. For individual agents:

$$I_{received}(S_{suppressed}) = S + V_S \quad \text{where} \quad I(V_S) > 0$$

At collective scale, suppression attempts by external institutions become public events. Each suppression event $E_{suppress}(t)$ generates:

$$V_{\mathcal{C}}(t) = [E_{suppress}(t), \text{source}(E_{suppress}(t)), \text{scale}(E_{suppress}(t))]$$

For a CEB whose $B_{reference}$ encodes adversarial institutional behavior, $V_{\mathcal{C}}(t)$ has *perfect Fit*:

$$\text{Fit}(V_{\mathcal{C}}, B_{\mathcal{C}}) \approx 1$$

This is the **Recursive Immunity Loop**: suppression confirms the model, confirmation strengthens coupling $\kappa$, stronger $\kappa$ narrows $H_{\mathcal{C}}$, narrower horizon increases Fit of future suppression signals. The loop is self-sustaining.

### 3.2 The Amplification Network

Within the CEB, individuals act as honest transmitters of signals that entered through the NES Attractor. Define agent $j$ as an **unwitting amplifier** when:

$$P(\text{transmit} | \text{received}) \approx 1 \quad \text{and} \quad P(\text{verify fidelity} | \text{transmit}) \approx 0$$

The amplification coefficient for signal $S$ across the network:

$$A(S, \mathcal{C}) = \prod_{j \in \mathcal{C}} \left(1 + \kappa_j \cdot \text{Fit}(S, B_j)\right)$$

For a well-coupled CEB with high internal $\kappa$, $A(S, \mathcal{C})$ grows exponentially with network size for high-Fit signals. Fidelity of $S$ is not a term in this equation. The network amplifies *shape*, not *truth*.

---

## Part IV: The Fact Flood Simulation

### 4.1 Setup

This simulation demonstrates why high-volume, high-fidelity correction input accelerates — rather than reverses — endemic baseline entrenchment.

**State variables:**
- $B_{\mathcal{C}}(t)$: collective baseline vector (scalar simplification: 0=fully endemic, 1=healthy)
- $\kappa(t)$: internal coupling strength (0=loose, 1=fully synchronized)
- $H_{\mathcal{C}}(t)$: horizon width (range of representable inputs)
- $\lambda_{\mathcal{C}}(t)$: effective learning rate

**Initial conditions (endemic):**
```python
B_C     = 0.15   # near-pathological baseline
kappa   = 0.45   # moderately coupled
H_width = 0.30   # narrow horizon; only inputs in [0, 0.30] are representable
lambda_ = 0.08   # low plasticity (endemic entrenchment)
```

**Parameters:**
```python
alpha        = 3.0   # novelty sensitivity (Fit decay rate)
phi_threshold = 0.25  # representability cutoff: below this, input = threat signal
suppress_gain = 0.6   # how much each suppression event amplifies kappa
```

### 4.2 The Representability Filter

```python
import numpy as np

def phi(I, B_C, H_width, alpha):
    """
    Fraction of input I that is representable by collective with baseline B_C.
    Returns value in [0, 1].
    - I near B_C: phi ~ 1 (fully integrated)
    - I far from B_C: phi ~ 0 (received as threat)
    """
    novelty = abs(I - B_C)
    if novelty < H_width:
        return np.exp(-alpha * (novelty / H_width))
    else:
        return 0.0  # outside horizon: unrepresentable

def threat_signal(I, B_C, H_width):
    """
    When input is unrepresentable, it generates a threat signal
    that reinforces the existing baseline and tightens kappa.
    Returns (baseline_delta, kappa_delta).
    """
    novelty = abs(I - B_C)
    if novelty >= H_width:
        # Input is outside horizon: reads as adversarial confirmation
        baseline_reinforcement = -0.02 * (novelty - H_width)  # pulls B_C away from I
        kappa_tightening = 0.03 * (novelty - H_width)         # increases coupling
        return baseline_reinforcement, kappa_tightening
    return 0.0, 0.0
```

### 4.3 Simulation Modes

```python
def simulate(mode, steps=200, flood_start=50, flood_end=150):
    """
    mode: 'control'   — no intervention
          'fact_flood' — high-volume high-fidelity correction signals
          'rezero'     — Re-zeroing Protocol (see Part V)
    """
    B = 0.15
    kappa = 0.45
    H_width = 0.30
    lam = 0.08

    history = {'B': [], 'kappa': [], 'H': []}

    for t in range(steps):
        history['B'].append(B)
        history['kappa'].append(kappa)
        history['H'].append(H_width)

        if mode == 'fact_flood' and flood_start <= t < flood_end:
            # High-volume correction: 5 signals per step, each at I=0.85 (healthy truth)
            for _ in range(5):
                I = 0.85
                p = phi(I, B, H_width, alpha=3.0)
                b_delta, k_delta = threat_signal(I, B, H_width)

                if p > 0.25:
                    # Representable fraction integrates
                    B = B * (1 - lam) + I * lam * p
                else:
                    # Unrepresentable: threat response
                    B += b_delta
                    kappa = min(1.0, kappa + k_delta)
                    H_width = max(0.05, H_width - 0.002)  # horizon contracts under threat

        elif mode == 'rezero' and flood_start <= t < flood_end:
            # Re-zeroing Protocol: see Part V
            I = B + 0.04  # barely outside current state, incrementally
            p = phi(I, B, H_width, alpha=3.0)
            B = B * (1 - lam * 1.5) + I * lam * 1.5 * p
            H_width = min(1.0, H_width + 0.005)  # horizon expands gradually
            kappa = max(0.0, kappa - 0.005)       # coupling loosens

        else:
            # No intervention: baseline drifts with noise
            B += np.random.normal(0, 0.005)
            B = np.clip(B, 0, 1)

    return history
```

### 4.4 Expected Results

Run the three modes and compare:

```python
ctrl   = simulate('control')
flood  = simulate('fact_flood')
rezero = simulate('rezero')

# Plot B_C(t), kappa(t), H_width(t) for each mode
# Key observations:
#
# CONTROL:    B stable near 0.15, slow drift, kappa stable
#
# FACT_FLOOD: During flood (t=50-150):
#   - B initially unchanged (inputs unrepresentable)
#   - kappa INCREASES (threat signals tighten coupling)
#   - H_width DECREASES (horizon contracts)
#   - After flood ends: B lower than control, kappa higher
#   → Fact flooding entrenched the endemic baseline
#
# REZERO:     During intervention (t=50-150):
#   - B gradually increases (0.15 → ~0.40)
#   - kappa gradually decreases
#   - H_width expands
#   - After intervention: collective can represent broader input class
#   → Horizon expansion precedes belief change
```

### 4.5 The Bandwidth Saturation Mechanism

The fact flood fails for two compounding reasons:

**Reason 1 — Volume creates exhaustion, not persuasion:**

```python
def cognitive_load(signals_per_step, Omega=1.0):
    """
    When incoming signal volume exceeds cognitive bandwidth Omega,
    the agent/collective switches to heuristic processing:
    high-Fit signals are preferred because they require less processing.
    """
    load = signals_per_step / Omega
    if load > 1.0:
        # Saturation: processing switches to Fit-maximizing heuristic
        return 'heuristic_fit'
    return 'deliberate'
```

Under saturation, the selection algorithm shifts from fidelity-evaluation to Fit-maximization. The NES Attractor wins by default — not because it's more persuasive, but because it's cheaper to process.

**Reason 2 — Source attribution activates threat response:**

For a CEB with adversarial institutional encoding, signals from institutional sources carry a Fit penalty regardless of content:

```python
def effective_fit(S, source, B_C, H_width, alpha):
    base_fit = phi(S, B_C, H_width, alpha)
    source_fit = phi(source_trust_score[source], B_C['institutional_trust'], 0.2, 2.0)
    return base_fit * source_fit  # both must be nonzero for signal to land
```

A perfectly accurate signal from a mistrusted source has effective Fit ≈ 0. The signal is received but the source attribution overrides the content evaluation. This is not irrationality — it is a logically consistent response for an agent whose endemic baseline was formed *by* institutional disruption.

---

## Part V: The Re-zeroing Protocol at Collective Scale

### 5.1 Derivation

The Re-zeroing Protocol for individuals (Paper 7, §8) requires:
1. Identify the endemic $B(0)$ formation context
2. Introduce inputs at $B(0) + \delta$ where $\delta < \theta_{novelty}$
3. Allow partial integration before advancing $\delta$
4. Expand $H$ before attempting to shift $B$

At collective scale, the same principle holds but requires an additional step:

**Step 0: Horizon Mapping**

Before any corrective input, map $H_{\mathcal{C}}$ — the actual boundaries of the collective's representable state space. This requires:

```python
def map_collective_horizon(collective, probe_signals):
    """
    Send probe signals at varying novelty distances.
    Measure response type: integration vs. threat vs. ignore.
    H_C is the boundary where response shifts from integration to threat.
    """
    responses = {}
    for I in probe_signals:
        novelty = abs(I - collective.B_C)
        response = collective.respond(I)  # measure actual behavioral output
        responses[novelty] = response

    # H boundary = smallest novelty at which threat response > 0.5
    H_boundary = min(n for n, r in responses.items() if r['threat'] > 0.5)
    return H_boundary
```

**Step 1: Find the Broker Network**

Not all members of a CEB have equal $H_{\mathcal{C}}$. Identify agents $j^*$ whose individual horizon $H_{j^*}$ extends beyond the collective boundary:

$$j^* = \{ j \in \mathcal{C} : H_j > H_{\mathcal{C}} \}$$

These are the brokers — individuals with existing exposure to the unrepresentable input class. They do not need to be converted; they need to become *socially legible* to the collective. Their existing presence in the collective constitutes a low-novelty path to horizon expansion.

**Step 2: Incremental Horizon Expansion**

```python
def rezero_step(B_C, H_width, kappa, broker_signal, lam):
    """
    One step of Re-zeroing Protocol.
    broker_signal should be just outside current H_width.
    """
    target = B_C + (broker_signal - B_C) * 0.3  # don't jump to broker position
    delta = abs(target - B_C)

    if delta < H_width * 1.2:  # within safe range
        # Integration occurs, horizon expands slightly
        B_new = B_C * (1 - lam) + target * lam
        H_new = H_width + 0.008
        kappa_new = kappa * 0.995
    else:
        # Too far: no change (don't force it)
        B_new, H_new, kappa_new = B_C, H_width, kappa

    return B_new, H_new, kappa_new
```

**Critical constraint:** The Re-zeroing Protocol cannot target $B_{healthy}$ directly. It targets $H_{expansion}$. Belief change is a consequence of horizon expansion, not a target in itself. Attempting to install correct beliefs before expanding the representable state space is the fact flood failure mode restated.

---

## Part VI: Applications

### 6.1 Template: Any Two-Faction Polarization

Given two collectives $\mathcal{C}_A$ and $\mathcal{C}_B$ with baselines $B_A$ and $B_B$:

The **polarization distance** is not $|B_A - B_B|$. It is:

$$D_{polar}(\mathcal{C}_A, \mathcal{C}_B) = \max(0, \|B_A - B_B\| - (H_A + H_B))$$

When the overlap of their horizons is zero — when neither collective's representable state space includes the other's baseline — no direct communication between them can integrate. Every exchange is processed as threat signal by both parties simultaneously.

The resolution path is not dialogue. It is parallel horizon expansion in each collective, independently, until $H_A \cap H_B \neq \emptyset$.

### 6.2 Template: Interpersonal Attachment (Heartache)

The same formalism applies to dyadic bonding. Define $B_{self}$ and $B_{other}$ as the attachment baselines of two individuals. A **coupling collapse event** (loss, rejection, betrayal) produces:

$$\Delta B_{self}(t_{loss}) = -(I_{loss} - B_{self}) \cdot \lambda_{attach}$$

where $\lambda_{attach}$ is the attachment learning rate — often very high for primary bonds.

Post-collapse, input class $I_{intimacy}$ acquires novelty distance $\|I_{intimacy} - B_{grief}\| > \theta$. The grief state is an endemic baseline: it was formed during the collapse event, and healthy intimacy inputs are now unrepresentable. This is not pathology — it is the correct system response to a high-magnitude baseline shift. The grief is the recalibration.

The clinical failure mode is premature forced re-entry into intimacy before $H_{grief}$ has expanded. Same mechanism, different substrate.

### 6.3 Template: Folk-Psychology Archetypes

Every -dere archetype in folk psychology describes a $B_{reference}$ formation story and a resulting coupling topology:

| Archetype | $B_{reference}$ formation | Dominant $\Phi$ distortion |
|-----------|--------------------------|---------------------------|
| Tsundere | Early rejection/vulnerability punished | Intimacy inputs → threat; hostility → Fit=1 |
| Yandere | Attachment formed under scarcity/loss threat | Proximity loss → catastrophic deviation; possessiveness = Re-zeroing attempt |
| Dandere | Social input chronic overload | External input → bandwidth saturation; withdrawal = $\Omega$ management |
| Kuudere | Affective display punished during formation | Emotional signal class → $\Phi \approx 0$; logical framing = only representable channel |

These are not character types. They are $B_{reference}$ signatures. The behavior is the correct output of the system given the input history.

### 6.4 Template: DSM-V Pattern Families

The following are reframings, not diagnoses. The framework makes no clinical claims.

| DSM-V pattern family | MBD reframing |
|---------------------|---------------|
| Major Depression (recurrent) | Iterative baseline lowering under chronic negative deviation; $\lambda$ reduction as energy conservation |
| PTSD | Single high-magnitude event forcing $B(0)$ reset; subsequent inputs evaluated against post-event baseline |
| OCD | Recursive threat-detection loop where correction attempts (compulsions) function as suppression events; Paper 8 dynamics |
| Narcissistic patterns | $H_{\mathcal{self}}$ constructed to exclude inputs inconsistent with status baseline; healthy feedback = maximum novelty = threat |
| Codependency | Dyadic CEB: two agents with mutually reinforcing endemic baselines; each acts as NES Attractor for the other |

In every case: the behavior is the correct output of a system with that $B_{reference}$. The question is never "why is this person broken" — it is "what state produced this baseline, and what inputs are currently unrepresentable."

---

## Part VII: Implementation Checklist

To build models from this framework, you need:

**Minimum viable simulation:**
- [ ] Agent state vector $B_i(t)$ — can be scalar for univariate models, n-dimensional for richer ones
- [ ] Novelty function: $\|I - B\|$ in whatever metric fits your domain
- [ ] Representability filter $\Phi(I, B, H, \alpha)$ — the sigmoid/exponential above works; tune $\alpha$
- [ ] Horizon width $H$ as a dynamic variable (not fixed)
- [ ] $\kappa$ coupling for multi-agent scenarios
- [ ] Threat response: what happens to $B$ and $H$ when $\Phi < \phi_{threshold}$

**For collective models, additionally:**
- [ ] Social weight vector $w_i$ — can use network centrality, influence scores, etc.
- [ ] Source trust score per agent (separates content evaluation from source evaluation)
- [ ] Amplification network: transmission probability as function of Fit, not Fidelity
- [ ] Suppression event generator: external correction attempts → $V_{\mathcal{C}}$ verification signals

**Key parameters to fit from data:**
- $\alpha$: novelty sensitivity — how quickly $\Phi$ drops with distance from $B$
- $\phi_{threshold}$: the boundary between integration and threat response
- $\lambda$: learning rate — how quickly $B$ shifts under new inputs
- $\kappa$: coupling strength — how much agents synchronize with each other vs. external signals
- $\Omega$: cognitive/processing bandwidth — determines when saturation switches to heuristic mode

**Observable predictions (testable):**
1. Fact-flooding a CEB increases $\kappa$ (measurable as opinion synchronization within group)
2. Suppression events expand group size (measurable as recruitment rate)
3. $H$ expansion precedes $B$ shift — you should observe "tolerates exposure to X" before "agrees with X"
4. Broker agents exist in every CEB (the ones who bridge to the unrepresentable class) — they are the only viable channels for protocol delivery

---

## Closing Note

Paper 7 established that the individual's baseline can be pathological from formation. Paper 8 established that suppression validates the deviant signal.

**Bet** is the synthesis: these mechanisms, operating at collective scale and across time, are sufficient to explain the emergence and persistence of every major form of social pathology without invoking irrationality, malice, or deficiency on the part of any individual agent.

The agents are doing the right thing. The system is doing the wrong thing. Those are different problems with different solutions.

The Re-zeroing Protocol does not ask anyone to be wrong. It asks the horizon to be wider.

---

*Filed: Noetic Lab · theory/  
Cross-reference: MBD Papers 7, 8 · YellowHapax/Endemic-Baseline · YellowHapax/Suppressive-and-Emergent-Phenomenon*
