# Coupled double-integrator experiment ($N=3$): design notes

**Status:** design decided (Section 4b); nothing implemented or run yet. Decisions taken: common target at the centre, one shared pairwise barrier per pair, no obstacles, initial conditions as in the repo, standalone folder.
**Notation:** follows `submission_v2.tex` (Sections "End-to-End Training...", "Background", Appendix "HOCBF" and "Details on Numerical Experiments"). One deliberate deviation: the tex uses $z$ for both the full state and the position of a single double integrator. Here the position of agent $i$ is $p^{(i)}$ and its velocity is $v^{(i)}$, so $z^{(i)}=(p^{(i)},v^{(i)})$. (Open decision D5.)

---

## 1. Idea in one paragraph

Take $N=3$ double integrators that start at different positions $x\sim\rho$ and must all reach the **same** target position. There are **no fixed obstacles**. Instead, every pair of agents has its own barrier function $h_{ij}$ that keeps the two agents apart. The safety constraint for pair $(i,j)$ then involves *both* controls $u^{(i)}$ and $u^{(j)}$, so the CBF-QP no longer separates across agents. That coupling inside the QP is what makes this experiment different from the six in the paper.

Grounding: in `double_integrator_multi.py`, `construct_cbf_constraints` builds one row per (agent, obstacle) pair with `K_i[:, 2*i:2*i+2] = -2 * dp[:, i, :]`, i.e. only agent $i$'s control block is nonzero. So in the existing examples the QP separates across agents, and agents interact only through the shared policy network and the shared cost.

---

## 2. Dynamics

Agent $i\in\{1,\dots,N\}$, with $p^{(i)}\in\mathbb R^2$, $v^{(i)}\in\mathbb R^2$, $u^{(i)}\in\mathbb R^2$:
$$
\dot p^{(i)}=v^{(i)},\qquad \dot v^{(i)}=u^{(i)} .
$$
Joint state $z=(z^{(1)},\dots,z^{(N)})\in\mathbb R^{4N}$, joint control $u\in\mathbb R^{2N}$ ($n=4N$, $m=2N$; for $N=3$: $n=12$, $m=6$). This is the control-affine form $\dot z=f(t,z)+g(t,z)u$ of the tex, with $f(t,z)=(v^{(1)},0,\dots,v^{(N)},0)$ and $g$ the constant block matrix `B` already used in `double_integrator_multi.py`.

## 3. Cost

Same as the tex (Appendix "Dynamics and Optimal Control Objectives"), including the velocity penalty for the double integrator:
$$
L(t,z,u)=\tfrac12\|u\|^2+\tfrac12\|v\|^2,\qquad
G(z(T))=\tfrac12\sum_{i=1}^N\|p^{(i)}(T)-p_{\rm target}\|^2+\tfrac12\sum_{i=1}^N\|v^{(i)}(T)\|^2 ,
$$
with the **same** $p_{\rm target}\in\mathbb R^2$ for all agents, and training objective
$\min_\theta\ \mathbb E_{x\sim\rho}\big[\int_0^T L\,dt+\omega\,G(z_x(T))\big]$ with the continuation on $\omega$ used in the paper.

Conventions in the existing code (`double_integrator_multi.py`, `train.py`), which the standalone experiment will follow so results are comparable:
* $G$ contains the position error **and** a velocity penalty, as above ($v_{\rm target}=0$; this agrees with the tex form $\tfrac12\|z(T)-z_{\rm target}\|^2$).
* The running cost in `compute_loss` is `h*lagrangian(u) + 2*h*(0.5*||v||^2)`, i.e. $\tfrac12\|u\|^2+\|v\|^2$ per unit time: the velocity term has weight 1 on $\|v\|^2$, **twice** the $\tfrac12\|v\|^2$ written in the tex. (Decision D7.)
* The terminal weight $\omega$ (`alpha_terminal`) starts at $20$ and is increased by $5$ every $10$ epochs until it reaches $500$ (`train.py`, `double_integrator_multi` branch); `alpha_running = 1`; `T=10`, `dt=0.2` (50 RK4 steps), batch size 32, Adam with learning rate $10^{-3}$.

---

## 4. Pairwise barrier and its HOCBF constraint

For each pair $i<j$ and a safety distance $d>0$:
$$
h_{ij}(z)=\|p^{(i)}-p^{(j)}\|^2-d^2 ,\qquad \mathcal C_{ij}=\{z:\ h_{ij}(z)\ge 0\}.
$$
With $\delta p=p^{(i)}-p^{(j)}$ and $\delta v=v^{(i)}-v^{(j)}$:
$$
\dot h_{ij}=2\,\delta p^\top\delta v,\qquad
\ddot h_{ij}=2\|\delta v\|^2+2\,\delta p^\top\big(u^{(i)}-u^{(j)}\big).
$$
The control first appears in $\ddot h_{ij}$, so $h_{ij}$ has relative degree $l=2$ and we use the HOCBF of the tex appendix:
$$
h_{0}=h_{ij},\quad h_{1}=\dot h_{0}+\alpha_1(h_{0}),\quad h_{2}=\dot h_{1}+\alpha_2(h_{1})\ \ge 0 .
$$
Taking linear class-$\mathcal K$ functions $\alpha_1(s)=\alpha_2(s)=\gamma s$ (as in the code, `gamma(x)=1.0*x`):
$$
h_{2}=\ddot h_{ij}+2\gamma\,\dot h_{ij}+\gamma^2 h_{ij}\ \ge 0 .
$$
Written in the form $A(z)u\le b(z)$ of the tex, the row for pair $(i,j)$ is
$$
\underbrace{-2\,\delta p^\top}_{\text{block } i}\,u^{(i)}\;+\;\underbrace{2\,\delta p^\top}_{\text{block } j}\,u^{(j)}\;\le\;
b_{ij}(z):=2\|\delta v\|^2+2\gamma\,\dot h_{ij}+\gamma^2 h_{ij},\qquad \dot h_{ij}=2\,\delta p^\top\delta v .
$$
Check against the code: for an obstacle, `d = 2*||v||^2 + gamma(Lf_h) + gamma(psi1)` with `psi1 = Lf_h + gamma(h)`, which equals $2\|v\|^2+\gamma\dot h+\gamma(\dot h+\gamma h)=2\|v\|^2+2\gamma\dot h+\gamma^2 h$. The pair version above differs only in that the relative quantities $\delta p,\delta v$ replace $p-c,\ v$ and the row has two active blocks (with opposite signs). The sign convention in `construct_cbf_constraints` (`K = -2*dp`, constraint `K u <= d`) is the same as above.

**Number of constraints:** $c=\binom N2=3$ for $N=3$, so the lifted variable is $y=(u,s)\in\mathbb R^{m+c}=\mathbb R^{9}$.

**Initial feasibility:** the HOCBF theorem needs $z(0)\in\bigcap_{k}\mathcal C_k$, i.e. $h_{ij}(x)\ge0$ and $h_1(x)\ge0$. With zero initial velocity, $h_1=\gamma h_{ij}\ge0$ whenever $\|p^{(i)}-p^{(j)}\|\ge d$. So $\rho$ must put agents at least $d$ apart.

---

## 4b. The decided problem ($N=3$, shared pairwise barrier, common target)

**Data.** $N=3$ agents; $z^{(i)}=(p^{(i)},v^{(i)})\in\mathbb R^4$, $z\in\mathbb R^{12}$, $u\in\mathbb R^6$; centre $c=(1.5,\,1.5)$; common target $p_{\rm target}=c$; safety distance $d$ (start with $d=0.7$, to be tuned); $\gamma=1$ (so $\alpha_1(s)=\alpha_2(s)=s$); DYS step size $\zeta=0.5$ (`alpha` in `DYSProjector`).

**Initial distribution $\rho$** (the repo's, with $N=3$): for $i=1,2,3$
$$
p^{(i)}(0)=c+R\begin{pmatrix}\cos\varphi_i\\ \sin\varphi_i\end{pmatrix}+\sigma\,\xi_i,\quad
\varphi_i=-\tfrac{2\pi(i-1)}{3},\ \ R=\sqrt8,\ \ \sigma=0.05,\ \ \xi_i\sim\mathcal N(0,I_2),\qquad v^{(i)}(0)=0 .
$$

**Training problem** (tex eqs. for the training problem, specialised):
$$
\min_\theta\ \mathbb E_{x\sim\rho}\Big[\int_0^T\!\big(\tfrac12\|u^\star_\theta\|^2+\tfrac12\|v\|^2\big)\,dt+\omega\,G\big(z_x(T)\big)\Big],\qquad
G(z)=\tfrac12\sum_{i=1}^3\|p^{(i)}-c\|^2+\tfrac12\sum_{i=1}^3\|v^{(i)}\|^2,
$$
$$
\dot p^{(i)}=v^{(i)},\quad \dot v^{(i)}=u^{\star(i)}_\theta,\quad z_x(0)=x,\qquad
u^\star_\theta=Q_uy^\star_\theta,\quad y^\star_\theta=T_\theta(y^\star_\theta;z_x),
$$
with the running velocity weight as in D7. $T_\theta$ is the DYS operator of the tex for the constraint set below.

**Constraint set** $C(z)=\{u\in\mathbb R^6: A(z)u\le b(z)\}$, one row per pair. With $\delta p_{ij}=p^{(i)}-p^{(j)}$, $\delta v_{ij}=v^{(i)}-v^{(j)}$, $h_{ij}=\|\delta p_{ij}\|^2-d^2$:
$$
A(z)=\begin{pmatrix}
-2\,\delta p_{12}^\top & +2\,\delta p_{12}^\top & 0\\[2pt]
-2\,\delta p_{13}^\top & 0 & +2\,\delta p_{13}^\top\\[2pt]
0 & -2\,\delta p_{23}^\top & +2\,\delta p_{23}^\top
\end{pmatrix}\in\mathbb R^{3\times6},\qquad
b_{ij}(z)=2\|\delta v_{ij}\|^2+4\gamma\,\delta p_{ij}^\top\delta v_{ij}+\gamma^2h_{ij},
$$
(block columns correspond to $u^{(1)},u^{(2)},u^{(3)}\in\mathbb R^2$; $b=(b_{12},b_{13},b_{23})^\top$). The lifted variable is $y=(u,s)\in\mathbb R^{9}$, with $A(z)u+s=b(z)$, $s\ge0$ as in the tex. These HOCBF constraints enforce $h_{ij}(z(t))\ge0$ for all $t$ provided the start satisfies $h_{ij}\ge0$ and $h_1\ge0$ (true for $\rho$, since the agents start about $4.9$ apart with zero velocity).

**What the optimum looks like** (Section 6): the position part of $G$ cannot go below $d^2/2$, attained by an equilateral triangle of side $d$ centred at $c$. With $d=0.7$: $d^2/2=0.245$ and circumradius $d/\sqrt3\approx0.40$.

**Quantities to log** (each defined from $z$ along rollouts):
* Safety: $\min_{t,\,i<j}h_{ij}(z(t))\ge0$ and $\min_{t,\,i<j}h_{1,ij}(z(t))\ge0$, with $h_{1,ij}=\dot h_{ij}+\gamma h_{ij}=2\delta p_{ij}^\top\delta v_{ij}+\gamma h_{ij}$.
* Activity: fraction of steps where $A(z)u_\theta^{\rm nom}\le b(z)$ is violated for some row (the projection is active).
* Packing: $\mathcal E_{\rm pos}=\tfrac12\sum_i\|p^{(i)}(T)-c\|^2-\tfrac{d^2}{2}\ (\ge0)$, centroid offset $\|\bar p(T)-c\|$, and side-length error $\max_{i<j}\big|\|\delta p_{ij}(T)\|-d\big|$.
* DYS: iterations and final residual per step; $\sigma_{\min}$ of the rows of $A(z)$ that are active (see Section 7b).

*Checks done:* the form of the row and of $b_{ij}$ was verified numerically against finite differences (Section 12). The explicit $A(z)$ above is that row written out for $N=3$.

---

## 5. How it plugs into DYS-JFB (no change to the method)

Everything in the tex Background section applies verbatim: $A(z)u+s=b(z)$, $s\ge0$, $C_1=\{s\ge0\}$, $C_2(z)=\{A(z)u+s=b(z)\}$, $Q_u=[I,0]$, and the DYS operator $T_\theta$ with step size $\zeta$ (the code's `alpha`). Only the matrix $A(z)$ changes: its rows now have two nonzero blocks.

To verify before relying on it: how `DYSProjector` (in `utils.py`) forms $P_{C_2}$, and whether that cost depends on $A$ being block-sparse. Not read yet.

---

## 6. Important issue: "same target" conflicts with the barrier

If all agents are driven to exactly $p^{(i)}=p_{\rm target}$, then $\|p^{(i)}-p^{(j)}\|=0<d$, which **violates** $h_{ij}\ge0$. A literal common target point is not reachable by a safe trajectory. What the optimizer can do is get as close as safety allows.

**Proposition (terminal packing bound).** Let $p^{(1)},\dots,p^{(N)}\in\mathbb R^2$ with $\|p^{(i)}-p^{(j)}\|\ge d$ for all $i\ne j$, and let $\bar p=\frac1N\sum_i p^{(i)}$. Then
$$
\sum_i\|p^{(i)}-p_{\rm target}\|^2=\sum_i\|p^{(i)}-\bar p\|^2+N\|\bar p-p_{\rm target}\|^2\ \ge\ \sum_i\|p^{(i)}-\bar p\|^2=\frac1N\sum_{i<j}\|p^{(i)}-p^{(j)}\|^2\ \ge\ \frac{N-1}{2}\,d^2 .
$$
Equality holds iff $\bar p=p_{\rm target}$ and all pairwise distances equal $d$. For $N=3$ in the plane this is an **equilateral triangle of side $d$ centred at $p_{\rm target}$** (circumradius $d/\sqrt3$). Hence the position part of $G$ satisfies
$$
\tfrac12\sum_i\|p^{(i)}-p_{\rm target}\|^2\ \ge\ \tfrac{(N-1)d^2}{4}\ \ \big(=\tfrac{d^2}{2}\text{ for }N=3\big).
$$
(Proof: the first equality is the parallel-axis identity; the middle identity is $\sum_{i<j}\|p^{(i)}-p^{(j)}\|^2=N\sum_i\|p^{(i)}-\bar p\|^2$; the last step uses $\binom N2$ pairs each $\ge d^2$.)

Consequences:
* The terminal position cost cannot go to zero. It has a positive floor, so "terminal error $\to 0$" is the wrong success criterion here.
* The floor gives a **ground-truth diagnostic** for $N=3$: a well-trained controller with large $\omega$ should end near an equilateral triangle of side $\approx d$ around $p_{\rm target}$, with $\tfrac12\sum\|p^{(i)}-p_{\rm target}\|^2\approx d^2/2$. Caveat: the bound is for the position part of $G$ alone. The full objective also contains the running cost and finite $\omega$, so the learned terminal configuration will generally sit slightly outside the packing.
* For $N\ge4$ in the plane, all pairwise distances cannot equal $d$, so the bound is not attained (it is only a lower bound).

---

## 6b. Making the constraints (the "penalties") activate

The repo's `double_integrator_multi` setup (from `train.py`): agents start equally spaced on a circle of radius $\sqrt8\approx2.83$ around $c=(1.5,1.5)$, and agent $i$'s target is the point at the **same angle** on a small ring of radius $r_t=\sqrt2\cdot0.2\approx0.283$. For $N=3$ that ring has side $\sqrt3\,r_t\approx0.49$. So with the repo's targets the agents move radially inward and never need to avoid each other unless $d>0.49$.

I suggest one parameter $r_t$ (target ring radius) and one $d$ (safety distance):

| Variant | Targets | $d$ | What happens |
|---|---|---|---|
| **A (primary)** | all agents: $p_{\rm target}=c$ ($r_t=0$) | $d\approx0.7$ (to tune) | all three funnel to one point; all three pair constraints become active near the end and stay active; terminal floor $\tfrac12\sum\|p^{(i)}-p_{\rm target}\|^2\ge d^2/2\approx0.245$; checkable equilateral outcome (side $d$, circumradius $d/\sqrt3\approx0.40$) |
| B | repo's ring, $r_t=0.283$ | $d>0.49$ (e.g. $0.7$) | the targets themselves are closer than $d$, so constraints must be active at the end; partial packing |
| C (optional) | point reflection: $p^{(i)}_{\rm target}=2c-p^{(i)}(0)$ | any | three straight paths all cross at $c$ at the same time: a strong 3-way avoidance test |

Variant A is exactly "same target". Whether the constraints are active in a given rollout can be read off as the fraction of steps where the nominal control violates $h_2\ge0$ (the existing code already computes this as `is_projected`).

---

## 7. Feasibility of the coupled QP (corrected)

*An earlier draft of this note claimed collinear configurations could make the QP infeasible. That was wrong.*

**Statement.** Suppose controls are unbounded and $p^{(i)}\ne p^{(j)}$ for all $i\ne j$ (true on the safe set, since $h_{ij}\ge0$ gives $\|\delta p\|\ge d$). Then $\{u: A(z)u\le b(z)\}\neq\emptyset$.

**Why.** Push every agent away from the centroid $\bar p$: $u^{(i)}=M\,(p^{(i)}-\bar p)$. Then $\delta p_{ij}^\top(u^{(i)}-u^{(j)})=M\|\delta p_{ij}\|^2\ge Md^2$, so row $(i,j)$ gives $A_{ij}u=-2M\|\delta p_{ij}\|^2\le -2Md^2$. Choosing $M\ge\max_{ij}\big(-b_{ij}\big)_+/(2d^2)$ makes every row hold at once, for any number of agents and any geometry (including collinear).

**Checked numerically** (LP feasibility with random positions and velocities, $d=0.5$): 0 infeasible cases out of 300 each for $N=3,4,6$ with generic and with collinear positions, and for $N=12$ generic; the explicit $u^{(i)}=M(p^{(i)}-\bar p)$ satisfied all rows in every sampled case that was run.

**Caveats.** This is for unbounded controls. The double-integrator branch of `train.py` applies no clipping to the network output (`u_fn` returns `net(z, ti)`), so it applies. With control bounds the statement can fail.

## 7b. The rank assumption in the paper (corrected twice; read carefully)

**Assumption 1 (tex):** $A(z)\in\mathbb R^{c\times m}$ has full row rank, i.e. all $c$ rows are linearly independent, for all $z$.

**(a) What the pair-constraint matrix looks like.** Row $(i,j)$ is $-2\delta p_{ij}^\top$ in block $i$ and $+2\delta p_{ij}^\top$ in block $j$. This is the *rigidity matrix* of the complete graph on the agents' positions. Moving all agents by the same translation, or rotating the whole group, leaves every $\|\delta p_{ij}\|$ unchanged, so $A\cdot(\text{translation})=0$ and $A\cdot(\text{rotation})=0$ (checked numerically to $10^{-16}$). Hence for generic positions
$$
\operatorname{rank}A\ \le\ 2N-3,\qquad\text{while } c=\tbinom N2 .
$$
Numerical ranks over random planar configurations (200 samples each): $N=3$: rank 3 of $c=3$ (full); $N=4$: 5 of 6; $N=5$: 7 of 10; $N=6$: 9 of 15; $N=7$: 11 of 21; $N=8$: 13 of 28. **So literal full row rank holds only for $N=3$** (and then only for non-collinear positions, where the rank drops to 2). *My earlier statement "it can hold for $N\le5$" was wrong: $c\le m$ is necessary but the rigidity structure forces rank $\le 2N-3$.*

**(b) The paper's own examples.** As written, the assumption also cannot hold literally for them: `double_integrator_single` has 3 obstacle rows and $m=2$ controls; `double_integrator_multi` has $6\times3=18$ rows and $m=12$ (shapes asserted in `compute_loss`).

**(c) What the proof appears to use.** I read the proofs of Lemmas `dPC2`, `dPC1`, `intersect` and `PH2norm` (Appendix "Proof of Contraction").
* `dPC2` needs only that $[A\ I]$ has full row rank, which is automatic because of the identity block.
* `PH2norm` does not use the assumption in its argument (it only uses $[A\ I]p\ne0$).
* `intersect` ($\mathcal H_{1,y}^\perp\cap\mathcal H_{2,z}^\perp=\{0\}$) is where the assumption is used: from $\sum_i\beta_i a_i=0$ it concludes $\beta_i=0$. But $\mathcal H_{1,y}^\perp=\operatorname{span}\{e_{m+i}:y_{m+i}<0\}$, so the $\beta_i$ that matter are supported only on the indices $S=\{i:y_{m+i}<0\}$, i.e. the constraints that $P_{C_1}$ clips (the *active* ones). Reading the proof literally, it needs the rows $\{a_i\}_{i\in S}$ to be linearly independent, not all $c$ rows. That is the standard linear-independence constraint qualification (LICQ) on active constraints.

*This is my reading of one lemma. I have not checked the later steps (Theorem `contraction`, Corollary `contraction`, the uniform constant $\gamma<1$), which may need a quantitative version (e.g. $\sigma_{\min}$ of the active rows bounded away from 0 on the compact domain). The paper's statement and its examples appear inconsistent as written, and the active-set version looks like what is actually needed; worth checking with the proof in hand.*

**(d) What this means for the coupled experiment.**
* $N=3$ with all three pairs active (the common-target end state): the three rows are independent iff the agents are not collinear; at the equilateral end state they are independent. So the active-set condition holds there. Near-collinear configurations would make $\sigma_{\min}$ small, hence slow DYS convergence; log $\sigma_{\min}$ of the active rows.
* $N\ge4$: independence of the active rows is not automatic; whether it holds along trajectories is untested.

---

## 8. Initial conditions: same as the repo

`sample_initial_condition` in `double_integrator_multi.py`: agents equally spaced clockwise on a circle of radius $\sqrt8$ around $(1.5,1.5)$, Gaussian jitter of std $0.05$ on the positions (`z0_std=5e-2`), **zero** initial velocity. For $N=3$ this gives an almost equilateral triangle of side $\approx4.9\gg d$, so $h_{ij}>0$ and $h_1=\gamma h_{ij}>0$ at $t=0$ (initial feasibility of the HOCBF is automatic) and the start is far from collinear. The distribution is narrow (it is a small neighbourhood of one configuration), the same as in the existing experiments, which keeps results comparable.

---

## 9. Experiments to run

| # | Question | Setting | Compare | Outputs |
|---|---|---|---|---|
| E1 | Does the method work on a coupled QP? | $N=3$, $d$ fixed | DYS-JFB vs DYS-AD vs CVXPYLayers (hard and relaxed) | running cost $L$, terminal cost $G$, final packing error vs equilateral target |
| E2 | Is it safe? | same as E1 | — | $\min_{t,i<j}h_{ij}$ and $\min h_1$ along rollouts (must stay $\ge0$) |
| E3 | Scaling with number of coupled constraints | $N=3,6,12,\dots$, $c=\binom N2$ (outside the literal rank assumption for $N\ge4$, Section 7b) | all three methods | time/epoch, peak CPU/GPU memory, divergence |

Use the same hyperparameter conventions as the paper where possible ($T=10$, $dt=0.2$, RK4, batch 32, Adam, same $\omega$ continuation), so results are comparable.

---

## 10. Implementation plan: standalone, no changes to existing repo files

Everything lives in `coupled_double_integrator/`; no existing file is edited.

* **Reused from the repo (imported as-is):** `utils.py` -> `DYSProjector`, `rk4_step`, `ControlNet`, `ResBlock`. Call convention seen in `compute_loss`: `proj(u_nom, K, d, max_iter, tol)` with `K` of shape `(batch, c, m)` and `d` of shape `(batch, c, 1)`.
* **Not reusable from `double_integrator_multi.py`:** that module fixes `n_agent = 6` at import, builds `A`, `B` from it at import time, and hard-codes `z.view(z.shape[0], 6, 4)` inside `compute_loss`. Using it with $N=3$ would require editing it. So the new folder gets its own small `problem.py` (dynamics, cost, $\rho$, pair CBF, rollout with the same loop structure as `compute_loss`) and `train_coupled.py` (loop structure copied from `train.py`: Adam, $\omega$ continuation, logging).
* **CBF is one swappable function.** To let the barrier be redefined later, the row construction takes any smooth $h(\delta p)$ and uses its gradient and Hessian in $\delta p$:
$$
-\nabla h(\delta p)^\top\big(u^{(i)}-u^{(j)}\big)\ \le\ \delta v^\top\nabla^2 h(\delta p)\,\delta v+2\gamma\,\nabla h(\delta p)^\top\delta v+\gamma^2 h(\delta p).
$$
  For $h=\|\delta p\|^2-d^2$ ($\nabla h=2\delta p$, $\nabla^2h=2I$) this is exactly the row of Section 4. Changing the CBF means changing only $h$ (autograd for derivatives, or closed forms for speed). *The general formula follows from $h_2=\ddot h+2\gamma\dot h+\gamma^2h$ with $\dot h=\nabla h^\top\delta v$ and $\ddot h=\delta v^\top\nabla^2h\,\delta v+\nabla h^\top\delta u$; the $h=\|\delta p\|^2-d^2$ case was checked numerically, the general form has not been.*
* **Diagnostics logged per epoch:** fraction of steps where the projection is active, $\min_{t,i<j}h_{ij}$ and $\min h_1$, DYS iterations and residual, $\sigma_{\min}(A(z))$, terminal packing error vs the equilateral triangle (variant A).
* **Baselines (later):** DYS-AD via `grad_mode="ad"` in `DYSProjector`; CVXPYLayers once the DYS-JFB runs behave.

## 11. Decisions

* **Decided:** standalone folder, no edits to existing repo files; initial conditions as in the repo (Section 8); common target at the centre with $d\approx0.7$ (variant A, Section 6b); one shared barrier $h_{ij}$ per pair with the same $d$ for all pairs; no obstacles.
* **Follow-ups, not planned now:** variants B and C of Section 6b; heterogeneous radii ($d_{ij}=\max(d_i,d_j)$); adding fixed obstacles.
* **Open, small:** D5 notation ($p^{(i)},v^{(i)}$ here, vs the tex's overloaded $z$); D7 running velocity weight (code: $\|v\|^2$, tex: $\tfrac12\|v\|^2$; default is to follow the code); the final value of $d$ (tune; start at $0.7$).

## 12. What is grounded vs. not

* **Read in files:** the HOCBF recursion, $A(z)u\le b(z)$, Assumption 1 and the DYS setup (tex); `construct_cbf_constraints`, `G`, `compute_loss`, `sample_initial_condition` (`double_integrator_multi.py`); the target/obstacle/hyperparameter setup and training loop (`train.py`); the `grad_mode` branches of `DYSProjector` (`utils.py`).
* **Derived and checked numerically** (numpy/scipy scripts, random samples): the pair row and $b_{ij}$; the packing identity, bound and equilateral value; the rank of $A$ ($N=3$: 3 generic, 2 collinear; $N=4..8$: rank $=2N-3<c$); QP feasibility for $N=3,4,6,12$ incl. collinear.
* **Not checked:** the general-$h$ row formula; any training behaviour; whether the later steps of the contraction proof need more than independence of the active rows (only Lemmas `dPC2`, `dPC1`, `intersect`, `PH2norm` were read).
* **Not read:** `DYSProjector.__init__`/`forward` internals beyond the `grad_mode` branches (how $P_{C_2}$ is formed and whether its cost depends on block-sparse $A$).
