Promising Computational Gains, with Limitations in Theoretical Coverage and Experimental Evaluation
Official Reviewby Program Committee EuaX08 Oct 2026, 03:26 (modified: 24 Oct 2026, 08:19)Program Chairs, Area Chairs, Senior Program Committee, Program Committee EuaX, Authors, Program Committee SubmittedRevisions
Paper Summary:
This paper studies end-to-end learning of feedback control policies with control barrier function (CBF) safety layers. It combines Davis–Yin splitting (DYS) for solving the safety-layer quadratic programs with Jacobian-free backpropagation (JFB) to reduce training time and memory costs. The paper provides descent and gradient-flow convergence analyses and evaluates the method on single-integrator, double-integrator, and quadcopter tasks, with the largest example involving 1,200 state dimensions and 400 control inputs.

#### **Review**

**Official Review**by Program Committee ELP6**22 Oct 2026, 04:18 (modified: 24 Oct 2026, 08:19)**Program Chairs, Area Chairs, Senior Program Committee, Program Committee ELP6, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=yMLTzMZ3Io)

 **Paper Summary:** The paper proposes a method for learning high-dimensional feedback controllers under hard constraints by integrating control barrier functions into a neural network. This is done by embedding a quadratic program as an optimization layer within the network. The method is made to work with high dimensions by building off recent jacobian-free backpropagation work and shows this results in solutions in up to 1200 dimensional state problems while using fewer cpu and gpu memory than existing methods.

 **Strengths:** * The challenge of introducing hard constraints into a neural policy is an interesting one, and the proposed solution appears to be an effective solution to this.

* The results are fairly convincing, producing solutions across multiple types of dynamical systems with strong cpu/gpu reductions.

 **Weaknesses:** * The theoretical novelty is somewhat incremental, directly building off prior of DYS and JFB works and largely extends prior proofs (even the format of them).

* There is no non-end-to-end policy as a point of comparison, e.g. train a policy independently and then apply CBF-QF at deployment. This would help to justify the utility of end-to-end optimization.
* The experiments don't analyze the quality of the produced plans at all (just their existence and cpu/gpu memory).
* Some of the assumptions seem rather strong (particularly Assumptions 7 and 8 which are stronger than normal). For example assumption 7 says the variance must be 0 when the hamiltonian gradient is 0. This may not hold on relatively simple cases (e.g. a robots that go left/right around an obstacle). Similarly, assumption 8 would fail any time a time average is much smaller than a peak average (e.g. opposite inputs). These aren't necessarily huge problems, but they do restrict what problems the convergence analysis holds on.

 **Justification of Recommendation:** While I list more weaknesses than strengths, in general the weaknesses are relatively minor. My biggest issue is with the lack of a non-end-to-end baseline and that the experiments don't analyze plan quality at all. This means I can't give this paper more than a borderline score.

 **Specific Points of Feedback for Rebuttal:** Are the assumptions as restrictive as they appear to be? See my last weakness point above.

**Rating:** **6: Marginally above acceptance threshold**

**Confidence:** **2: The reviewer is willing to defend the evaluation, but it is quite likely that the reviewer did not understand central parts of the paper**

#### **Promising Computational Gains, with Limitations in Theoretical Coverage and Experimental Evaluation**

**Official Review**by Program Committee EuaX**08 Oct 2026, 03:26 (modified: 24 Oct 2026, 08:19)**Program Chairs, Area Chairs, Senior Program Committee, Program Committee EuaX, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=ingvI3e0B0)

 **Paper Summary:** This paper studies end-to-end learning of feedback control policies with control barrier function (CBF) safety layers. It combines Davis–Yin splitting (DYS) for solving the safety-layer quadratic programs with Jacobian-free backpropagation (JFB) to reduce training time and memory costs. The paper provides descent and gradient-flow convergence analyses and evaluates the method on single-integrator, double-integrator, and quadcopter tasks, with the largest example involving 1,200 state dimensions and 400 control inputs.

 **Strengths:** 1.Differentiating through safety layers at every time step creates a substantial training cost. Combining DYS with JFB avoids retaining the computation graph of all solver iterations and offers a relatively straightforward approach to reducing this cost.

2.The DYS-AD baseline uses the same forward solver, helping isolate the benefits of approximate backpropagation. Table 2 shows reduced GPU memory usage relative to DYS-AD. The proposed method also completes training on the 100-quadcopter example, where the other baselines fail, demonstrating a useful scaling advantage in the reported experiments.

3.The theoretical analysis considers both the nonsmooth safety layer and the gradient approximation introduced by JFB. Establishing descent and convergence results for the trajectory-level training objective goes beyond analyzing an individual optimization layer and is a valuable part of the work.

 **Weaknesses:** 1.Assumption 3 requires the constraint matrix to have full row rank, but both double-integrator examples have more constraints than control inputs and cannot satisfy this condition. Consequently, the contraction and training convergence results relying on this assumption do not cover these experimental settings.

2.The high-dimensional examples are constructed mainly by increasing the number of agents, without demonstrating constraints that couple different agents. Given the network output, the safety-layer QP can be decomposed across agents, but it is unclear whether the baselines exploit this structure. Scalability on high-dimensional tasks with coupled constraints remains untested.

3.The methods achieve different final control costs in Table 1, with some baselines obtaining lower costs than DYS-JFB. The paper does not compare the total training time required to reach comparable control performance, limiting the assessment of practical training efficiency. The extent to which the relaxed baseline violates the original CBF constraints is also not reported.

 **Justification of Recommendation:** I lean toward acceptance. The paper offers a relatively straightforward approach to a meaningful computational bottleneck in safe control policy learning. The DYS-AD comparison demonstrates computational benefits from approximate backpropagation, and the 100-quadcopter result shows a scaling advantage over the evaluated baselines. My main reservations concern the theoretical assumptions not covering some experiments, whether the baselines exploit the structure of the high-dimensional tasks, and the lack of efficiency comparisons at comparable control performance. These issues prevent a stronger recommendation, but the method and reported results still support a weak accept.

 **Specific Points of Feedback for Rebuttal:** 1.Does Assumption 3 require the full constraint matrix to have full row rank? How should the theoretical guarantees be interpreted for the two double-integrator examples that do not satisfy this condition?

2.Do the multi-agent examples include constraints coupling different agents? Do the baselines exploit the agent-wise separability of the safety-layer QP?

3.Are results available comparing the total training time needed to reach comparable control performance? What is the actual violation of the original CBF constraints under the relaxed baseline?

**Rating:** **6: Marginally above acceptance threshold**

**Confidence:** **4: The reviewer is confident but not absolutely certain that the evaluation is correct**

Strengths:
1.Differentiating through safety layers at every time step creates a substantial training cost. Combining DYS with JFB avoids retaining the computation graph of all solver iterations and offers a relatively straightforward approach to reducing this cost.

2.The DYS-AD baseline uses the same forward solver, helping isolate the benefits of approximate backpropagation. Table 2 shows reduced GPU memory usage relative to DYS-AD. The proposed method also completes training on the 100-quadcopter example, where the other baselines fail, demonstrating a useful scaling advantage in the reported experiments.

3.The theoretical analysis considers both the nonsmooth safety layer and the gradient approximation introduced by JFB. Establishing descent and convergence results for the trajectory-level training objective goes beyond analyzing an individual optimization layer and is a valuable part of the work.

Weaknesses:
1.Assumption 3 requires the constraint matrix to have full row rank, but both double-integrator examples have more constraints than control inputs and cannot satisfy this condition. Consequently, the contraction and training convergence results relying on this assumption do not cover these experimental settings.

2.The high-dimensional examples are constructed mainly by increasing the number of agents, without demonstrating constraints that couple different agents. Given the network output, the safety-layer QP can be decomposed across agents, but it is unclear whether the baselines exploit this structure. Scalability on high-dimensional tasks with coupled constraints remains untested.

3.The methods achieve different final control costs in Table 1, with some baselines obtaining lower costs than DYS-JFB. The paper does not compare the total training time required to reach comparable control performance, limiting the assessment of practical training efficiency. The extent to which the relaxed baseline violates the original CBF constraints is also not reported.

Justification of Recommendation:
I lean toward acceptance. The paper offers a relatively straightforward approach to a meaningful computational bottleneck in safe control policy learning. The DYS-AD comparison demonstrates computational benefits from approximate backpropagation, and the 100-quadcopter result shows a scaling advantage over the evaluated baselines. My main reservations concern the theoretical assumptions not covering some experiments, whether the baselines exploit the structure of the high-dimensional tasks, and the lack of efficiency comparisons at comparable control performance. These issues prevent a stronger recommendation, but the method and reported results still support a weak accept.

Specific Points of Feedback for Rebuttal:
1.Does Assumption 3 require the full constraint matrix to have full row rank? How should the theoretical guarantees be interpreted for the two double-integrator examples that do not satisfy this condition?

2.Do the multi-agent examples include constraints coupling different agents? Do the baselines exploit the agent-wise separability of the safety-layer QP?

3.Are results available comparing the total training time needed to reach comparable control performance? What is the actual violation of the original CBF constraints under the relaxed baseline?

Rating: 6: Marginally above acceptance threshold
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct


#### **Paper addresses an important problem, but lacks in key areas**

**Official Review**by Program Committee 4Ksj**30 Aug 2026, 20:25 (modified: 24 Oct 2026, 08:19)**Program Chairs, Area Chairs, Senior Program Committee, Program Committee 4Ksj, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=ziHUdcsbir)

 **Paper Summary:** This paper presents an approach to end-to-end learning of safe control based on the CBF-QP formulation using Davis–Yin Splitting and Jacobian-Free Backpropagation to address the computational bottlenecks that prevent its application to high-dimensional systems. Convergence analysis and proofs are presented, followed by numerical experiments on common control benchmarks and some high-dimensional systems, showing that the proposed approach outperforms baselines.

 **Strengths:** * The paper addresses the important problem of end-to-end learning of safe control based on the CBF-QP formulation.

* The proposed method achieves better results and computational efficiency compared to the baselines considered in the paper.

 **Weaknesses:** * The discussion of related work is extremely limited. Several works that address safety through learned certificates are omitted (see Dawson et al. 2023 for an overview), including works on learned CBFs from vision (Abdi et al. 2023, and similar):

* H. Abdi, G. Raja, and R. Ghabcheloo (2023). “Safe control using vision-based control barrier function (V-CBF)”. In: IEEE international conference on robotics and automation. IEEE, pp. 782–788.
* C. Dawson, S. Gao, and C. Fan (2023). “Safe control with learned certificates: A survey of neural lyapunov, barrier, and contraction methods for robotics and control”. In: IEEE Transactions on Robotics 39.3, pp. 1749–1767.
* While the analysis and proofs of convergence are contributions, it is not fully clear what the exact methodological novelty is. Is it the application of DYS and JFB to the CBF-QP-based safety filter formulation or something else? Maybe, this can be clarified.
* The experiment section is sparse, and the most important details belong in the main paper and not in the appendix. Baselines are neither cited nor explained properly. The presentation of the results could have been better, especially Table 1. Baselines that actually produce results need to be considered, or the failure of the baselines needs to be analyzed (e.g. the Quadcopter (100,3) only has results for the proposed method).
* Experiments with more complex systems (quadrupeds, manipulators, etc.) in collision avoidance scenarios would have been highly beneficial, but are not considered.

 **Justification of Recommendation:** In spite of the theoretical contribution, the paper lacks a proper discussion of related works, includes a sparse description of the experimental setup, and does not fully justify the choice of baselines and included experiments.

 **Specific Points of Feedback for Rebuttal:** * Please refer to the points under "Weaknesses"

Minor:

* In Table 2, the lowest GPU utilization for the Single Integrator is achieved by CVXPYLayers (121.9) but a different entry is highlighted.

**Rating:** **5: Marginally below acceptance threshold**

**Confidence:** **3: The reviewer is fairly confident that the evaluation is correct**


#### **Addressing the important computational bottleneck, but limited novelty and gaps between theory and implementation.**

**Official Review**by Program Committee 8Gh9**30 Aug 2026, 13:17 (modified: 24 Oct 2026, 08:19)**Program Chairs, Area Chairs, Senior Program Committee, Program Committee 8Gh9, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=uY6RWxX9x8)

 **Paper Summary:** This paper proposes a framework for the learning of high-dimensional feedback controllers with hard safety constraints enforced by control barrier functions (CBFs). Incorporating CBFs as QP-based optimization layers can be computationally expensive, particularly in high-dimensional systems. To address this limitation, the paper combines operator splitting with Jacobian-Free Backpropagation (JFB), enabling scalable end-to-end training while retaining the safety guarantees provided by the CBF filter. The method is evaluated on high-dimensional nonlinear multi-agent control problems with state and control dimensions of up to 1,200 and 400, respectively. The experiments indicate substantial memory and runtime improvements over automatic differentiation and CVXPYLayer.

 **Strengths:** 1. The paper provides a convergence proof for policy training with embedded nonsmooth projection layers. The analysis based on Clarke generalized Jacobians is a useful theoretical contribution.

1. The method scales to high-dimensional control problems, including the experiment with 100 quadcopters and 1,200 state and 400 control dimensions.
2. The paper addresses the computational bottleneck of differentiating through CBF-QP layers. The reported results show clear reductions in time and memory usage compared with unrolled differentiation and CVXPYLayers.

 **Weaknesses:** 1. There is a gap between the theoretical guarantees and the implementation. The claimed hard safety guarantee is established for a continuous-time CBF-QP layer, while the implementation uses a finite DYS tolerance and discrete-time settings with . In addition, Theorem 12 also considers continuous-time gradient flow dynamics. The convergence claim should be stated more carefully to reflect this gap.

1. The claim of end-to-end learning is not appropriate. The paper is actually trained by directly minimizing a model-based trajectory cost through the known dynamics and the CBF layer.
2. Important experimental information is missing. The number of independent runs is not reported, and the main results do not include means and variances.
3. The baseline comparison is incomplete. Comparisons with representative learning-to-optimize (L2O) methods would strengthen the evaluation, as the main contribution focuses on learning to solve optimization problems with constraints. In addition to a CBF-based layer, the correction-based layer such as DC3, which can enforce the safety constraints with less computation time, should also be considered.
4. The novelty is limited. The method mainly combines the existing DYS and JFB techniques. The paper should clarify the new contribution beyond this integration.

 **Justification of Recommendation:** The paper addresses an important computational bottleneck and demonstrates promising scalability on high-dimensional control problems. The convergence analysis is also a useful contribution. However, the gap between the theoretical guarantees and the implementation, incomplete experimental reporting and baselines, and limited algorithmic novelty weaken the current submission. Clarifying these issues and providing more convincing comparisons would lead to a higher rating.

 **Specific Points of Feedback for Rebuttal:** 1. How do the safety and convergence guarantees apply to finite DYS tolerance and discrete-time integration?

1. How many independent runs were conducted?

**Rating:** **5: Marginally below acceptance threshold**

**Confidence:** **5: The reviewer is absolutely certain that the evaluation is correct and very familiar with the relevant literature**



#### **AI Review**

**AI Review**by AI Reviewer ()**27 Jul 2026, 07:59 (modified: 24 Oct 2026, 08:19)**Program Chairs, AI Reviewer, Area Chairs, Senior Program Committee, Authors, Program Committee Submitted[Revisions](https://openreview.net/revisions?id=lWPgdpzbf0)

 **Review:** **Title:** End-to-End Learning of Safe Optimal Feedback Control in High Dimensions with Control Barrier Function Layers

**Synopsis of the paper**
The paper studies end-to-end learning of feedback controllers with control-barrier-function safety filters embedded along simulated trajectories. It combines Davis–Yin operator splitting with Jacobian-Free Backpropagation to reduce the memory and differentiation costs of these filters. A nonsmooth convergence analysis and experiments on large multi-agent control systems are presented.

**Summary of Review**
The work addresses a concrete computational bottleneck, and the experiments demonstrate substantial memory scaling beyond the tested unrolled and CVXPYLayers implementations. Its methodological novelty lies in extending an existing DYS-JFB combination to repeated trajectory-level CBF layers and analyzing the resulting nonsmooth update. However, the hard-safety claim is not established for the finite-tolerance, discretized implementation, and the stated convergence conclusion is stronger than Theorem 12 and does not cover Algorithm 1. The theoretical differentiation argument, attribution of gains, and comparative control quality also require further substantiation.

**Strengths**

* The experiments reach a consequential scaling regime. Tables 1–2 include a system with 1,200 state and 400 control dimensions, where DYS-JFB trains within the stated GPU budget while both alternatives exceed it; this directly supports the memory-scalability argument.
* The optimization-layer construction is traceable from the original QP to implementation. Equations (7)–(14) derive the lifted feasibility sets, splitting operator, fixed point, and recovered control, while Algorithm 1 places that layer within a trajectory rollout.
* The DYS-AD comparison isolates the effect of replacing solver unrolling with JFB. Tables 1–2 show especially large GPU-memory reductions for larger quadcopter problems while retaining the same splitting-based forward formulation.
* The analysis directly targets active-set nonsmoothness rather than assuming an everywhere-differentiable safety layer. Equations (19)–(27) formulate the argument using Clarke generalized Jacobians and seek alignment with the entire Clarke subdifferential.

**Weaknesses**

* The implemented pipeline is not connected to the paper’s claimed hard-safety guarantee. Theorem 1 assumes exact continuous-time enforcement of Equation (4), whereas Algorithm 1 terminates DYS numerically, takes another fixed-point update, and advances the dynamics with explicit Euler; approximate sample-time feasibility does not imply continuous-time invariance. The claim therefore needs a sampled-data or robust-CBF result accounting for solver and integration errors, or explicit qualification accompanied by residual, margin, and inter-sample violation measurements.
* The prose overstates Theorem 12 and extends it beyond the analyzed procedure. The theorem establishes only  convergence of the stationarity measure for exact population continuous-time gradient flow, not convergence of  to a stationary point; Algorithm 1 instead uses discrete minibatch updates, numerical trajectories, and inexact fixed points. Moreover, Assumption 4’s compact parameter domain is not enforced by Equation (28) or Algorithm 1.
* The generalized differentiation identities need stronger stated justification. Equations (20)–(21) present exact generalized-Jacobian sets and equalities, although the generic Clarke chain rule provides inclusions and nonsmooth implicit differentiation requires additional regularity and compatible joint-Jacobian selections. Because Theorem 11 relies on these identities, the theorem statement should expose the conditions under which equality, rather than only an outer approximation, is valid.
* The experiments do not isolate the claimed advantage over implicit differentiation. DYS-JFB versus DYS-AD changes only the backward method, but DYS-JFB versus CVXPYLayers changes both the forward solver and differentiation method; no DYS implementation with implicit differentiation is included. Consequently, failures and resource differences relative to the implicit baseline cannot be attributed specifically to JFB rather than to solver formulation or conditioning.
* Table 1 establishes trainability but not comparative control quality. The trained objective cannot be reconstructed because running and terminal costs are reported separately without ; for the six-agent double integrator, DYS-JFB changes the regularized CVXPY result from 5.39 to 5.19 in running cost but from 0.000582 to 0.0132 in terminal cost, a factor of 22.7. Without seeds, uncertainty, or an optimal-control reference, the practical meaning of these tradeoffs and small margins is unclear.
* The reported dimensions do not characterize the actual optimization-layer scale. Assumption 3 requires full row rank for , hence , while projection onto  generally involves a state-dependent system containing ; Tables 1–2 report neither  nor sparsity, iteration counts, or factorization strategy. This prevents checking both whether the theory covers the experiments and whether the stated backward complexity reflects total layer cost.

**Suggestions for Improvement**

* Position the novelty earlier around the exact delta from the closest precursors: McKenzie et al. (2024) already combine DYS and JFB for an optimization layer, while Xiao et al. (2023) embed differentiable CBF layers in learned control. A compact comparison by forward solver, backward method, trajectory composition, and theoretical scope would clarify the contribution boundary.
* Distinguish centralized optimization-layer dimension from decentralized agent-count scalability. Qin et al. (2021) report deployment up to 1,024 agents through decentralized neural barrier certificates, so framing the present advance specifically as scaling centralized differentiable CBF-QP layers would make the dimensional claim more precise.
* Surface one representative HOCBF inequality and its initialization conditions before the experiments. The double-integrator and quadcopter results rely on HOCBFs, while the main derivation presents only the relative-degree-one construction; a concise formula would connect the method to those benchmarks.
* Reorganize the convergence section so that the adjoint, Hamiltonian gradient, and trajectory-level quantity  are defined before Equation (21), followed by an intuitive dependency map from assumptions to contraction, descent, and convergence. This would make the theoretical scope easier to evaluate in a linear reading.

**Additional Details and Minor Issues**

* Equation (1) imposes , but Equations (5)–(6) omit  unless actuator limits are included in the stacked inequalities.
* The transition from scalar  in Equations (2)–(4) to  inequalities in Equation (7) should be defined explicitly.
* Table 2 contradicts the claim of lower CPU and GPU memory “across all examples”: CVXPYLayers uses 121.9 MB GPU memory versus 564.5 MB for DYS-JFB on the single-integrator task.
* The paragraph after Assumption 4 refers to “Theorem 19,” which appears to mean Corollary 9.
* Algorithm 1, line 12 should specify which fixed-point iterate is detached by the stop-gradient operation.

**References**

McKenzie, D., Heaton, H., & Fung, S. W. (2024). Differentiating through integer linear programs with quadratic regularization and Davis–Yin splitting.  *Transactions on Machine Learning Research* .

Qin, Z., Zhang, K., Chen, Y., Chen, J., & Fan, C. (2021). Learning safe multi-agent control with decentralized neural barrier certificates.  *International Conference on Learning Representations* .

Xiao, W., Wang, T.-H., Hasani, R., Chahine, M., Amini, A., Li, X., & Rus, D. (2023). BarrierNet: Differentiable control barrier functions for learning of safe robot control.  *IEEE Transactions on Robotics, 39* (3), 2289–2307.

# Answers


**Reviewer ELP6 (6)

– Clarify novelty

Please see response to 4Ksj

– Evaluate trajectory/control quality

For large scale problems with no ground truth available, reporting average control objectives over paths as we do in Tab. 1 is standard. We will clarify this better.

– Compare against soft penalty policy with online filter

We trained the same network without projection on 6 DI exp, adding a Gaussian obstacle penalty of weight w to cost, we evaluated it raw and with a CBF-QP applied at deployment. All methods tested on 64 shared initial states against an IPOPT hard-constraint solution of the problem as GT. With a low w the raw policy collides in 98% of runs, adding the QP removes the collisions but the trajectories drift from the optimum (MSE to GT 2.9e-2 vs 1.1e-2 for ours), since the policy never saw the projections in training and is pushed outside the regime it learned. With a high w the policy is safe but conservative: MSE to GT is 0.59, 50x worse than ours. Similar differences observed in control objectives. Ours has no collisions and is closest to GT. Penalties can work, but only with careful tuning, whereas our method gives safety and near optimal solution directly, as in Xiao 2023. Will add these to draft.

– Justify Assumptions

Asm. 7 requires bounded variance which is standard in all SGD proofs ( Buttou 2018), also its the average of the h term, which is never 0. Asm 8 is time regularity, also standard but can be stronger, we numerically verify this in Fig 4.

Reviewer EuaX (6)

– Explain Assump 3 rank condition and implication

We thank the reviewer for the question. We clarify that our theory is better viewed as a sufficient but not necessary condition for convergence. Both our SI and quadcopter exp satisfy the rank condition. The DI does not, but it nonetheless converges in training, and we show it as a generalization validation of our method. In practice, constraint pruning idea from CBF literature can be used to ensure full row rank, though it is beyond the scope of this work.

– Compare time at matched cost.

Tab. 1 is showing comparable performances as low terminal cost show target reaching. We report  time in Fig. 5 for comparable results and will add more. JFB is 3-4x faster than baselines.

– Report CBF errors for relaxed baseline

We use high slack penalty for relaxed CVXPY models, we observe no CBF violation in final checkpts.

– On coupled agents and constraints

To address this and test coupling, we ran 3 double integrators with a pairwise HOCBF per pair, so each row of A(z) involves two agents and the QP cannot be split across agents. On 256 test starts, the best safe DYS-JFB checkpoint keeps every pair at least d=0.70 apart (min 0.71), performances also comparable against baselines at dt=0.02. (L, G): (4.79, 0.27) for JFB  vs (4.36, 0.25) for CVXPY, at about 20x lower in compute time per epoch for us. New results will be added to draft.

Reviewer 4Ksj (5)

– Add missing related work

Thank you for the pointers; we will add them and more.We thank the reviewer and will incorporate them and more.

– Clarify novelty

We claim a substantial, novel contribution, not a DYS+JFB reuse. We carefully designed the splitting to limit nonsmoothness reaching the primal variable, enabling our convergence proof. No existing analysis applies: McKenzie et al. handle 1 Clarke Jacobian of 1 layer; Gelphman et al. assume contraction and apply only to smooth objectives. We prove contraction & descent for ALL Clarke Jacobians/subgradients of the trajectory objective with a nonsmooth layer at EVERY time step, and gradient flow convergence using NEW TECHNIQUES (Lebourg’s MVT, Dini derivatives). No existing methods can work for the problem and to our knowledge this is the first such guarantee, significant in both results and technicality. Experiment results also show significance, our method is minimum 3-4x faster, memory efficient and the only stable one in high dim problems.

– Move key details to main text.

We agree and will move more experiment details to main text.

– Explain baseline choices and missing/failed results

Baseline choices including CVXPY are standard differentiable layers methods common in relevant works. Failures are either memory overflow or numerical blow up, we will clarify better in writing.

– Correct Tab. 2 highlighting

The current highlight is for combined memory usage, we will clarify in writing.

– Other systems

New results see ELP6 4Ksj and EuaX, we intend to do more in future work.

—-- Samy Version —--

Reviewer 4KsJ (5/3)

On related work: Thanks for the pointers! We have added them to the manuscript.

On novelty: Thank you for recognizing the convergence analysis. Our contribution is making end-to-end policy training scalable when state-dependent CBF-QPs must be solved and differentiated through at every time step of every training trajectory. We designed the splitting to limit nonsmoothness reaching the primal variable, which enables our analysis: descent for all Clarke Jacobians of the trajectory objective with a nonsmooth layer at every time step (McKenzie et al. treat one layer; Gelphman et al. assume contraction), and stationarity for gradient flow. To our knowledge this is the first such guarantee. DYS-JFB also has the lowest per-epoch time in all examples and is the only method that trains the 100-quadcopter problem. Thank you for recognizing our convergence analysis as an important contribution! We make end-to-end policy training scalable when state-dependent CBF-QPs must be solved and differentiated through at every time step of every training trajectory. DYS-JFB makes this repeated computation tractable, while our nonsmooth analysis establishes descent uniformly over Clarke subgradients and stationarity guarantees for idealized gradient flow. We have made this explicit in the manuscript.

On experimental details: We have added baseline details in the main text. The reason why the Quadcopter (100,3) only has results for the proposed method is precisely because no other method can handle this dimensionality - this is the point of this work.

On more complex experiments: Thanks for pointing us to these experiments!

Pm

Reviewer 8Gh9 (5)

– Explain discrete vs continuous time difference

Our theory is in continuous time, and the gap to the discrete implementation is real, as in all sampled-data CBF theory. In the reported experiments we saw no violations. In an extra coupled test, a late network violated at dt=0.2 but was safe at dt<=0.1, so step size and solver tolerance are the levers. Thm 12 is for gradient flow: the nonsmooth CBF layer at every time step rules out standard tools, so we use nonsmooth analysis, and a discrete-step result is open. Training converged in all reported runs. We will state this scope clearly in the revision. All our theory is in continuous time. Under discretization the gap is real, as is standard in CBF theory, we inherit the guarantee in continuous time. Note that we do not observe safety violations in any numerical testing. Thm 12 establishes convergence in gradient flow. Due to nonsmooth CBF layer in every time step rendering most analysis tools infeasible, we have to circumvent them with new techniques from nonsmooth analysis, GD convergence is to our knowledge, simply impossible. Similarly, we do not observe convergence difficulties in actual training. Adjusting solver tolerance or step size can help if violations occur. I will clarify these better in revision.

– Clarify terminology “end-to-end”

We refer to “end-to-end” as the full pipeline shared between training and deployment without pre or post processing steps, we are open to adjusting the framing as needed.

– Report seeded run statistics

We report these numbers on page 22 of the appendix, more details on baselines and hyperparameter sweeps will be added.

– Discuss L2O and DC3

DC3 only gives approximate feasibility, not hard safety guarantee; L2O learns solvers, not safe control policies. They aren’t comparable baselines, but we’ll discuss both in related work. More relevant comparisons are also added, please see response to ELP6.

– Clarify novelty

Please see response to 4Ksj

AI Reviewer

We appreciate your comments, majority of these questions have answers in the appendix, we will further clarify and improve our writing.

– Clarify discrete vs continuous time difference

Please see response to 8Gh9

– Properly state theory scope

We mention this when introducing assumps, we will polish writing to better reflect scope.

– Justify Clarke Jacobian selection in proof

We prove against all Clarke Jacobians, hence no selection is needed and this makes our results strong and novel.

– Isolate JFB’s and DYS's benefit

We cover these exact points under final discussion on page 22 and 23 of the appendix.

– Report implementation details and hyperparameters

These are all reported in the appendix from page 18 onwards, we will incorporate them to the main text as best as we can.

– Clarify Assumption 3 rank condition and relevant claims

Please see response to EuaX

– Differentiate from related works

We believe we did extensively in method design, literature review and appendix, we will further clarify.

– Define CBF config and initialization used in experiments.

All details are included in the appendix, we will clarify in the main text as well.

– Clarify proof notation and structure

We will adjust and polish as recommended, thank you.

– Fix minor consistency issues

We thank the reviewer and will address them.

AC message:

We thank the reviewers and the AC for their time and constructive feedback. In summary: We thank the reviewers and AC for overseeing the review process. We believe we have extremely strong responses to all questions raised, which we summarize as follows:

1. Contribution: we prove descent for all Clarke Jacobians of a trajectory objective with a nonsmooth safety layer at every time step, and DYS-JFB makes this training scalable: 100 quadcopters (1,200 states) train within the GPU budget where the baselines do not. See our response to 4Ksj. We believe the reviewers misread the importance of our work. Our theory is novel and strong, leveraging new techniques to derive new and strong results, its not a simple combination of existing work. Numerically we solve problems previously impossible. Please refer to response to 4Ksj for more details.
2. We added two experiments: a soft-penalty policy with a test-time filter as an extra baseline (see ELP6), and a test with pairwise constraints coupling agents (see EuaX). Results are consistent with the paper. Experiments on new robot models are future work. We have added two numerical experiments and results to further strengthen our claims and address reviewer concerns. Including additional baseline comparisons showing the advantages of our method against soft penalty policies with test time filters only, and a new setting where we test coupling between agents with nonstationary obstacles, results also consistent with the paper. See responses to 4Ksj and EuaX. As we have covered many popular dynamics in path planning, experiments on new models while possible, is beyond the scope of this work.
3. The remaining questions are mostly about clarity, and many are answered in the appendix. We have and will improve the writing to clarify the contributions and scope, and to address points of confusion. We will also move implementation details from the appendix to the main text as far as space allows.
4. The remaining questions are relatively minor and most have answers in the appendix. We have and will improve the writing in clarifying the contributions and scope, and address points of confusion. We will also try our best to incorporate implementation details from appendix to main text.

We believe all points are well addressed with no outstanding concerns, weI thank you again for your time and effort.

**
