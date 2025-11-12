import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from scipy.special import gammaln, i0e
from scipy.optimize import minimize
from tqdm.auto import tqdm


# =========================
# Helpers (IQR/MAD, utils)
# =========================
def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _iqr(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _contig_lengths(ids):
    lengths, last, c = [], None, 0
    for v in ids:
        if last is None or v == last:
            c += 1
        else:
            lengths.append(c)
            c = 1
        last = v
    if c > 0:
        lengths.append(c)
    return lengths


def _aic_from_loglik(loglik, n_states, angle_model: str):
    # params: start (S-1) + trans S(S-1) + emissions per state
    # step gamma: (k_step, theta_step) = 2
    # angle vm: (mu, kappa) = 2  |  angle gamma: (k_angle, theta_angle) = 2
    k_emiss = 4  # both cases = 4 per state
    k = (n_states - 1) + n_states * (n_states - 1) + k_emiss * n_states
    return 2 * k - 2 * loglik


# =========================
# Parameter ranges (for init)
# =========================
def compute_parameter_ranges(
    df: pd.DataFrame,
    angle_var: str,
    use_vm: bool = False,
) -> Dict[str, Tuple[float, float]]:
    step = df["step"].to_numpy(float)
    step_iqr = _iqr(step)
    step_median = np.nanmedian(step)
    step_mean_range = (max(1e-6, step_median - step_iqr), step_median + step_iqr)
    step_sd_range = (0.1, max(0.2, 1.5 * _mad(step)))

    angle = df[angle_var].to_numpy(float)
    angle_median = np.nanmedian(angle)
    angle_iqr = _iqr(angle)
    angle_mean_range = (max(0.0, angle_median - angle_iqr), angle_median + angle_iqr)
    angle_sd_range = (0.1, max(0.2, 1.5 * _mad(angle)))

    angle_conc_range = None
    if use_vm:
        c = np.nanmean(np.cos(angle))
        s = np.nanmean(np.sin(angle))
        R = np.sqrt(c**2 + s**2) if np.isfinite(c) and np.isfinite(s) else 0.0
        if R > 0:
            if R < 0.53:
                kappa = 2 * R + R**3 + (5 * R**5) / 6
            elif R < 0.85:
                kappa = -0.4 + 1.39 * R + 0.43 / (1 - R)
            else:
                kappa = 1 / (R**3 - 4 * R**2 + 3 * R)
            angle_conc_range = (1.0, float(min(max(5.0, 2 * kappa + 1.0), 50.0)))
        else:
            angle_conc_range = (1.0, 15.0)

    return dict(
        step_mean_range=step_mean_range,
        step_sd_range=step_sd_range,
        angle_mean_range=angle_mean_range,
        angle_sd_range=angle_sd_range,
        angle_conc_range=angle_conc_range,
    )


# =========================
# Emission log-densities
# =========================
def logpdf_gamma(x, k, theta):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-12, None)
    k = np.clip(k, 1e-8, None)
    theta = np.clip(theta, 1e-8, None)
    return (k - 1) * np.log(x) - (x / theta) - k * np.log(theta) - gammaln(k)


def logpdf_vonmises(phi, mu, kappa):
    kappa = np.clip(kappa, 0.0, None)
    # stable log C = -log(2πI0(k)), with i0e handling exp(k) internally
    logC = -(np.log(2 * np.pi) + (np.log(i0e(kappa)) + np.abs(kappa)))
    return kappa * np.cos(phi - mu) + logC


# =========================
# Forward–Backward & Viterbi
# =========================
def forward_backward(log_lik, startprob, transmat, lengths):
    S = startprob.shape[0]
    N = log_lik.shape[0]
    alpha = np.zeros((N, S))
    scales = np.zeros(N)
    idx = 0
    for L in lengths:
        a0 = np.log(startprob + 1e-15) + log_lik[idx]
        c0 = np.logaddexp.reduce(a0)
        alpha[idx] = a0 - c0
        scales[idx] = c0
        for t in range(idx + 1, idx + L):
            at = alpha[t - 1][:, None] + np.log(transmat + 1e-15)
            a = np.logaddexp.reduce(at, axis=0) + log_lik[t]
            ct = np.logaddexp.reduce(a)
            alpha[t] = a - ct
            scales[t] = ct
        idx += L
    beta = np.zeros((N, S))
    idx = 0
    for L in lengths:
        for t in range(idx + L - 2, idx - 1, -1):
            btn = (beta[t + 1] + log_lik[t + 1])[None, :] + np.log(transmat + 1e-15)
            b = np.logaddexp.reduce(btn, axis=1)
            beta[t] = b - np.logaddexp.reduce(b)
        idx += L
    g = alpha + beta
    g -= np.max(g, axis=1, keepdims=True)
    g = np.exp(g)
    g /= g.sum(axis=1, keepdims=True)
    xisum = np.zeros_like(transmat)
    idx = 0
    for L in lengths:
        for t in range(idx, idx + L - 1):
            M = alpha[t][:, None] + np.log(transmat + 1e-15) + log_lik[t + 1][None, :] + beta[t + 1][None, :]
            M -= np.max(M)
            P = np.exp(M)
            P /= P.sum()
            xisum += P
        idx += L
    total_ll = float(scales.sum())
    return g, xisum, total_ll


def viterbi(log_lik, startprob, transmat, lengths):
    S = startprob.shape[0]
    N = log_lik.shape[0]
    path = np.empty(N, dtype=int)
    idx = 0
    for L in lengths:
        delta = np.log(startprob + 1e-15) + log_lik[idx]
        psi = np.zeros((L, S), dtype=int)
        deltas = np.zeros((L, S))
        deltas[0] = delta
        for t in range(1, L):
            prev = deltas[t - 1][:, None] + np.log(transmat + 1e-15)
            psi[t] = np.argmax(prev, axis=0)
            deltas[t] = np.max(prev, axis=0) + log_lik[idx + t]
        sT = int(np.argmax(deltas[L - 1]))
        seq = np.empty(L, dtype=int)
        seq[L - 1] = sT
        for t in range(L - 2, -1, -1):
            seq[t] = psi[t + 1, seq[t + 1]]
        path[idx : idx + L] = seq
        idx += L
    return path


# =========================
# HMM class: Gamma(step) + VM(angle) or Gamma(abs(angle))
# =========================
class GammaHMM:
    """
    angle_model:
        - "vm"       : use angle (radians) ~ von Mises
        - "absgamma" : use |angle| ~ Gamma
    """

    def __init__(
        self,
        n_states=2,
        angle_model="vm",
        max_iter=200,
        tol=1e-4,
        method="L-BFGS-B",
        random_state=0,
        stationary=True,
        angle_bias: Optional[Tuple[float, float]] = None,
    ):
        self.S = n_states
        self.angle_model = angle_model
        self.max_iter = max_iter
        self.tol = tol
        self.method = method
        self.rng = np.random.default_rng(random_state)
        self.stationary = stationary
        self.angle_bias = angle_bias  # only used if VM

    def _init_params(self, step, angle):
        S = self.S
        self.startprob_ = np.full(S, 1.0 / S)
        diag = 0.95 if self.stationary else 0.90
        self.transmat_ = np.full((S, S), (1 - diag) / (S - 1))
        np.fill_diagonal(self.transmat_, diag)

        # Step Gamma init
        s_med, s_iqr = np.nanmedian(step), _iqr(step)
        mean1, mean2 = max(1e-3, s_med - 0.3 * s_iqr), max(1e-3, s_med + 0.3 * s_iqr)
        var_guess = (max(1e-6, 0.5 * s_iqr)) ** 2 + 1e-6

        def ktheta(m, v):
            k = m**2 / v
            th = v / m
            return max(1e-3, k), max(1e-3, th)

        k1, t1 = ktheta(mean1, var_guess)
        k2, t2 = ktheta(mean2, var_guess)
        self.k_step_ = np.array([k1, k2][:S], dtype=float)
        self.theta_step_ = np.array([t1, t2][:S], dtype=float)

        if self.angle_model == "vm":
            if self.angle_bias is not None and len(self.angle_bias) >= S:
                self.mu_ = np.array([_wrap_angle(m) for m in self.angle_bias[:S]], dtype=float)
            else:
                self.mu_ = self.rng.uniform(-np.pi, np.pi, size=S)
            self.kappa_ = self.rng.uniform(0.5, 5.0, size=S)
        else:
            # abs(angle) as gamma
            a_med, a_iqr = np.nanmedian(angle), _iqr(angle)
            ma1, ma2 = max(1e-3, a_med - 0.3 * a_iqr), max(1e-3, a_med + 0.3 * a_iqr)
            v_ag = (max(1e-6, 0.5 * a_iqr)) ** 2 + 1e-6
            ka1, tha1 = ktheta(ma1, v_ag)
            ka2, tha2 = ktheta(ma2, v_ag)
            self.k_ang_ = np.array([ka1, ka2][:S], dtype=float)
            self.theta_ang_ = np.array([tha1, tha2][:S], dtype=float)

    def _loglik(self, step, angle):
        N = len(step)
        L = np.zeros((N, self.S))
        if self.angle_model == "vm":
            for s in range(self.S):
                L[:, s] = logpdf_gamma(step, self.k_step_[s], self.theta_step_[s]) + logpdf_vonmises(
                    angle, self.mu_[s], self.kappa_[s]
                )
        else:
            for s in range(self.S):
                L[:, s] = logpdf_gamma(step, self.k_step_[s], self.theta_step_[s]) + logpdf_gamma(
                    angle, self.k_ang_[s], self.theta_ang_[s]
                )
        return L

    def _mstep_emissions(self, step, angle, gamma):
        for s in range(self.S):
            w = gamma[:, s]
            w = w / (w.sum() + 1e-12)
            if self.angle_model == "vm":

                def nll(p):
                    logk_s, logth_s, mu, logkp1 = p
                    k_s = np.exp(logk_s)
                    th_s = np.exp(logth_s)
                    kp = np.exp(logkp1) - 1.0
                    return -(w * (logpdf_gamma(step, k_s, th_s) + logpdf_vonmises(angle, mu, kp))).sum()

                x0 = np.array(
                    [np.log(self.k_step_[s]), np.log(self.theta_step_[s]), self.mu_[s], np.log(self.kappa_[s] + 1.0)]
                )
                res = minimize(nll, x0, method=self.method, options=dict(maxiter=500, disp=False))
                if res.success:
                    logk_s, logth_s, mu, logkp1 = res.x
                    self.k_step_[s] = np.exp(logk_s)
                    self.theta_step_[s] = np.exp(logth_s)
                    self.mu_[s] = _wrap_angle(mu)
                    self.kappa_[s] = np.exp(logkp1) - 1.0
            else:

                def nll(p):
                    logk_s, logth_s, logk_a, logth_a = p
                    k_s = np.exp(logk_s)
                    th_s = np.exp(logth_s)
                    k_a = np.exp(logk_a)
                    th_a = np.exp(logth_a)
                    return -(w * (logpdf_gamma(step, k_s, th_s) + logpdf_gamma(angle, k_a, th_a))).sum()

                x0 = np.array(
                    [
                        np.log(self.k_step_[s]),
                        np.log(self.theta_step_[s]),
                        np.log(self.k_ang_[s]),
                        np.log(self.theta_ang_[s]),
                    ]
                )
                res = minimize(nll, x0, method=self.method, options=dict(maxiter=500, disp=False))
                if res.success:
                    logk_s, logth_s, logk_a, logth_a = res.x
                    self.k_step_[s] = np.exp(logk_s)
                    self.theta_step_[s] = np.exp(logth_s)
                    self.k_ang_[s] = np.exp(logk_a)
                    self.theta_ang_[s] = np.exp(logth_a)

    def fit(self, df, id_col="Session", step_col="step", angle_col="angle"):
        ids = df[id_col].to_numpy()
        step = df[step_col].to_numpy(float)
        if self.angle_model == "vm":
            ang = df[angle_col].to_numpy(float)
        else:
            ang = np.abs(df[angle_col].to_numpy(float))
        mask = np.isfinite(step) & np.isfinite(ang)
        ids, step, ang = ids[mask], step[mask], ang[mask]
        lengths = _contig_lengths(ids)

        self._init_params(step, ang)
        prev_ll = -np.inf
        for _ in range(self.max_iter):
            logL = self._loglik(step, ang)
            gamma, xisum, ll = forward_backward(logL, self.startprob_, self.transmat_, lengths)

            # start probs (average of posteriors at sequence starts)
            start = np.zeros(self.S)
            idx = 0
            for L in lengths:
                start += gamma[idx]
                idx += L
            self.startprob_ = (start / len(lengths)).clip(1e-12)
            self.startprob_ /= self.startprob_.sum()

            # transition matrix
            A = xisum.clip(1e-12)
            self.transmat_ = A / A.sum(axis=1, keepdims=True)

            # emissions with chosen optimizer
            self._mstep_emissions(step, ang, gamma)

            if ll - prev_ll < self.tol:
                break
            prev_ll = ll

        self.posterior_ = gamma
        self.states_post_ = gamma.argmax(axis=1)
        self.viterbi_ = viterbi(self._loglik(step, ang), self.startprob_, self.transmat_, lengths)
        self.lengths_ = lengths
        self.loglik_ = float(ll)
        self.AIC_ = _aic_from_loglik(self.loglik_, self.S, self.angle_model)

        # State diagnostics for objective
        step_means = np.zeros(self.S)
        for s in range(self.S):
            w = self.posterior_[:, s]
            w /= w.sum() + 1e-12
            step_means[s] = np.sum(w * step)
        self.step_means_ = step_means
        if self.angle_model == "vm":
            self.turn_metric_ = 1.0 / (getattr(self, "kappa_", np.ones(self.S)) + 1e-12)
        else:
            # for gamma |angle|, larger mean ≈ more turning
            ang_means = np.zeros(self.S)
            for s in range(self.S):
                w = self.posterior_[:, s]
                w /= w.sum() + 1e-12
                ang_means[s] = np.sum(w * ang)
            self.turn_metric_ = ang_means
        return self

    def score(self, df, id_col="Session", step_col="step", angle_col="angle"):
        ids = df[id_col].to_numpy()
        step = df[step_col].to_numpy(float)
        ang_raw = df[angle_col].to_numpy(float)
        ang = ang_raw if self.angle_model == "vm" else np.abs(ang_raw)
        m = np.isfinite(step) & np.isfinite(ang)
        ids, step, ang = ids[m], step[m], ang[m]
        lengths = _contig_lengths(ids)
        _, _, ll = forward_backward(
            self._loglik(step, ang),
            self.startprob_,
            self.transmat_,
            lengths,
        )
        return float(ll)

    def reorder_states(self, order: List[int]):
        order = np.asarray(order, dtype=int)
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[order][:, order]
        self.k_step_ = self.k_step_[order]
        self.theta_step_ = self.theta_step_[order]
        if self.angle_model == "vm":
            self.mu_ = self.mu_[order]
            self.kappa_ = self.kappa_[order]
        else:
            self.k_ang_ = self.k_ang_[order]
            self.theta_ang_ = self.theta_ang_[order]
        if hasattr(self, "posterior_"):
            self.posterior_ = self.posterior_[:, order]
            self.states_post_ = np.argmax(self.posterior_, axis=1)
        if hasattr(self, "viterbi_"):
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            self.viterbi_ = inv[self.viterbi_]


# =========================
# Summaries & Orchestrator
# =========================
def _summarize(model: GammaHMM) -> Dict[str, Any]:
    d = {
        "optimizer": model.method,
        "n_states": model.S,
        "angle_type": ("von Mises" if model.angle_model == "vm" else "|angle| ~ Gamma"),
        "loglik": model.loglik_,
        "AIC": model.AIC_,
        "startprob": np.round(model.startprob_, 4).tolist(),
        "transmat": np.round(model.transmat_, 4).tolist(),
        "step_k": np.round(model.k_step_, 3).tolist(),
        "step_theta": np.round(model.theta_step_, 3).tolist(),
        "step_means": np.round(model.step_means_, 4).tolist(),
        "turn_metric": np.round(model.turn_metric_, 4).tolist(),
        "behavioral_constraint_met": _enforce_or_reject(model),
    }
    if model.angle_model == "vm":
        d["vm_mu"] = np.round(model.mu_, 3).tolist()
        d["vm_kappa"] = np.round(model.kappa_, 3).tolist()
    else:
        d["ang_k"] = np.round(model.k_ang_, 3).tolist()
        d["ang_theta"] = np.round(model.theta_ang_, 3).tolist()
    return d


def print_hmm_summary(model_summary: Dict[str, Any], model: GammaHMM):
    s = model_summary
    print("\n Best Model Characteristics:")
    print(f"• Angle type: {s['angle_type']}")
    print(f"• Optimizer: {s['optimizer']}")
    print(f"• AIC: {s['AIC']:.2f} | logLik: {s['loglik']:.2f}")
    print(f"• Start probs: {s['startprob']}")
    print("• Transmat:\n", np.array(s["transmat"]))
    print(f"• Step Means: {s['step_means']}")
    if model.angle_model == "vm":
        print(f"• VM mu: {s['vm_mu']}")
        print(f"• VM kappa: {s['vm_kappa']}")
    else:
        print(f"• |angle| Gamma k: {s['ang_k']}")
        print(f"• |angle| Gamma theta: {s['ang_theta']}")
    print(f"• Met behavioral constraints: {s['behavioral_constraint_met']}")
    print("• Final state ordering: State 1 = low step + high turn; State 2 = high step + low turn\n")


@dataclass
class BestResult:
    model: GammaHMM
    summary: Dict[str, Any]
    records: pd.DataFrame
    data: pd.DataFrame

    def save(self, config: Dict[str, Any]):
        save_compass_level_1_results(config, self)


def _enforce_or_reject(model: GammaHMM) -> bool:
    step = model.step_means_
    turn = model.turn_metric_
    s1 = int(np.argmax(-(step - step.mean()) / (step.std() + 1e-12) + (turn - turn.mean()) / (turn.std() + 1e-12)))
    s2 = 1 - s1 if model.S == 2 else [i for i in range(model.S) if i != s1][0]
    if not (step[s1] <= step[s2] and turn[s1] >= turn[s2]):
        return False
    if not (s1 == 0 and s2 == 1):
        model.reorder_states([s1, s2])
    return True


def fit_best_hmm(
    preproc_df: pd.DataFrame,
    n_states: int = 2,
    n_repetitions: int = 20,
    opt_methods: list[str] = ["BFGS", "L-BFGS-B", "Nelder-Mead", "Powell"],
    max_iter: int = 200,
    use_abs_angle: tuple[bool, ...] = (True, False),  # True => |angle|~Gamma ; False => angle~VM
    stationary_flag: str | bool = "auto",
    use_data_driven_ranges: bool = True,
    angle_mean_biased: tuple[float, float] = (np.pi / 2, 0.0),  # only for VM branch
    session_col: str = "Session",
    seed: int = 123,
    enforce_behavioral_constraints: bool = True,
    show_progress: bool = True,
) -> BestResult:

    rng = np.random.default_rng(seed)
    base = preproc_df.copy()

    if stationary_flag == "auto":
        lens = _contig_lengths(base[session_col].to_numpy())
        stationary = bool(np.median(lens) >= 100)
    else:
        stationary = bool(stationary_flag)

    records = []
    candidates: List[Tuple[GammaHMM, pd.DataFrame, Dict[str, Any]]] = []

    total = len(use_abs_angle) * len(opt_methods) * n_repetitions
    pbar = tqdm(total=total, desc="Gamma/VM HMM search", leave=False) if show_progress else None

    for abs_flag in use_abs_angle:
        angle_var = "angle_abs" if abs_flag else "angle"
        df_use = base.copy()
        if abs_flag:
            df_use["angle_abs"] = np.abs(df_use["angle"].to_numpy(float))
        if use_data_driven_ranges:
            _ = compute_parameter_ranges(df_use, angle_var, use_vm=(not abs_flag))

        for opt in opt_methods:
            for it in range(n_repetitions):
                this_seed = int(rng.integers(0, 10_000_000))
                try:
                    model = GammaHMM(
                        n_states=n_states,
                        angle_model=("absgamma" if abs_flag else "vm"),
                        max_iter=max_iter,
                        tol=1e-4,
                        method=opt,
                        random_state=this_seed,
                        stationary=stationary,
                        angle_bias=(angle_mean_biased if not abs_flag else None),
                    )
                    model.fit(df_use, id_col=session_col, step_col="step", angle_col="angle")

                    # Strict behavioral gate + enforce ordering
                    if enforce_behavioral_constraints and not _enforce_or_reject(model):
                        continue

                    summ = _summarize(model)
                    records.append(
                        {
                            "abs_angle": abs_flag,
                            "optimizer": opt,
                            "seed": this_seed,
                            "AIC": summ["AIC"],
                            "loglik": summ["loglik"],
                        }
                    )

                    # attach states/posteriors back to the original df rows
                    out = preproc_df.copy()
                    out["HMM_State"] = np.nan
                    for s in range(model.S):
                        out[f"Post_Prob_{s+1}"] = np.nan
                    # valid mask (same as fit)
                    m = np.isfinite(preproc_df["step"].to_numpy(float)) & np.isfinite(
                        (
                            np.abs(preproc_df["angle"].to_numpy(float))
                            if abs_flag
                            else preproc_df["angle"].to_numpy(float)
                        )
                    )
                    out.loc[m, "HMM_State"] = (model.viterbi_ + 1).astype(int)  # 1/2 labeling
                    # for s in range(model.S):      # Optional add the State Probs
                    #     out.loc[m, f"Post_Prob_{s+1}"] = model.posterior_[:, s]

                    candidates.append((model, out, summ))

                except Exception:
                    pass
                finally:
                    if pbar:
                        pbar.update(1)

    if pbar:
        pbar.close()
    if len(candidates) == 0:
        raise RuntimeError("No valid models met the behavioral objective across configurations.")

    # Select best by AIC
    best_idx = int(np.argmin([c[2]["AIC"] for c in candidates]))
    best_model, df_with_states, best_summary = candidates[best_idx]
    rec_df = pd.DataFrame.from_records(records).sort_values(
        ["AIC", "optimizer"],
        ascending=[True, True],
        kind="mergesort",
    )
    return BestResult(
        model=best_model,
        summary=best_summary,
        records=rec_df,
        data=df_with_states,
    )


def save_compass_level_1_results(
    config: Dict[str, Any],
    results: BestResult,
):
    """
    Save CoMPASS Level 1 HMM results to disk.

    Exports:
        - model_summary.json: Model parameters and statistics
        - data_with_states.csv: Full dataset with HMM state assignments
        - model_selection_records.csv: All candidate models tried during fitting
        - fitted_model.joblib: Serialized GammaHMM object (if joblib available)

    Parameters
    ----------
    config : dict
        Project configuration containing 'project_path_full'
    results : BestResult
        Result object from fit_best_hmm()
    """
    dest_dir = Path(config["project_path_full"]) / "results" / "compass_level_1"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save model summary as JSON
    summary_path = dest_dir / "model_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results.summary, f, indent=2)
    print(f"Saved model summary: {summary_path.name}")

    # 2. Save data with state assignments
    data_path = dest_dir / "data_with_states.csv"
    results.data.to_csv(data_path, index=False)
    print(f"Saved data with states: {data_path.name}")

    # 3. Save model selection records
    records_path = dest_dir / "model_selection_records.csv"
    results.records.to_csv(records_path, index=False)
    print(f"Saved model selection records: {records_path.name}")

    # 4. Save fitted model object (if joblib available)
    if joblib is not None:
        model_path = dest_dir / "fitted_model.joblib"
        joblib.dump(results.model, model_path)
        print(f"Saved fitted model: {model_path.name}")
    else:
        print(f"Skipped model serialization (joblib not installed)")

    print(f"\n All CoMPASS Level 1 results saved to: {dest_dir}")

