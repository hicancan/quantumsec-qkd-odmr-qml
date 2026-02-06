from __future__ import annotations
import argparse, os, time
from dataclasses import asdict
from typing import Dict, Any
import numpy as np

from qmsl.config import GlobalConfig
from qmsl.utils import ensure_dir, set_global_seed, write_json, merge_dicts

from qmsl.datasets.bb84_numpy import generate_bb84_dataset
from qmsl.datasets.bb84_qutip_channel import simulate_bb84_session_qutip
from qmsl.datasets.odmr_qutip import generate_odmr_dataset, sympy_verify_eigs_example

from qmsl.features.odmr_features import ODMRFeatureConfig, odmr_to_features
from qmsl.models.baselines import fit_threshold_by_auc
from qmsl.models.classical_ml import train_classical_models
from qmsl.models.qsvm import train_qsvm_precomputed
from qmsl.kernels.backend_kernel_interface import KernelConfig
from qmsl.eval.metrics import evaluate_binary, fpr_at_tpr
from qmsl.eval.plots import save_confusion_matrix, save_roc_curve, save_scatter, save_line


def cmd_env_check(_args) -> None:
    import platform, sys
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    # optional imports
    for name in ["numpy", "scipy", "pandas", "matplotlib", "sklearn", "sympy", "qutip", "pennylane", "qiskit", "qiskit_aer", "cirq"]:
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "unknown")
        except Exception as e:
            info[name] = f"NOT_AVAILABLE: {e.__class__.__name__}"
    try:
        import cudaq
        info["cudaq"] = getattr(cudaq, "__version__", "unknown")
    except Exception as e:
        info["cudaq"] = f"NOT_AVAILABLE: {e.__class__.__name__}"
    print(info)

def _split_train_test(X, y, rng, test_frac=0.25):
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(n * test_frac)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], train_idx, test_idx

def cmd_bb84(args) -> Dict[str, Any]:
    cfg = GlobalConfig(seed=args.seed)
    rng = set_global_seed(args.seed)

    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(os.path.join("docs", "figures"))
    print(f"[BB84] Generating dataset... backend={args.backend}")

    if args.backend == "numpy":
        X, y, meta = generate_bb84_dataset(
            n_sessions=args.sessions,
            n_bits=args.bits,
            rng=rng,
            p_eve_range=(cfg.bb84_p_eve_min, cfg.bb84_p_eve_max),
            p_chan_range=(cfg.bb84_p_chan_min, cfg.bb84_p_chan_max),
            eve_threshold=args.eve_threshold,
        )
    elif args.backend == "qutip":
        # small-scale validation dataset
        X_list = []
        y_list = []
        p_eve_list = []
        for _ in range(args.sessions):
            p_eve = float(rng.uniform(cfg.bb84_p_eve_min, cfg.bb84_p_eve_max))
            p_depol = float(rng.uniform(0.0, 0.08))
            p_dephase = float(rng.uniform(0.0, 0.08))
            feats, label, m = simulate_bb84_session_qutip(
                n_bits=args.bits,
                p_eve=p_eve,
                p_depol=p_depol,
                p_dephase=p_dephase,
                rng=rng,
                eve_threshold=args.eve_threshold,
            )
            X_list.append(feats.as_vector())
            y_list.append(label)
            p_eve_list.append(m["p_eve"])
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=int)
        meta = {"p_eve": np.array(p_eve_list, dtype=float)}
    else:
        raise ValueError("Unknown backend")

    X_train, y_train, X_test, y_test, train_idx, test_idx = _split_train_test(X, y, rng, test_frac=0.25)

    # Baseline threshold on qber feature 0
    thr = fit_threshold_by_auc(X_train, y_train, feature_index=0)
    thr_scores = thr.predict_proba(X_test)[:, 1]
    res_thr = evaluate_binary(y_test, thr_scores)

    # Classical ML
    classical = train_classical_models(X_train, y_train)
    rbf_scores = classical["svm_rbf"].model.predict_proba(X_test)[:, 1]
    res_rbf = evaluate_binary(y_test, rbf_scores)

    # QSVM with chosen backend kernel
    kcfg = KernelConfig(n_qubits=args.n_qubits, entangle=args.entangle, shots=args.shots, seed=args.seed)
    qsvm_model, qsvm_scores = train_qsvm_precomputed(X_train, y_train, X_test, backend=args.kernel_backend, cfg=kcfg, C=2.0)
    res_qsvm = evaluate_binary(y_test, qsvm_scores)

    # Plots
    save_roc_curve(res_qsvm, os.path.join(fig_dir, "bb84_roc_qsvm.png"), title="BB84 QSVM ROC")
    save_confusion_matrix(res_qsvm.cm, os.path.join(fig_dir, "bb84_cm_qsvm.png"), title="BB84 QSVM Confusion Matrix")

    # QBER vs SKR scatter (from full dataset)
    qber = X[:, 0]
    skr = X[:, -1]
    save_scatter(qber, skr, os.path.join(fig_dir, "bb84_qber_skr.png"),
                 title="BB84: QBER vs Secret Key Rate (SKR)", xlabel="QBER", ylabel="SKR (proxy)" )

    # Metrics bundle
    metrics = {
        "bb84": {
            "backend": args.backend,
            "kernel_backend": args.kernel_backend,
            "threshold": {"auc": res_thr.auc, "acc": res_thr.acc},
            "svm_rbf": {"auc": res_rbf.auc, "acc": res_rbf.acc},
            "qsvm": {"auc": res_qsvm.auc, "acc": res_qsvm.acc},
            "fpr@tpr95_qsvm": fpr_at_tpr(y_test, qsvm_scores, 0.95),
        }
    }
    write_json(os.path.join(out_dir, "bb84_metrics.json"), metrics)
    return metrics

def cmd_odmr(args) -> Dict[str, Any]:
    rng = set_global_seed(args.seed)
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(os.path.join("docs", "figures"))

    print("[ODMR] SymPy check:", sympy_verify_eigs_example())

    X_spec, y, meta = generate_odmr_dataset(
        n_per_class=args.n_per_class,
        n_freq=args.n_freq,
        f_center=args.f_center,
        span=args.span,
        B0_T=args.B0,
        B1_T=args.B1,
        noise_std=args.noise,
        rng=rng,
    )

    # plot mean spectra
    freqs = meta["freqs"]
    mean0 = X_spec[y==0].mean(axis=0)
    mean1 = X_spec[y==1].mean(axis=0)
    # save mean spectra plot
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(freqs, mean0, label="benign" )
    plt.plot(freqs, mean1, label="perturbed" )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized fluorescence")
    plt.title("ODMR mean spectra")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "odmr_mean_spectra.png"), dpi=180)
    plt.close()

    # features (PCA -> n_qubits dims)
    X, pca = odmr_to_features(X_spec, ODMRFeatureConfig(n_components=args.n_qubits))

    X_train, y_train, X_test, y_test, *_ = _split_train_test(X, y, rng, test_frac=0.25)

    # Classical baseline
    classical = train_classical_models(X_train, y_train)
    rbf_scores = classical["svm_rbf"].model.predict_proba(X_test)[:, 1]
    res_rbf = evaluate_binary(y_test, rbf_scores)

    # QSVM
    kcfg = KernelConfig(n_qubits=args.n_qubits, entangle=args.entangle, shots=args.shots, seed=args.seed)
    _, qsvm_scores = train_qsvm_precomputed(X_train, y_train, X_test, backend=args.kernel_backend, cfg=kcfg, C=2.0)
    res_qsvm = evaluate_binary(y_test, qsvm_scores)

    save_roc_curve(res_qsvm, os.path.join(fig_dir, "odmr_roc_qsvm.png"), title="ODMR QSVM ROC")
    save_confusion_matrix(res_qsvm.cm, os.path.join(fig_dir, "odmr_cm_qsvm.png"), title="ODMR QSVM Confusion Matrix")

    metrics = {
        "odmr": {
            "kernel_backend": args.kernel_backend,
            "svm_rbf": {"auc": res_rbf.auc, "acc": res_rbf.acc},
            "qsvm": {"auc": res_qsvm.auc, "acc": res_qsvm.acc},
        }
    }
    write_json(os.path.join(out_dir, "odmr_metrics.json"), metrics)
    return metrics

def cmd_kernel_bench(args) -> Dict[str, Any]:
    rng = set_global_seed(args.seed)
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(os.path.join("docs", "figures"))
    from qmsl.kernels.backend_kernel_interface import kernel_matrix, KernelConfig

    # small benchmark set
    X = rng.normal(0, 1, size=(args.n, args.n_qubits))
    cfg = KernelConfig(n_qubits=args.n_qubits, entangle=args.entangle, shots=args.shots, seed=args.seed)

    K_pl = kernel_matrix(X, X, backend="pennylane", cfg=cfg)
    K_qk = kernel_matrix(X, X, backend="qiskit", cfg=cfg)
    K_cq = kernel_matrix(X, X, backend="cirq", cfg=cfg)
    # cudaq is stochastic; optional
    try:
        K_cu = kernel_matrix(X, X, backend="cudaq", cfg=cfg)
    except Exception as e:
        K_cu = None

    max_diff = float(np.max(np.abs(K_pl - K_qk)))
    max_diff2 = float(np.max(np.abs(K_pl - K_cq)))
    out = {
        "kernel_bench": {
            "max_abs_diff_pl_vs_qiskit": max_diff,
            "max_abs_diff_pl_vs_cirq": max_diff2,
            "cudaq_available": K_cu is not None,
        }
    }
    if K_cu is not None:
        out["kernel_bench"]["max_abs_diff_pl_vs_cudaq"] = float(np.max(np.abs(K_pl - K_cu)))

    write_json(os.path.join(out_dir, "kernel_bench.json"), out)
    return out

def cmd_run_all(args) -> None:
    out_dir = ensure_dir(args.out)

    bb84 = cmd_bb84(argparse.Namespace(
        seed=args.seed, out=out_dir, backend=args.bb84_backend,
        sessions=args.bb84_sessions, bits=args.bb84_bits,
        eve_threshold=args.eve_threshold,
        kernel_backend=args.kernel_backend, n_qubits=args.n_qubits,
        entangle=args.entangle, shots=args.shots
    ))
    odmr = cmd_odmr(argparse.Namespace(
        seed=args.seed, out=out_dir, kernel_backend=args.kernel_backend,
        n_qubits=args.n_qubits, entangle=args.entangle, shots=args.shots,
        n_per_class=args.odmr_n_per_class, n_freq=args.odmr_n_freq,
        f_center=args.odmr_f_center, span=args.odmr_span,
        B0=args.odmr_B0, B1=args.odmr_B1, noise=args.odmr_noise
    ))
    kb = cmd_kernel_bench(argparse.Namespace(
        seed=args.seed, out=out_dir, n=args.kernel_bench_n,
        n_qubits=args.n_qubits, entangle=args.entangle, shots=args.shots
    ))

    metrics = merge_dicts(bb84, odmr, kb)
    write_json(os.path.join(out_dir, "metrics.json"), metrics)

    print("[DONE] Wrote docs/figures/*.png")

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qmsl", description="Quantum Measurement–Driven Security Lab (QKD+ODMR+QML)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("env-check", help="Print environment versions")
    pe.set_defaults(func=cmd_env_check)

    pb = sub.add_parser("bb84", help="Run BB84 experiment")
    pb.add_argument("--backend", choices=["numpy", "qutip"], default="numpy")
    pb.add_argument("--kernel-backend", choices=["analytic", "pennylane", "qiskit", "cirq", "cudaq"], default="pennylane")
    pb.add_argument("--sessions", type=int, default=1500)
    pb.add_argument("--bits", type=int, default=400)
    pb.add_argument("--eve-threshold", type=float, default=0.3)
    pb.add_argument("--n-qubits", type=int, default=4)
    pb.add_argument("--entangle", default=True, action=argparse.BooleanOptionalAction)
    pb.add_argument("--shots", type=int, default=4096)
    pb.add_argument("--seed", type=int, default=0)
    pb.add_argument("--out", type=str, default="results")
    pb.set_defaults(func=lambda a: cmd_bb84(a))

    po = sub.add_parser("odmr", help="Run ODMR experiment")
    po.add_argument("--kernel-backend", choices=["analytic", "pennylane", "qiskit", "cirq", "cudaq"], default="pennylane")
    po.add_argument("--n-qubits", type=int, default=4)
    po.add_argument("--entangle", default=True, action=argparse.BooleanOptionalAction)
    po.add_argument("--shots", type=int, default=4096)
    po.add_argument("--n-per-class", type=int, default=400)
    po.add_argument("--n-freq", type=int, default=80)
    po.add_argument("--f-center", type=float, default=2.80e9)
    po.add_argument("--span", type=float, default=20e6)
    po.add_argument("--B0", type=float, default=0.0050)  # Tesla
    po.add_argument("--B1", type=float, default=0.00515) # Tesla
    po.add_argument("--noise", type=float, default=0.004)
    po.add_argument("--seed", type=int, default=0)
    po.add_argument("--out", type=str, default="results")
    po.set_defaults(func=lambda a: cmd_odmr(a))

    pk = sub.add_parser("kernel-bench", help="Cross-backend kernel equivalence benchmark")
    pk.add_argument("--n", type=int, default=24)
    pk.add_argument("--n-qubits", type=int, default=4)
    pk.add_argument("--entangle", default=True, action=argparse.BooleanOptionalAction)
    pk.add_argument("--shots", type=int, default=4096)
    pk.add_argument("--seed", type=int, default=0)
    pk.add_argument("--out", type=str, default="results")
    pk.set_defaults(func=lambda a: cmd_kernel_bench(a))

    pa = sub.add_parser("run-all", help="Run BB84 + ODMR + kernel-bench and build article")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument("--out", type=str, default="results")
    pa.add_argument("--kernel-backend", choices=["analytic", "pennylane", "qiskit", "cirq", "cudaq"], default="pennylane")
    pa.add_argument("--n-qubits", type=int, default=4)
    pa.add_argument("--entangle", default=True, action=argparse.BooleanOptionalAction)
    pa.add_argument("--shots", type=int, default=4096)
    pa.add_argument("--bb84-backend", choices=["numpy", "qutip"], default="numpy")
    pa.add_argument("--bb84-sessions", type=int, default=1500)
    pa.add_argument("--bb84-bits", type=int, default=400)
    pa.add_argument("--eve-threshold", type=float, default=0.3)
    pa.add_argument("--odmr-n-per-class", type=int, default=400)
    pa.add_argument("--odmr-n-freq", type=int, default=80)
    pa.add_argument("--odmr-f-center", type=float, default=2.80e9)
    pa.add_argument("--odmr-span", type=float, default=20e6)
    pa.add_argument("--odmr-B0", type=float, default=0.0050)
    pa.add_argument("--odmr-B1", type=float, default=0.00515)
    pa.add_argument("--odmr-noise", type=float, default=0.004)
    pa.add_argument("--kernel-bench-n", type=int, default=24)
    pa.set_defaults(func=lambda a: cmd_run_all(a))

    return p

def main():
    p = build_parser()
    args = p.parse_args()
    out = args.func(args)
    if isinstance(out, dict):
        print(out)

if __name__ == "__main__":
    main()
