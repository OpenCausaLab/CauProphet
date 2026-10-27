from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
import scipy.linalg as slin
from scipy.optimize import minimize
from scipy.special import expit as sigmoid

from .base import BaseDiscoveryMethod


class NOTEARSAlgorithm(BaseDiscoveryMethod):
    """NOTEARS (Zheng et al., 2018) via augmented Lagrangian."""

    def fit(self, data: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        lambda1 = kwargs.get("lambda1", 0.1)
        w_threshold = kwargs.get("w_threshold", 0.3)
        max_iter = kwargs.get("max_iter", 100)
        h_tol = kwargs.get("h_tol", 1e-8)
        rho_max = kwargs.get("rho_max", 1e+16)
        loss_type = kwargs.get("loss_type", "l2")

        X: np.ndarray = data.to_numpy(dtype=np.float64)
        n, d = X.shape

        if loss_type == "l2":
            X = X - np.mean(X, axis=0, keepdims=True)

        w_est = np.zeros(2 * d * d)
        rho, alpha, h = 1.0, 0.0, np.inf

        bnds = [
            (0, 0) if i == j else (0, None)
            for _ in range(2)
            for i in range(d)
            for j in range(d)
        ]

        for _ in range(max_iter):
            w_new, h_new = w_est, np.inf
            while rho < rho_max:
                sol = minimize(
                    fun=self._func,
                    x0=w_est,
                    args=(X, d, lambda1, alpha, rho, loss_type),
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bnds,
                )
                w_new = sol.x
                h_new, _ = self._h(self._adj(w_new, d))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            w_est, h = w_new, h_new
            alpha += rho * h
            if h <= h_tol or rho >= rho_max:
                break

        W_est = self._adj(w_est, d)
        W_est[np.abs(W_est) < w_threshold] = 0
        return (W_est != 0).astype(int)

    @staticmethod
    def _adj(w: np.ndarray, d: int) -> np.ndarray:
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    @staticmethod
    def _loss(W: np.ndarray, X: np.ndarray, loss_type: str) -> tuple[float, np.ndarray]:
        n = X.shape[0]
        M = X @ W

        if loss_type == "l2":
            R = X - M
            loss = 0.5 / n * (R ** 2).sum()
            G_loss = -1.0 / n * X.T @ R
        elif loss_type == "logistic":
            loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
            G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
        elif loss_type == "poisson":
            S = np.exp(M)
            loss = 1.0 / n * (S - X * M).sum()
            G_loss = 1.0 / n * X.T @ (S - X)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        return loss, G_loss

    @staticmethod
    def _h(W: np.ndarray) -> tuple[float, np.ndarray]:
        d = W.shape[0]
        E = slin.expm(W * W)
        h = np.trace(E) - d
        G_h = E.T * W * 2
        return h, G_h

    @staticmethod
    def _func(
        w: np.ndarray,
        X: np.ndarray,
        d: int,
        lambda1: float,
        alpha: float,
        rho: float,
        loss_type: str,
    ) -> tuple[float, np.ndarray]:
        W = NOTEARSAlgorithm._adj(w, d)
        loss, G_loss = NOTEARSAlgorithm._loss(W, X, loss_type)
        h, G_h = NOTEARSAlgorithm._h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate(
            (G_smooth + lambda1, -G_smooth + lambda1), axis=None
        )
        return obj, g_obj
