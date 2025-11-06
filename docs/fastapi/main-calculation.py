from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from scipy.fft import fft

app = FastAPI(
    title="Signal Lab API",
    description="Endpoints for FFT demos (sin-cos combos) and simple items CRUD.",
    version="0.1.0",
    contact={"name": "Boris Bolliet"},
    license_info={"name": "MIT"},
)


class Item(BaseModel):
    name: str
    price: float

@app.get("/")
def hello():
    return {"message": "Hello Boris 👋"}

@app.post("/items")
def create(item: Item):
    return {"ok": True, "item": item}

# --- FFT helpers ---
def signal(x):
    return np.sin(5*x) * np.cos(9*x)

def fft_at_k(k: int, n: int = 4096, L: float = 2*np.pi):
    """
    Evaluate the FFT of y(x)=sin(5x)*cos(9x) sampled on [0,L) at the harmonic `k`.
    We map harmonic k (i.e., sin(kx), cos(kx) world) to FFT bin m ≈ k*L/(2π).
    Returns both raw FFT Y[m] and Y[m]/n (so the DC term matches the mean).
    """
    if n <= 0:
        raise ValueError("n must be positive")
    x = np.linspace(0, L, n, endpoint=False)
    y = signal(x)
    Y = fft(y)  # unnormalized
    # Map harmonic k to nearest FFT bin m
    m_f = k * L / (2*np.pi)           # ideal bin (float)
    m = int(np.round(m_f)) % n        # nearest integer bin, wrapped
    offset = float(m_f - np.round(m_f))  # leakage indicator (0 means perfect alignment)
    return {
        "k": k,
        "n": n,
        "L": L,
        "bin_index": m,
        "ideal_bin_float": m_f,
        "bin_offset_from_integer": offset,
        "Y_m_real": float(Y[m].real),
        "Y_m_imag": float(Y[m].imag),
        "Y_m_abs": float(np.abs(Y[m])),
        "Y_m_norm_real": float((Y[m]/n).real),
        "Y_m_norm_imag": float((Y[m]/n).imag),
        "Y_m_norm_abs": float(np.abs(Y[m]/n)),
        "note": "If bin_offset_from_integer != 0, expect spectral leakage."
    }

@app.get("/fft")
def fft_endpoint(n: int = 4096, L: float = 2*np.pi, k: int = 0):
    """
    Compute FFT of sin(5x)*cos(9x) on [0,L) with N samples and return the value at harmonic k.
    - n: number of samples
    - L: domain length
    - k: harmonic index (e.g., 0, ±4, ±14). Maps to FFT bin m ≈ k*L/(2π).
    """
    try:
        return fft_at_k(k=k, n=n, L=L)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
