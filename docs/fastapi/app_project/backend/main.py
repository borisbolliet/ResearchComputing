from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import base64
import io
from typing import List, Optional, Literal
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Signal Lab API",
    description="Endpoints for FFT demos (sin-cos combos) and simple items CRUD.",
    version="0.1.0",
    contact={"name": "Boris Bolliet"},
    license_info={"name": "MIT"},
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # your Next.js dev URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    name: str
    price: float

class FunctionFFTRequest(BaseModel):
    function_type: Literal["sin", "cos", "sin_cos", "custom", "square", "triangle", "sawtooth"]
    parameters: dict = {}  # For custom parameters like frequency, amplitude, etc.
    custom_expression: Optional[str] = None  # For custom functions like "np.sin(2*x) + 0.5*np.cos(5*x)"
    n_samples: int = 1024
    domain_start: float = 0.0
    domain_end: float = 2 * np.pi
    plot_title: Optional[str] = None

@app.get("/")
def hello():
    return {"message": "Hello Boris 👋"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker containers"""
    return {"status": "healthy", "service": "signal-lab-backend"}

@app.post("/items")
def create(item: Item):
    return {"ok": True, "item": item}

# --- FFT helpers ---
def signal(x):
    return np.sin(5*x) * np.cos(9*x)

def generate_function(x: np.ndarray, function_type: str, parameters: dict = {}, custom_expression: Optional[str] = None) -> np.ndarray:
    """Generate various types of functions for FFT analysis."""
    if function_type == "sin":
        freq = parameters.get("frequency", 1.0)
        amp = parameters.get("amplitude", 1.0)
        phase = parameters.get("phase", 0.0)
        return amp * np.sin(2 * np.pi * freq * x + phase)
    
    elif function_type == "cos":
        freq = parameters.get("frequency", 1.0)
        amp = parameters.get("amplitude", 1.0)
        phase = parameters.get("phase", 0.0)
        return amp * np.cos(2 * np.pi * freq * x + phase)
    
    elif function_type == "sin_cos":
        # Default to the existing signal function if no parameters
        if not parameters:
            return np.sin(5*x) * np.cos(9*x)
        freq1 = parameters.get("frequency1", 5.0)
        freq2 = parameters.get("frequency2", 9.0)
        amp = parameters.get("amplitude", 1.0)
        return amp * np.sin(freq1*x) * np.cos(freq2*x)
    
    elif function_type == "square":
        freq = parameters.get("frequency", 1.0)
        amp = parameters.get("amplitude", 1.0)
        duty_cycle = parameters.get("duty_cycle", 0.5)
        period = 2 * np.pi / freq if freq != 0 else 1
        return amp * (((x % period) < (duty_cycle * period)) * 2 - 1)
    
    elif function_type == "triangle":
        freq = parameters.get("frequency", 1.0)
        amp = parameters.get("amplitude", 1.0)
        period = 2 * np.pi / freq if freq != 0 else 1
        normalized_x = (x % period) / period
        return amp * (4 * np.abs(normalized_x - 0.5) - 1)
    
    elif function_type == "sawtooth":
        freq = parameters.get("frequency", 1.0)
        amp = parameters.get("amplitude", 1.0)
        period = 2 * np.pi / freq if freq != 0 else 1
        normalized_x = (x % period) / period
        return amp * (2 * normalized_x - 1)
    
    elif function_type == "custom" and custom_expression:
        try:
            # Safe evaluation of mathematical expressions
            # Allow only safe numpy functions
            safe_dict = {
                "x": x,
                "np": np,
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "exp": np.exp,
                "log": np.log,
                "sqrt": np.sqrt,
                "pi": np.pi,
                "e": np.e
            }
            return eval(custom_expression, {"__builtins__": {}}, safe_dict)
        except Exception as e:
            raise ValueError(f"Invalid custom expression: {e}")
    
    else:
        raise ValueError(f"Unknown function type: {function_type}")

def create_fft_plot(x: np.ndarray, y: np.ndarray, fft_freqs: np.ndarray, fft_magnitude: np.ndarray, 
                   function_type: str, title: Optional[str] = None) -> str:
    """Create a plot showing the original function and its FFT, return as base64 encoded string."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Original function plot
    ax1.plot(x, y, 'b-', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(title or f'{function_type.replace("_", " ").title()} Function')
    ax1.grid(True, alpha=0.3)
    
    # FFT magnitude plot
    # Only plot positive frequencies for clarity
    n = len(fft_freqs)
    mid = n // 2
    ax2.plot(fft_freqs[:mid], fft_magnitude[:mid], 'r-', linewidth=2)
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('FFT Magnitude Spectrum')
    ax2.grid(True, alpha=0.3)
    
    # Set reasonable axis limits
    if np.max(fft_magnitude[:mid]) > 0:
        ax2.set_ylim(0, np.max(fft_magnitude[:mid]) * 1.1)
    
    plt.tight_layout()
    
    # Convert plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    plot_data = buffer.getvalue()
    buffer.close()
    plt.close(fig)
    
    return base64.b64encode(plot_data).decode('utf-8')

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

@app.post("/fft-function")
def compute_function_fft(request: FunctionFFTRequest):
    """
    Compute FFT of any function and return both numerical data and plots.
    Supports predefined functions (sin, cos, square, triangle, sawtooth) and custom expressions.
    """
    try:
        # Generate x values
        x = np.linspace(request.domain_start, request.domain_end, request.n_samples, endpoint=False)
        
        # Generate function values
        y = generate_function(x, request.function_type, request.parameters, request.custom_expression)
        
        # Compute FFT
        Y = fft(y)
        fft_freqs = fftfreq(request.n_samples, d=(x[1] - x[0]))
        fft_magnitude = np.abs(Y)
        fft_phase = np.angle(Y)
        
        # Create plot
        plot_base64 = create_fft_plot(x, y, fft_freqs, fft_magnitude, request.function_type, request.plot_title)
        
        # Prepare response data
        response_data = {
            "success": True,
            "function_type": request.function_type,
            "parameters": request.parameters,
            "n_samples": request.n_samples,
            "domain": [request.domain_start, request.domain_end],
            "plot_image": f"data:image/png;base64,{plot_base64}",
            "fft_summary": {
                "max_magnitude": float(np.max(fft_magnitude)),
                "max_magnitude_freq": float(fft_freqs[np.argmax(fft_magnitude)]),
                "dc_component": float(Y[0].real / request.n_samples),
                "total_power": float(np.sum(fft_magnitude**2))
            }
        }
        
        # Include raw data if requested (limit to reasonable size)
        if request.n_samples <= 2048:
            response_data["raw_data"] = {
                "x_values": x.tolist(),
                "y_values": y.tolist(),
                "fft_frequencies": fft_freqs.tolist(),
                "fft_magnitude": fft_magnitude.tolist(),
                "fft_phase": fft_phase.tolist()
            }
        
        return response_data
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

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
