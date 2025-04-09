# 📘 Big Picture: Neural Operators and the Math Behind Fourier Neural Operators

In traditional machine learning, models learn a function  
`f: R^n → R^m`,  
i.e., a mapping between finite-dimensional vector spaces. This works well for structured data like images, sequences, or tabular data.

However, many problems in physics and engineering involve **functions as inputs and outputs** — for example, solving a **parametric partial differential equation (PDE)**:

`G_y: a(x) ↦ u(x)`

where:
- `a(x)` is an **input function** (e.g., diffusivity, initial condition)
- `u(x)` is the **solution function** (e.g., temperature, velocity)
- `G_y` is the **solution operator** of the PDE with parameters `y`

This setting lives in **infinite-dimensional Banach spaces**:
- `a ∈ A ⊆ B_a`  
- `u ∈ U ⊆ B_u`  
where `B_a`, `B_u` are function spaces like `L^2(D)` or `C(D)`.

---

## 🎯 The Learning Objective

Given a finite set of observations `{(a_j, u_j)}_{j=1}^N` sampled from a distribution `μ` over `A`,  
the goal is to approximate the operator `G_y` using a parametric model `G_θ ≈ G_y` by minimizing the expected loss:

min_θ E_{a ∼ μ} [ L(G_θ(a), G_y(a)) ]


Here, `L` is a suitable cost function (e.g., squared `L^2` norm).

This is a **learning problem between function spaces** — not just between vectors. That’s what makes Neural Operators powerful.

---

## 🧠 Neural Operators

A **Neural Operator** is a deep architecture that learns the operator `G_y` directly. It typically consists of:

1. **Lifting layer**: maps the input function `a(x)` to a higher-dimensional representation `v₀(x)`
2. **Operator layers**: perform updates like:

v_{t+1}(x) = φ( W v_t(x) + (K(a; θ) v_t)(x) )


Where:
- `W` is a local linear transformation (e.g., 1x1 Conv)
- `φ` is a nonlinearity (e.g., ReLU)
- `K(a; θ)` is a **kernel integral operator**, parameterized by neural nets (captures global structure)

3. **Projection layer**: maps the final output `v_T(x)` back to the target function `u(x)`

---

## 🌊 Fourier Neural Operators (FNO)

The **Fourier Neural Operator** improves efficiency by performing the kernel operation in **Fourier space**.

### Step-by-step:

1. Apply Fourier transform to `v_t`:  
   `v̂_t = FFT(v_t)`
2. Truncate high-frequency modes to first `K` values
3. Multiply by learnable complex-valued weights `W(k)`:
   `v̂_{t+1}(k) = W(k) * v̂_t(k)` for `k ≤ K`
4. Set remaining modes to zero
5. Apply inverse FFT to return to real space:
   `v_{t+1} = IFFT(v̂_{t+1})`

This makes the operator:
- **Global** (via spectral convolutions)
- **Efficient** (convolutions become multiplications in Fourier space)
- **Resolution-independent** (good for zero-shot generalization)

---

## ✅ Summary

**Fourier Neural Operators** are a new class of models that:
- Learn mappings between **functions**, not just vectors
- Use **spectral convolution** for efficient, global context
- Can model **entire families of PDEs**
- Generalize to unseen resolutions (**zero-shot superresolution**)

FNOs offer a new approach to scientific ML — bridging classical numerical analysis and deep learning, with applications in climate modeling, fluid dynamics, and beyond.





![Fourier Neural Operator](https://github.com/mehranmo93/JAX-_-Fourier-Neural-Operators/blob/main/FNO.jpg)
# Introduction to implementation
Neural Operators are mappings between discretized function spaces, for example:

- Map from an initial condition to the solution function at a later point in time (or to the entire spatiotemporal solution function)
- Map from the function describing an inhomogeneous diffusivity distribution to the solution of the heat equation
- Autoregressive timesteppers, map state $ u_h^{[t]} $ to state $ u_h^{[t+1]} $

Fourier Neural Operators do so by employing the FFT to perform efficient **spectral convolution** taking into account global features. In that sense they are a multiscale architecture (Classical convolutional architectures are only local and their receptive field depends on the depth of the network).

Neural Operators allow for the solution of a whole parametric family of PDEs!

FNOs allow for zero-shot superresolution.


## ⚙️ Spectral Convolutions

Given the (real-valued) input discretized state `a` (with potentially more than one channel) defined on an equidistant mesh, the spectral convolution proceeds as follows:

1. **Fourier Transform**: Convert `a` into Fourier space using the real-valued FFT:  
   `â = rfft(a)`  
   (batched over the channel dimension)

2. **Linear Transformation (Spectral Step R)**:  
   Multiply the first `K` Fourier modes by a learnable complex-valued weight tensor `W`:  
   `â_{0:K} = W * â_{0:K}`

3. **Truncation**:  
   Set all remaining modes to zero:  
   `â_K = 0 + 0i`

4. **Inverse FFT**:  
   Transform back into real space:  
   `ã = irfft(â)`

The learnable parameters for each spectral convolution are contained in a complex-valued weight matrix of shape:

(channels_out, channels_in, modes)


Since these weights are complex-valued, the actual number of real parameters is:

2 × channels_out × channels_in × modes


---

## 🌐 Fourier Neural Operator

A classical **Fourier Neural Operator (FNO)** consists of:

- A **lifting layer**: expands input features into a higher-dimensional space  
- Several **"ResNet-style" blocks** with:
  - A spectral convolution (as described above)
  - A pointwise 1×1 convolution for residual connection
- A **projection layer**: reduces the high-dimensional representation back to output space

The core building block looks like this:

b = activation( spectral_conv(a) + Conv1x1(a) )


We will implement an example from the original paper by [Li et al. (2020)](https://arxiv.org/abs/2010.08895), using their [reference code](https://github.com/zongyi-li/fourier_neural_operator), to solve the **1D Burgers' equation**:

∂u/∂t + (1/2) ∂(u^2)/∂x = ν ∂²u/∂x²


### 🧪 Problem Setup

- **Domain**: Ω = (0, 2π), with periodic boundary conditions  
  `u(t, 0) = u(t, 2π)`

- **Diffusivity**: `ν = 0.1` (fixed)

- **Dataset**:  
  - 2048 initial conditions `u(t=0, x)`  
  - Resolution: `N = 8192` spatial points  
  - Ground truth solution at time one: `u(t=1, x)`

### 🎯 Learning Goal

Train an FNO to learn the mapping:

u(t=0, x) → u(t=1, x)


This is done via supervised learning.  
- **Input**: initial condition `u(t=0, x)`, plus spatial coordinates  
- **Output**: solution at time one `u(t=1, x)`

