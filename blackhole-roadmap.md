# Black Hole Raytracer Development Roadmap

## Project Overview

Build a specialized C++ raytracing engine that visualizes light bending around a single black hole using geodesic integration through curved spacetime (Schwarzschild metric). No materials, no multiple objects, no complex shading—only rays, a camera, the black hole, and physics.

## Phase 1: Vector Math & Ray Setup (Day 1 - 1-2 hours)

### 1.1 Vector Math Library

- [ ] Create `src/math/vec3.h`
- [ ] Implement `vec3` class with:
  - [ ] Constructor: `vec3(double x, double y, double z)`
  - [ ] Operator overloads: `+`, `-`, `*`, `/`
  - [ ] Methods: `dot()`, `cross()`, `length()`, `squared_length()`, `unit_vector()`
  - [ ] Utility: color conversion to PPM format

### 1.2 Ray Class

- [ ] Create `src/core/ray.h`
- [ ] Implement `ray` class with:
  - [ ] Constructor: `ray(vec3 origin, vec3 direction)`
  - [ ] Method: `point_at_parameter(double t)` returns origin + t * direction

### 1.3 PPM Output

- [ ] Create `src/rendering/ppm_writer.h`
- [ ] Implement PPM file writer function
- [ ] Write pixels to file in format: P3 (ASCII PPM)

### 1.4 Simple Camera

- [ ] Create `src/core/camera.h`
- [ ] Implement camera class with:
  - [ ] Fixed position (e.g., origin at z=0)
  - [ ] Field of view
  - [ ] Method: `get_ray(double u, double v)` to cast rays through pixels
  - [ ] u, v normalized to [0, 1] from pixel coordinates

### 1.5 Checkpoint: Gradient Image

- [ ] Create `src/main.cpp` main function
- [ ] Cast rays and set pixel colors based on ray direction
- [ ] Output 200×100 gradient test image to `output/test_gradient.ppm`
- [ ] View in Preview/image viewer—should show smooth color gradient

---

## Phase 2: Schwarzschild Metric & Geodesic Integration (Days 2-5 - 8-10 hours)

### 2.1 Schwarzschild Metric Implementation

- [ ] Create `src/physics/SchwarzschildMetric.h`
- [ ] Implement class with:
  - [ ] Constructor: `SchwarzschildMetric(double mass)` where mass is black hole mass M
  - [ ] Method: `get_metric_tensor(double r, double theta)` returns 4×4 metric tensor components
    - Only store non-zero diagonal elements for Schwarzschild (faster)
    - Components: g_tt, g_rr, g_theta_theta, g_phi_phi
  - [ ] Method: `get_christoffel_symbol(double r, double theta, int mu, int nu, int lambda)` returns scalar
    - Pre-compute and hardcode Christoffel symbols for Schwarzschild (find online or derive)
  - [ ] Method: `delta(double r)` returns r² - 2M*r (used for step-size control)
  - [ ] Method: `event_horizon_radius()` returns 2M

### 2.2 4-Momentum Initialization

- [ ] Create `src/physics/FourMomentum.h`
- [ ] Implement helper functions:
  - [ ] `ray_to_four_momentum(ray, camera_position, schwarzschild_metric)`
    - Convert 3D ray direction to 4-momentum k^μ
    - Enforce null constraint: g_μν k^μ k^ν = 0
  - [ ] Normalize so proper calculations work

### 2.3 Geodesic State

- [ ] Create `src/physics/GeodesicState.h`
- [ ] Implement structure to hold:
  - [ ] Position: x^μ = (t, r, θ, φ)
  - [ ] 4-momentum: k^μ = (k_t, k_r, k_θ, k_φ)
  - [ ] Affine parameter: λ

### 2.4 RK4 Geodesic Integrator

- [ ] Create `src/physics/GeodesicIntegrator.h`
- [ ] Implement `GeodesicIntegrator` class:
  - [ ] Constructor takes a `SchwarzschildMetric` reference
  - [ ] Method: `step(GeodesicState& state, double step_size, SchwarzschildMetric& metric)`
    - Implements one RK4 step of geodesic equation:
      - d²x^λ/dλ² = -Γ^λ_μν (dx^μ/dλ)(dx^ν/dλ)
    - Updates state position and 4-momentum
  - [ ] Method: `integrate(GeodesicState state, SchwarzschildMetric& metric)` runs full integration
    - Max iterations: 10,000 steps
    - Returns termination reason (hit horizon, escaped, max steps)

### 2.5 Adaptive Step-Size Logic

- [ ] In integrator, implement step-size selection:
  - [ ] Base step: ds = 0.01M (reasonable for most regions)
  - [ ] Adaptive: ds = 0.01M × |Δ(r)| / (1 + |Δ(r)|) to shrink near horizon
  - [ ] Prevents instability near r = 2M

### 2.6 Termination Conditions

- [ ] In integrator, check after each step:
  - [ ] If r < 2M + ε (hit event horizon) → return "ABSORBED"
  - [ ] If r > 1000M (escaped far away) → return "ESCAPED"
  - [ ] If steps > 10,000 → return "MAX_STEPS"

### 2.7 Checkpoint: Test Geodesic Integration

- [ ] Create test function in main:
  - [ ] Initialize geodesic far from black hole (r = 50M)
  - [ ] Integrate backward along a ray
  - [ ] Print: final r, final θ, termination reason
  - [ ] Verify: integration completes without crashes or NaN
  - [ ] Test several rays at different angles

---

## Phase 3: Raytracing Main Loop (Days 5-6 - 5-6 hours)

### 3.1 Background Starfield

- [ ] Create `src/rendering/background.h`
- [ ] Implement simple background function:
  - [ ] Input: direction vector (where ray escaped to)
  - [ ] Output: color (vec3)
  - [ ] Simple version: gradient based on angle (blue at top, black at edges)
  - [ ] Or: load star texture file (optional complexity)

### 3.2 Raytracing Color Function

- [ ] Create `src/rendering/raytracer.h`
- [ ] Implement `color(ray, camera_pos, schwarzschild_metric, integrator)` function:
  - [ ] Convert ray to 4-momentum
  - [ ] Create initial GeodesicState
  - [ ] Integrate geodesic backward from camera
  - [ ] Check termination:
    - [ ] If ABSORBED → return black (0, 0, 0)
    - [ ] If ESCAPED → return background_color(escape_direction)
    - [ ] If MAX_STEPS → return dark gray (shouldn't happen normally)

### 3.3 Main Raytracing Loop

- [ ] Modify `src/main.cpp`:
  - [ ] Set up camera at position (distance from black hole, viewing angle)
  - [ ] Set black hole mass M (in geometric units where c = G = 1)
  - [ ] Create SchwarzschildMetric object
  - [ ] Create GeodesicIntegrator object
  - [ ] For each pixel (i, j):
    - [ ] Cast ray: `ray r = camera.get_ray(i/width, j/height)`
    - [ ] Compute color: `color(r, camera_pos, metric, integrator)`
    - [ ] Write to PPM buffer
  - [ ] Write PPM file to `output/blackhole.ppm`

### 3.4 Checkpoint: Black Hole Visible

- [ ] Run program
- [ ] Output `output/blackhole.ppm`
- [ ] View image—should show:
  - [ ] Dark circle at center (black hole shadow)
  - [ ] Lensing effect around edges (distortion of background stars)
  - [ ] Light gray/colored background where rays escape

### 3.5 Debugging & Validation

- [ ] Vary camera position and verify shadow size changes correctly
- [ ] Verify shadow diameter ≈ 5.2M (Schwarzschild radius of photon sphere)
- [ ] Check lensing distortion makes physical sense
- [ ] Profile performance: time per frame (should be seconds for 200×100)

---

## Phase 4: Visualization Enhancements (Days 7-9 - 4-6 hours) [OPTIONAL]

### 4.1 Antialiasing

- [ ] Modify raytracing loop:
  - [ ] For each pixel, cast N rays with random offsets (e.g., N=4)
  - [ ] Average colors of all rays
  - [ ] Results in smoother black hole edge

### 4.2 Higher Resolution

- [ ] Change output resolution to 800×600 or 1920×1080
- [ ] Adjust antialiasing samples accordingly
- [ ] Increase max iterations if needed for accuracy

### 4.3 Accretion Disk Model (Optional)

- [ ] Create `src/physics/AccretionDisk.h`
- [ ] Define orbiting particles at specific radii (e.g., r = 6M to r = 20M)
- [ ] Modify color function:
  - [ ] After integration, check if ray's closest approach to black hole is near disk radius
  - [ ] If yes, return heat-based color (hotter/bluer for closer orbits)
  - [ ] If no, return black hole shadow or background

### 4.4 Camera Positioning Enhancements

- [ ] Add parameters for arbitrary viewpoint:
  - [ ] lookfrom, lookat, up vectors
  - [ ] Adjustable field of view
  - [ ] Compute camera basis from these parameters

### 4.5 Output Formats (Optional)

- [ ] Extend to PNG/JPEG output (add external library or stick with PPM)
- [ ] Add command-line parameters for resolution, mass, position

---

## Phase 5: Advanced Physics (Days 10+ - Optional)

### 5.1 Kerr Metric (Rotating Black Hole)

- [ ] Create `src/physics/KerrMetric.h`
- [ ] Extend metric to rotating spacetime
- [ ] Parameters: mass M, spin parameter a
- [ ] New effects: frame-dragging, offset photon ring
- [ ] Update integrator to use new Christoffel symbols

### 5.2 Metrics Framework

- [ ] Create abstract `Metric` base class
- [ ] Implement SchwarzschildMetric and KerrMetric as subclasses
- [ ] Allow runtime metric switching (Schwarzschild by default)

### 5.3 GPU Acceleration with Metal (Optional)

- [ ] Move geodesic integration to Metal compute shader
- [ ] Parallelize pixel processing on GPU
- [ ] Requires C++ ↔ Metal interop code

---

## Deliverables Checklist

### Minimum Viable (Phases 1-3)

- [ ] Working C++ black hole raytracer
- [ ] Schwarzschild metric implementation
- [ ] RK4 geodesic integrator with adaptive step-size
- [ ] PPM output showing black hole silhouette
- [ ] Visible gravitational lensing effect
- [ ] Clean code structure with separate physics/rendering modules

### Enhanced (Phase 4)

- [ ] Antialiasing for smooth edges
- [ ] Higher resolution output
- [ ] Accretion disk visualization
- [ ] Customizable camera parameters

### Advanced (Phase 5)

- [ ] Kerr metric support
- [ ] Pluggable metric system
- [ ] GPU acceleration

---

## Key Mathematical References

### Schwarzschild Metric

- Boyer-Lindquist coordinates: (t, r, θ, φ)
- Metric tensor (non-zero components):
  - g_tt = -(1 - 2M/r)
  - g_rr = 1/(1 - 2M/r)
  - g_θθ = r²
  - g_φφ = r²sin²(θ)
- Event horizon: r_+ = 2M
- Photon sphere: r_photon = 3M

### Geodesic Equation

- d²x^λ/dλ² = -Γ^λ_μν (dx^μ/dλ)(dx^ν/dλ)
- For null geodesics (photons): g_μν k^μ k^ν = 0
- λ is the affine parameter (use proper distance as proxy)

### Christoffel Symbols for Schwarzschild

- Pre-compute or find online (e.g., Wikipedia, GR textbooks)
- For implementation, hardcode specific non-zero components
- Example: Γ^r_tt = M(r-2M)/r³
