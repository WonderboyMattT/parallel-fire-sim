// fire.cl
// Kernel 1: init_grid
// Each work-item initialises one cell of the n×n grid.
// Uses a simple LCG (linear congruential generator) for
// random numbers since OpenCL has no built-in RNG.

kernel void init_grid(global int* grid,
                      int n,
                      float probTree,
                      float probBurning,
                      ulong seed) {

    int id = get_global_id(0);
    if (id >= n * n) return;  // bounds check

    // ─────────────────────────────────────────────
    // RANDOM NUMBER GENERATION
    // Each work-item gets a unique seed by mixing
    // the global seed with its own id. This prevents
    // all cells getting the same random value.
    // LCG constants are from Numerical Recipes.
    // ─────────────────────────────────────────────
    ulong state = seed + (ulong)id * 1099087573UL;
    state = state * 6364136223846793005UL + 1442695040888963407UL;
    float r1 = (float)(state >> 33) / (float)(1u << 31);

    state = state * 6364136223846793005UL + 1442695040888963407UL;
    float r2 = (float)(state >> 33) / (float)(1u << 31);

    // ─────────────────────────────────────────────
    // CELL STATE RULES (from assignment spec):
    // 0 = empty, 1 = tree, 2 = burning tree
    // First decide if cell is a tree (probTree),
    // then if that tree is burning (probBurning).
    // ─────────────────────────────────────────────
    if (r1 >= probTree) {
        grid[id] = 0;  // empty
    } else if (r2 < probBurning) {
        grid[id] = 2;  // burning tree
    } else {
        grid[id] = 1;  // tree, not burning
    }
}

// Kernel 2: fire_step (placeholder)
kernel void fire_step(global int* grid) {
    int id = get_global_id(0);
    grid[id] = grid[id];
}