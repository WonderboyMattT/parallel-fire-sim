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
    if (id >= n * n) return;

    // ─────────────────────────────────────────────
    // Hash-based RNG — same approach as fire_spread
    // to avoid the diagonal pattern from the LCG
    // ─────────────────────────────────────────────
    ulong state = seed ^ ((ulong)id * 2654435761UL);
    state ^= (state >> 33);
    state *= 0xff51afd7ed558ccdUL;
    state ^= (state >> 33);
    state *= 0xc4ceb9fe1a85ec53UL;
    state ^= (state >> 33);
    float r1 = (float)(state & 0xFFFFFFFF) / (float)0xFFFFFFFF;

    state ^= ((ulong)id * 1000003UL + 1UL);
    state ^= (state >> 33);
    state *= 0xff51afd7ed558ccdUL;
    state ^= (state >> 33);
    float r2 = (float)(state & 0xFFFFFFFF) / (float)0xFFFFFFFF;

    if (r1 >= probTree) {
        grid[id] = 0;
    } else if (r2 < probBurning) {
        grid[id] = 2;
    } else {
        grid[id] = 1;
    }
}

// ─────────────────────────────────────────────
// Kernel 2: fire_spread
// Each work-item computes the next state of one cell.
//
// Rules (from assignment spec):
//   0 (empty)   -> always stays 0
//   2 (burning) -> always becomes 0 (burns out)
//   1 (tree)    -> catches fire if:
//                  - not immune (r < probImmune means immune)
//                  - AND (a neighbour is burning OR lightning strikes)
//
// Moore neighbourhood: 8 surrounding cells.
// Periodic boundary conditions: edges wrap around
// using modular arithmetic so the grid is toroidal.
//
// Double buffering: reads from gridIn, writes to gridOut.
// This avoids race conditions where one work-item reads
// a cell that another has already updated this step.
// ─────────────────────────────────────────────
kernel void fire_spread(global const int* gridIn,
                        global int* gridOut,
                        int n,
                        float probImmune,
                        float probLightning,
                        ulong seed) {

    int id = get_global_id(0);
    if (id >= n * n) return;

    // Convert flat id to 2D coordinates
    int x = id % n;
    int y = id / n;

    int current = gridIn[id];

    // ─────────────────────────────────────────────
    // RULE 1: empty cells stay empty
    // ─────────────────────────────────────────────
    if (current == 0) {
        gridOut[id] = 0;
        return;
    }

    // ─────────────────────────────────────────────
    // RULE 2: burning trees always burn out
    // ─────────────────────────────────────────────
    if (current == 2) {
        gridOut[id] = 0;
        return;
    }

    // ─────────────────────────────────────────────
    // RULE 3: tree — check Moore neighbourhood
    // Periodic boundaries: (x-1+n)%n wraps left edge
    // to right edge, etc.
    // ─────────────────────────────────────────────
    int neighbourBurning = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue; // skip self
            int nx = (x + dx + n) % n;
            int ny = (y + dy + n) % n;
            if (gridIn[ny * n + nx] == 2) {
                neighbourBurning = 1;
            }
        }
    }

    // ─────────────────────────────────────────────
    // RANDOM NUMBERS FOR THIS CELL
    // Better hash-based RNG
    // ─────────────────────────────────────────────
    ulong state = seed ^ ((ulong)id * 2654435761UL);
    state ^= (state >> 33);
    state *= 0xff51afd7ed558ccdUL;
    state ^= (state >> 33);
    state *= 0xc4ceb9fe1a85ec53UL;
    state ^= (state >> 33);
    float rImmune = (float)(state & 0xFFFFFFFF) / (float)0xFFFFFFFF;

    state ^= ((ulong)id * 1000003UL + 1UL);
    state ^= (state >> 33);
    state *= 0xff51afd7ed558ccdUL;
    state ^= (state >> 33);
    float rLightning = (float)(state & 0xFFFFFFFF) / (float)0xFFFFFFFF;

    // ─────────────────────────────────────────────
    // IMMUNITY CHECK
    // If rImmune < probImmune the tree is immune
    // and cannot catch fire this step.
    // ─────────────────────────────────────────────
    if (rImmune < probImmune) {
        gridOut[id] = 1; // immune, stays as tree
        return;
    }

    // ─────────────────────────────────────────────
    // FIRE SPREAD / LIGHTNING
    // Tree catches fire if a neighbour is burning
    // OR if struck by lightning.
    // ─────────────────────────────────────────────
    if (neighbourBurning || rLightning < probLightning) {
        gridOut[id] = 2; // catches fire
    } else {
        gridOut[id] = 1; // stays as tree
    }
}