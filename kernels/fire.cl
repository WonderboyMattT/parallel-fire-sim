kernel void fire_step(global int* grid) {
    int id = get_global_id(0);
    grid[id] = grid[id];
}