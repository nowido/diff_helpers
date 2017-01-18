    // 1D dispatch, 'dim' work items
__kernel void col_swap
                (
                    unsigned int dim, 
                    unsigned int c1, 
                    unsigned int c2, 
                    __global double* workspace
                )
{
    unsigned int item_id = get_global_id(0);
    
    int offset1 = c1 + item_id * dim;
    int offset2 = c2 + item_id * dim;

    double fv1 = workspace[offset1];
    workspace[offset1] = workspace[offset2];
    workspace[offset2] = fv1;
}

    // 1D dispatch
__kernel void row_divide
                (
                    double divisor, 
                    unsigned int offset, 
                    __global double* workspace
                )
{
    workspace[offset + get_global_id(0)] /= divisor;
}

    // 2D dispatch
__kernel void eliminator
                (
                    unsigned int dim, 
                    unsigned int step, 
                    __global double* workspace
                )
{
    int id_col = get_global_id(0);  // already with offset 'step + 1'
    int id_row = get_global_id(1);  // already with offset 'step + 1'

    int vRowOffset = id_row * dim;

    workspace[vRowOffset + id_col] -= workspace[vRowOffset + step] * workspace[step * dim + id_col];
}
