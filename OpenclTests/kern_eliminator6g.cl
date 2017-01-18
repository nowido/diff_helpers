    // 1D dispatch, 'dim' work items, no groups
__kernel void col_swap
                (
                    unsigned int dim, 
                    unsigned int c1, 
                    unsigned int c2, 
                    __global float* workspace
                )
{
    unsigned int item_id = get_global_id(0);
    
    int offset1 = c1 + item_id * dim;
    int offset2 = c2 + item_id * dim;

    float fv1 = workspace[offset1];
    workspace[offset1] = workspace[offset2];
    workspace[offset2] = fv1;
}

    // 1D dispatch, no groups
__kernel void row_divide
                (
                    float divisor, 
                    unsigned int offset, 
                    __global float* workspace
                )
{
    workspace[offset + get_global_id(0)] /= divisor;
}

    // 2D dispatch, with groups and local storage
__kernel void eliminator
                (
                    unsigned int dim, 
                    unsigned int step,                         
                    __global float* workspace,
                    __local float* ldRowBlock,
                    __local float* ldColumnBlock                    
                )
{
    int id_col = get_global_id(0);  // already with offset 'step + 1'
    int id_row = get_global_id(1);  // already with offset 'step + 1'

    bool actualArea = ((id_col < (int)dim) && (id_row < (int)dim));

    int threadIdX;
    int threadIdY;

    int vRowOffset;

    float currentValue;

    if(actualArea)
    {
        vRowOffset = id_row * dim;

        currentValue = workspace[vRowOffset + id_col];
        
        threadIdX = get_local_id(0);
        threadIdY = get_local_id(1);

        if(threadIdY == 0)
        {
            ldRowBlock[threadIdX] = workspace[step * dim + id_col];    
        }

        if(threadIdX == 0)
        {
            ldColumnBlock[threadIdY] = workspace[vRowOffset + step];    
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(actualArea)
    {
        workspace[vRowOffset + id_col] = currentValue - ldColumnBlock[threadIdY] * ldRowBlock[threadIdX];
    }
}
