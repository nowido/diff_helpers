// 1d ("vertical") workgroup, N threads
// kernel constant runLength - how many row elements each thread writes
// N threads, each loads runLength/N top elements into shared memory,
//  and 1 lead column element into private mem (registry), to be used in processing later
// also first thread in group loads shared leadDivisor

__kernel void eliminator
                (
                    size_t dim, 
                    size_t left, 
                    size_t top, 
                    int leadingIndex,                 
                    size_t runLength,                               
                    __local float* topRowElements, // array of size runLength                    
                    __global float* matrixIn, 
                    __global float* matrixOut    
                )
{        
    __local float leadDivisor;

    size_t id_col = get_global_id(0) * runLength + left;  
    size_t id_row = get_global_id(1);  // already with offset 'top'

    size_t workGroupSize = get_local_size(1);
    size_t threadId = get_local_id(1) % workGroupSize;

    bool validLead = (leadingIndex >= 0); // we set leadingIndex == -1 if there is no leading element in the top row

    bool needSwap = (leadingIndex != (int)left) && validLead;

    int topOffset = top * dim;
    int rowOffset = id_row * dim;

    int leadCol = needSwap ? leadingIndex : left;

        // use thread with local id 0 to fetch leadDivisor

    if(threadId == 0)
    {
        leadDivisor = matrixIn[topOffset + leadCol];
    }

        // use threads to fetch top row elements

    size_t localCount = runLength / workGroupSize;
    size_t localOffset = threadId * localCount;

    for(size_t i = 0, currentCol = id_col + localOffset; i < localCount; ++i, ++currentCol)
    {
        bool itIsLeft = (currentCol == left);
        bool itIsLead = ((int)currentCol == leadingIndex);

        int colToFetch = needSwap ? (itIsLeft ? leadingIndex : (itIsLead ? left : currentCol)) : currentCol;

        size_t matrixOffset = topOffset + colToFetch;

        topRowElements[localOffset + i] = (colToFetch < (int)dim) ? matrixIn[matrixOffset] : 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);        

    if(id_row <= dim)
    {
            // fetch lead column values and safely divide it by leadDivisor

        float leadValue = matrixIn[rowOffset + leadCol];
        float leadSafeRatio = (fabs(leadDivisor) > 0) ? (leadValue / leadDivisor) : leadValue;

            // perform output elements computations using locally cached top row elements
            
        for(size_t i = 0, currentCol = id_col; i < runLength; ++i, ++currentCol)
        {
            bool itIsLeft = (currentCol == left);
            bool itIsLead = ((int)currentCol == leadingIndex);

            int colToFetch = needSwap ? (itIsLeft ? leadingIndex : (itIsLead ? left : currentCol)) : currentCol;

            float value = matrixIn[rowOffset + colToFetch];
            float divisor = topRowElements[i];

            if(currentCol < dim)
            {
                matrixOut[rowOffset + currentCol] = 
                    itIsLeft ? leadSafeRatio : ((fabs(divisor) > 0) ? (leadSafeRatio - value / divisor) : value);
            }             
        }
    }
}
