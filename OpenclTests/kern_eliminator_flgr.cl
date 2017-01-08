__kernel void eliminator
                (
                    unsigned int dim, 
                    int left, 
                    int top, 
                    int leadingIndex, 
                    __local float* leadColumnRatio, // array of size groupRowsCount
                    __local float* topRowElements, // array of size groupColumnsCount
                    __local bool* safeDivisors, // array of size groupColumnsCount
                    __global float* matrixIn, 
                    __global float* matrixOut)
{
    __local float leadDivisor;

        // global and local thread ids

    int id_col = get_global_id(0);  // already with offset 'left'
    int id_row = get_global_id(1);  // already with offset 'top'

    if((id_col >= (int)dim) || (id_row > (int)dim + 1))
    {
        return;
    }
    
    size_t threadIdX = get_local_id(0);
    size_t threadIdY = get_local_id(1);

        //

    bool validLead = (leadingIndex >= 0); // we set leadingIndex == -1 if there is no leading element in the top row

    bool needSwap = (leadingIndex != left) && validLead;

    int topOffset = top * dim;
    int rowOffset = id_row * dim;

    int leadCol = needSwap ? leadingIndex : left;

    bool weAreLeft = (id_col == left);
    bool weAreLead = (id_col == leadingIndex);

    int ourCol = needSwap ? (weAreLeft ? leadingIndex : (weAreLead ? left : id_col)) : id_col;

        // use thread with local id (0, 0) to fetch leadDivisor

    bool safeDivisor;

    if((threadIdX == 0) && (threadIdY == 0))
    {
        leadDivisor = matrixIn[topOffset + leadCol];

        safeDivisor = (fabs(leadDivisor) > 0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

        // use threads with local ids (0, 0 .. groupRowsCount - 1) to fetch lead column values and divide it by leadDivisor

    if(threadIdX == 0)
    {
        float leadValue = matrixIn[rowOffset + leadCol];

        leadColumnRatio[threadIdY] = safeDivisor ? (leadValue / leadDivisor) : leadValue;
    }

        // use threads with local ids (0 .. groupColumnsCount - 1, 0) to fetch top row elements

    if(threadIdY == 0)    
    {
        float ourDivisor = matrixIn[topOffset + ourCol];

        topRowElements[threadIdX] = ourDivisor;

        safeDivisors[threadIdX] = (fabs(ourDivisor) > 0);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

        // use all threads to perform elements computations using locally cached values

    float ourLeadRatio = leadColumnRatio[threadIdY];
    float ourDivisor = topRowElements[threadIdX];
    float ourValue = matrixIn[rowOffset + ourCol];

    matrixOut[rowOffset + id_col] = 
        validLead ? 
            (weAreLeft ? ourLeadRatio : (safeDivisors[threadIdX] ? (ourLeadRatio - ourValue / ourDivisor) : ourValue)) : 
            ourValue;    
}