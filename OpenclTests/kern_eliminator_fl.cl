__kernel void eliminator
                (
                    unsigned int dim, 
                    int left, 
                    int top, 
                    int leadingIndex, 
                    __global float* matrixIn, 
                    __global float* matrixOut)
{
    int id_col = get_global_id(0);  // already with offset 'left'
    int id_row = get_global_id(1);  // already with offset 'top'

    int topOffset = top * dim;
    int rowOffset = id_row * dim;

    bool validLead = (leadingIndex >= 0); // we set leadingIndex == -1 if there is no leading element in the top row

    bool needSwap = (leadingIndex != left) && validLead;

    bool weAreLeft = (id_col == left);
    bool weAreLead = (id_col == leadingIndex);

    int leadCol = needSwap ? leadingIndex : left;

    float leadDivisor = matrixIn[topOffset + leadCol];
    float leadValue = matrixIn[rowOffset + leadCol];

    float leadSafeRatio = (fabs(leadDivisor) > 0) ? (leadValue / leadDivisor) : leadValue;

    int ourCol = needSwap ? (weAreLeft ? leadingIndex : (weAreLead ? left : id_col)) : id_col;

    float ourDivisor = matrixIn[topOffset + ourCol];
    float ourValue = matrixIn[rowOffset + ourCol];

    matrixOut[rowOffset + id_col] = 
        validLead ? 
            (weAreLeft ? leadSafeRatio : ((fabs(ourDivisor) > 0) ? (leadSafeRatio - ourValue / ourDivisor) : ourValue)) : 
            ourValue;
}