__kernel void eliminator
                (
                    unsigned int dim, 
                    unsigned int left, 
                    unsigned int top, 
                    int leadingIndex, 
                    __global float* matrixIn, 
                    __global float* matrixOut)
{
    int id_col = get_global_id(0);  // already with offset 'left'
    int id_row = get_global_id(1);  // already with offset 'top'

    int topOffset = top * dim;
    int rowOffset = id_row * dim;

    bool validLead = (leadingIndex >= 0); // leadingIndex == -1 if no leading element

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

    float ourSafeRatio = (fabs(ourDivisor) > 0) ? (ourValue / ourDivisor) : ourValue;

    matrixOut[rowOffset + id_col] = weAreLeft ? ourSafeRatio : (leadSafeRatio - ourSafeRatio);
}