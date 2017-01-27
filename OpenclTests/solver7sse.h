#ifndef SOLVERSSE_H
#define SOLVERSSE_H

//-------------------------------------------------------------
// aligned memory allocation cross-stuff

#include <stdlib.h>

#ifdef _WIN32

#include <malloc.h>

#define aligned_alloc(alignment, size) _aligned_malloc((size), (alignment))
#define aligned_free _aligned_free
#define align_as(alignment) __declspec(align(alignment))

#else

void* aligned_alloc(size_t alignment, size_t size);

void* aligned_alloc(size_t alignment, size_t size)
{
    void* p = NULL;
    posix_memalign (&p, alignment, size);
    return p;
}

#define aligned_free free
#define align_as(alignment) __attribute__((aligned((alignment))))

#endif

//-------------------------------------------------------------

#include <xmmintrin.h>

#include "threadpool.h"

//-------------------------------------------------------------

struct Solver : public ThreadPool
{
    size_t sseAlignment;
    size_t sseBaseCount;
    size_t sseBlockStride;
    
    size_t ncpu;
    size_t lastProcessorIndex;
    
    size_t dimension;

    size_t expandedDimension;

    size_t extra;
    size_t trail;

    size_t sseBlocksCount;
    size_t lastBlockIndex;
    
    size_t fp32VectorSize;    
    size_t fp64VectorSize;    

    size_t fp32VectorStride;
    size_t fp32ResourceStride;

    size_t fp64VectorStride;
    size_t fp64ResourceStride;

    float* fp32Matrix;    
    double* fp64Matrix;

    double* fp64Vector;

    static const int STOP = 0;
    static const int FIND_PIVOT = 1;
    static const int SWAP_COLUMNS = 2;
    static const int DIVIDE_ROW_ELEMENTS = 3;
    static const int PROCESS_MAIN_BLOCK = 4;

    static const size_t LinearMinMtWorksize = 64;
    static const size_t SquareMinMtWorksize = 8 * 8;

    struct TaskItem
    {
        int code;
        
        size_t step;
        float maxAbsValue;
        int pivotIndex;
        size_t c1;
        size_t c2;
        // ... and anything
    };

    TaskItem* taskItems;

    /////////////////////////////////////////
    Solver() :        
        sseAlignment(16),
        sseBaseCount(4),
        sseBlockStride(16),
        fp32Matrix(NULL),
        fp64Matrix(NULL),
        fp64Vector(NULL),
        taskItems(NULL)        
    {}

    /////////////////////////////////////////
    void Dispose()
    {   
        chargeStop();
        WaitResults();

        ThreadPool::Dispose();

        aligned_free(fp32Matrix);     
        aligned_free(fp64Matrix);   
        aligned_free(fp64Vector);     

        free(taskItems);
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension)
    {
        ncpu = get_ncpu();                                                  
        lastProcessorIndex = ncpu - 1;

        if(!ThreadPool::Init(ncpu))
        {
            return false;
        }

        dimension = useDimension;

        fp32VectorSize = dimension * sizeof(float);        
        fp64VectorSize = dimension * sizeof(double);

        extra = dimension % sseBaseCount;
        
        expandedDimension = extra ? (dimension + (sseBaseCount - extra)) : dimension;
        
        trail = expandedDimension - dimension;
        
        sseBlocksCount = expandedDimension / sseBaseCount;
        lastBlockIndex = sseBlocksCount - 1;

        fp32VectorStride = expandedDimension * sizeof(float);        
        fp32ResourceStride = dimension * fp32VectorStride;

        fp64VectorStride = expandedDimension * sizeof(double);
        fp64ResourceStride = dimension * fp64VectorStride;

        fp32Matrix = (float*)aligned_alloc(sseAlignment, fp32ResourceStride); 
        fp64Matrix = (double*)aligned_alloc(sseAlignment, fp64ResourceStride); 

        fp64Vector = (double*)aligned_alloc(sseAlignment, fp64VectorStride); 

        taskItems = (TaskItem*)malloc(ThreadPool::capacity * sizeof(TaskItem));

        return true;
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
            // copy initial matrix transposed
        
        float* src;
        float* dest = fp32Matrix;

        size_t stop = extra ? (sseBlocksCount - 1) : sseBlocksCount;        
        
            // tmp buffer to manipulate blocks
        align_as(16) float buf[4]; 
        __m128* pBuf = (__m128*)buf;

            // scan cols of initial matrix, store rows of transposed matrix
        for(size_t col = 0; col < dimension; ++col)
        {
            src = argMatrix + col;

            for(size_t blockIndex = 0; blockIndex < stop; ++blockIndex)
            {
                    // accumulate elements in temporary array
                for(size_t i = 0; i < sseBaseCount; ++i, src += dimension)
                {
                    buf[i] = *src;                    
                }
                    // then store whole block
                _mm_store_ps(dest, *pBuf);                
                dest += sseBaseCount;
            }
        
            if(extra)
            {
                    // accumulate last block in column of initial matrix 
                    //  (block will not be full)
                for(size_t i = 0; i < extra; ++i, src += dimension)
                {
                    buf[i] = *src;                    
                }

                    // then store whole block
                _mm_store_ps(dest, *pBuf);                
                dest += sseBaseCount;
            }

        } // end for col     

            // expand fp32 transposed matrix to fp64 
            // (also transposed, by relation to initial matrix)
        for(size_t row = 0, index = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col, ++index)
            {                
                fp64Matrix[index] = (double)(fp32Matrix[index]);    
            }

                // skip copying trailing bytes
            index += trail;
        }        
    }

    /////////////////////////////////////////
    void useVector(float* argVector)
    {
        for(size_t i = 0; i < dimension; ++i)
        {
            fp64Vector[i] = (double)(argVector[i]);
        }
    }

    /////////////////////////////////////////
    bool Solve()
    {
        for(size_t step = 0; step < dimension; ++step)        
        {            
            //int pivotIndex = sseFindPivot(step);
            //int pivotIndex = findPivot(step);
            int pivotIndex = findPivotMt(step);
            /*
            float pv = fp32Matrix[step * expandedDimension + pivotIndex];
            float pv2 = fp32Matrix[step * expandedDimension + pivotIndex2];

            if(pv != pv2)
            {
                printf("!%.0f %.0f ", pv, pv2);
            }
            */
            if(pivotIndex < 0)
            {
                return false;
            }
            
            if(pivotIndex != step)
            {
                //swapColumns(step, pivotIndex);
                swapColumnsMt(step, pivotIndex);
            }
            
            //divideRowElements(step);
            divideRowElementsMt(step);

            //processMainBlock(step);            
            processMainBlockMt(step);
        }
        
        return true;
    }

protected:
    
    virtual bool ProcessTask(int index)
    {
        int code = taskItems[index].code;

        if(code == STOP)
        {
            return false;
        }
        else if(code == FIND_PIVOT)
        {
            kernelFindPivot(index);            
        }
        else if(code == SWAP_COLUMNS)
        {
            kernelSwapColumns(index);            
        }
        else if(code == DIVIDE_ROW_ELEMENTS)
        {
            kernelDivideRowElements(index);
        }
        else if(code == PROCESS_MAIN_BLOCK)
        {
            kernelProcessMainBlock(index);
        }
        
        return true;
    }

private:

    /////////////////////////////////////////
    void chargeStop()
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = Solver::STOP;            
        }        
    }

    /////////////////////////////////////////
    void chargeKernelFindPivot(size_t step)
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = FIND_PIVOT;
            taskItems[i].step = step;
        }
    }

    int gatherPivotIndex()
    {
        float maxAbsValue = 0;
        int pivotIndex = -1;

        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            if(taskItems[i].maxAbsValue > maxAbsValue)
            {
                maxAbsValue = taskItems[i].maxAbsValue;
                pivotIndex = taskItems[i].pivotIndex;
            }
        }    

        return pivotIndex;    
    }
    
    void kernelFindPivot(int index)
    {
        size_t step = taskItems[index].step;

        size_t start = step + index;

        float* pScan = fp32Matrix + step * expandedDimension;

        float maxValue = 0;
        int pivotIndex = -1;

        for(size_t i = start; i < dimension; i += ThreadPool::capacity)
        {
            float fv = pScan[i];
            float fav = fv;

                // clear sign bit to get absolute value
            ((char*)&fav)[3] &= '\x7F';                
            
            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = i;
            }
        }

        taskItems[index].maxAbsValue = maxValue;
        taskItems[index].pivotIndex = pivotIndex;        
    }

    int findPivotMt(size_t step)
    {        
        size_t workSize = dimension - step;

        if(workSize < LinearMinMtWorksize)
        {
            return findPivot(step);
        }
        else
        {
            chargeKernelFindPivot(step);        
            
            WaitResults();

            int pivotIndex = gatherPivotIndex();

            Recharge();

            return pivotIndex;
        }        
    }
    
    int findPivot(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;
            
        for(size_t col = step, scanIndex = step * expandedDimension + col; col < dimension; ++col, ++scanIndex)
        {
            float fv = fp32Matrix[scanIndex];
            float fav = fv;

                // clear sign bit to get absolute value
            ((char*)&fav)[3] &= '\x7F';                
            
            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = col;
            }
        }

        return pivotIndex;        
    }

    int sseFindPivot(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;

            // tmp buffer to manipulate blocks
        align_as(16) float buf[4]; 
        __m128* pBuf = (__m128*)buf;

            // find nearest block-aligned index
        size_t blockAlignedIndex = step / sseBaseCount;
        bool lastBlock = (blockAlignedIndex == lastBlockIndex);
        blockAlignedIndex *= sseBaseCount;
        
        size_t skipCount = step - blockAlignedIndex;

        size_t stop = lastBlock ? (extra ? extra : sseBaseCount) : sseBaseCount;        

        float* srcIndex = fp32Matrix + step * expandedDimension + blockAlignedIndex;

            // load block as a whole
        *pBuf = _mm_load_ps(srcIndex);

            // find max in the block, skipping some elements, if needed
        for(size_t i = skipCount; i < stop; ++i)
        {            
            float fav = buf[i];

                // clear sign bit to get absolute value
            ((char*)&fav)[3] &= '\x7F';                
            
            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = i + blockAlignedIndex;
            }            
        }            

        if(lastBlock)
        {
            return pivotIndex;
        }

        srcIndex += sseBaseCount;
        blockAlignedIndex += sseBaseCount;
        
        stop = extra ? (sseBlocksCount - 1) : sseBlocksCount;
        stop *= sseBaseCount;

            // continue search in homogeneous blocks
        for(; blockAlignedIndex < stop; srcIndex += sseBaseCount, blockAlignedIndex += sseBaseCount)
        {
                // load block as a whole
            *pBuf = _mm_load_ps(srcIndex);
                        
                // retry max with data in the block
            for(size_t i = 0; i < sseBaseCount; ++i)
            {            
                float fav = buf[i];

                    // clear sign bit to get absolute value
                ((char*)&fav)[3] &= '\x7F';                
                
                if(fav > maxValue)
                {
                    maxValue = fav;
                    pivotIndex = i + blockAlignedIndex;
                }            
            }                        
        }        
            // at last, check trailing elements
        if(extra)
        {
                // load block as a whole
            *pBuf = _mm_load_ps(srcIndex);

                // retry max with data in the last block
            for(size_t i = 0; i < extra; ++i)
            {            
                float fav = buf[i];

                    // clear sign bit to get absolute value
                ((char*)&fav)[3] &= '\x7F';                
                
                if(fav > maxValue)
                {
                    maxValue = fav;
                    pivotIndex = i + blockAlignedIndex;
                }            
            }                        
        }

        return pivotIndex;
    }     

    /////////////////////////////////////////
    void chargeKernelSwapColumns(size_t c1, size_t c2)
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = SWAP_COLUMNS;
            taskItems[i].c1 = c1;
            taskItems[i].c2 = c2;
        }
    }

    void kernelSwapColumns(int index)
    {
        size_t c1 = taskItems[index].c1;
        size_t c2 = taskItems[index].c2;

        float* pScan1 = fp32Matrix + c1;
        float* pScan2 = fp32Matrix + c2;

        size_t skipSize = expandedDimension * ThreadPool::capacity;

        for(size_t i = 0; i < dimension; i += ThreadPool::capacity, pScan1 += skipSize, pScan2 += skipSize)
        {
            float t = *pScan1;
            *pScan1 = *pScan2;
            *pScan2 = t;            
        }
    }

    void swapColumnsMt(size_t c1, size_t c2)
    {
        chargeKernelSwapColumns(c1, c2);        
        
        WaitResults();

        Recharge();
    }

    void swapColumns(size_t c1, size_t c2)
    {
        float* pScan1 = fp32Matrix + c1;
        float* pScan2 = fp32Matrix + c2;

        for(size_t i = 0; i < dimension; ++i, pScan1 += expandedDimension, pScan2 += expandedDimension)
        {
            float t = *pScan1;
            *pScan1 = *pScan2;
            *pScan2 = t;
        }
    }           

    /////////////////////////////////////////
    void chargeKernelDivideRowElements(size_t step)
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = DIVIDE_ROW_ELEMENTS;
            taskItems[i].step = step;            
        }
    }

    /*
    void kernelDivideRowElements(int index)
    {
        size_t step = taskItems[index].step;

        float* pScan = fp32Matrix + step * expandedDimension + step;

        float divisor = *pScan;

        size_t offset = index + 1;

        pScan += offset;

        for(size_t i = step + offset; i < dimension; i += ThreadPool::capacity, pScan += ThreadPool::capacity)
        {
            *pScan /= divisor;
        }
    }
    */

        // SSE version
    //*
    void kernelDivideRowElements(int index)
    {
        size_t step = taskItems[index].step;

        size_t offset = step + 1;

            // find nearest block-aligned index

        size_t blockAlignedIndex = offset / sseBaseCount;        
        blockAlignedIndex *= sseBaseCount;

        size_t runStart = blockAlignedIndex + ((offset > blockAlignedIndex) ? sseBaseCount : 0);

        float* pScanRow = fp32Matrix + step * expandedDimension;

            // tmp buffer to manipulate sse blocks

        align_as(16) float bufDivisor[4]; 

        bufDivisor[0] = pScanRow[step];

            //

        if(index == 0)
        {
                // let one of threads (thread 0) update memory to aligned run start

            for(size_t col = offset; col < runStart; ++col)
            {
                pScanRow[col] /= bufDivisor[0];
            }
        }

            // copy element into all 4 words

        __m128 divisor = _mm_load1_ps(bufDivisor);
        
            // do main run
        
        size_t skipSize = sseBaseCount * ThreadPool::capacity;

        for(size_t col = runStart + index * sseBaseCount; col < dimension; col += skipSize)
        {
            // [col] is block aligned

            float *p = pScanRow + col;
                
            _mm_store_ps(p, _mm_div_ps(_mm_load_ps(p), divisor));
        }            
    }
    //*/

    void divideRowElementsMt(size_t step)
    {
        size_t workSize = dimension - step;

        if(workSize < LinearMinMtWorksize)
        {
            divideRowElements(step);
        }
        else
        {
            chargeKernelDivideRowElements(step);        
            
            WaitResults();

            Recharge();
        }        
    }

    /*
    void divideRowElements(size_t step)
    {
        float* pScan = fp32Matrix + step * expandedDimension + step;

        float divisor = *pScan;

        ++pScan;

        for(size_t i = step + 1; i < dimension; ++i, ++pScan)
        {
            *pScan /= divisor;
        }
    }
    */

        // single-threaded SSE version

    void divideRowElements(size_t step)
    {
        size_t offset = step + 1;

            // find nearest block-aligned index

        size_t blockAlignedIndex = offset / sseBaseCount;        
        blockAlignedIndex *= sseBaseCount;

        size_t runStart = blockAlignedIndex + ((offset > blockAlignedIndex) ? sseBaseCount : 0);

        float* pScanRow = fp32Matrix + step * expandedDimension;

            // tmp buffer to manipulate sse blocks

        align_as(16) float bufDivisor[4]; 

        bufDivisor[0] = pScanRow[step];

            // move to run start

        for(size_t col = offset; col < runStart; ++col)
        {
            pScanRow[col] /= bufDivisor[0];
        }

            // copy element into all 4 words           
        __m128 divisor = _mm_load1_ps(bufDivisor);

            // do main run

        for(size_t col = runStart; col < dimension; col += sseBaseCount)
        {
            // [col] is block aligned

            float *p = pScanRow + col;
                
            _mm_store_ps(p, _mm_div_ps(_mm_load_ps(p), divisor));
        }            
    }
    
    /////////////////////////////////////////
    void chargeKernelProcessMainBlock(size_t step)
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {
            taskItems[i].code = PROCESS_MAIN_BLOCK;
            taskItems[i].step = step;            
        }
    }

    /*
    void kernelProcessMainBlock(int index)
    {        
        size_t step = taskItems[index].step;

        float* pLdRow = fp32Matrix + step * expandedDimension;

        float* pScanRow = pLdRow + expandedDimension * (index + 1);

        size_t offset = step + 1;

        size_t skipSize = expandedDimension * ThreadPool::capacity;

        for(size_t row = offset + index; row < dimension; row += ThreadPool::capacity, pScanRow += skipSize)
        {
            float leadColElement = pScanRow[step];

            for(size_t col = offset; col < dimension; ++col)
            {
                pScanRow[col] -= leadColElement * pLdRow[col];
            }
        }
    }    
    */

        // SSE version
    //*    
    void kernelProcessMainBlock(int index)
    {                
        size_t step = taskItems[index].step;

        size_t offset = step + 1;

        float* pLdRow = fp32Matrix + step * expandedDimension;

        float* pScanRow = pLdRow + expandedDimension * (index + 1);
        
        size_t skipSize = expandedDimension * ThreadPool::capacity;

            // find nearest block-aligned index

        size_t blockAlignedIndex = offset / sseBaseCount;        
        blockAlignedIndex *= sseBaseCount;

        size_t runStart = blockAlignedIndex + ((offset > blockAlignedIndex) ? sseBaseCount : 0);

            // tmp buffer to manipulate sse blocks

        align_as(16) float bufLdCol[4]; 

            //

        for(size_t row = offset + index; row < dimension; row += ThreadPool::capacity, pScanRow += skipSize)
        {
                // load 1 (may be, unaligned) element

            bufLdCol[0] = pScanRow[step];

                // move to aligned run start

            size_t col = offset;

            for(; col < runStart; ++col)
            {
                pScanRow[col] -= bufLdCol[0] * pLdRow[col];
            }

                // copy element into all 4 words

            __m128 ldCol = _mm_load1_ps(bufLdCol);

                // do main run

            for(; col < dimension; col += sseBaseCount)
            {
                // [col] is block aligned

                float *p = pScanRow + col;
                   
                _mm_store_ps(p, _mm_sub_ps(_mm_load_ps(p), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col))));
            }            
        }
    }    
    //*/

    void processMainBlockMt(size_t step)
    {
        size_t workSize = dimension - step;
        workSize *= workSize;

        if(workSize < SquareMinMtWorksize)
        {
            processMainBlock(step);
        }
        else
        {
            chargeKernelProcessMainBlock(step);        
            
            WaitResults();

            Recharge();
        }        
    }

    void processMainBlock(size_t step)
    {        
        float* pLdRow = fp32Matrix + step * expandedDimension;

        float* pScanRow = pLdRow + expandedDimension;

        size_t offset = step + 1;

        for(size_t row = offset; row < dimension; ++row, pScanRow += expandedDimension)
        {
            float leadColElement = pScanRow[step];

            for(size_t col = offset; col < dimension; ++col)
            {
                pScanRow[col] -= leadColElement * pLdRow[col];
            }
        }
    }    
};

//-------------------------------------------------------------

#endif
