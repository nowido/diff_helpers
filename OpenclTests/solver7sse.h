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

    struct TaskItem
    {
        int code;
        
        size_t step;
        float maxAbsValue;
        int pivotIndex;

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
            if(pivotIndex != pivotIndex2)
            {
                printf("!%d %d ", pivotIndex, pivotIndex2);
            }
            */
            if(pivotIndex < 0)
            {
                return false;
            }
            
            if(pivotIndex != step)
            {
                // swap cols
            }
            
            // divide elements in 'step' row
            // process main block of values 
        }
        
        return true;
    }

protected:
    
    virtual bool ProcessTask(int index)
    {
        int code = taskItems[index].code;

        if(code == Solver::STOP)
        {
            return false;
        }
        else if(code == FIND_PIVOT)
        {
            kernelFindPivot(index);            
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
            taskItems[i].code = Solver::FIND_PIVOT;
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

        size_t workSize = dimension - step;        
        size_t workItemSize = workSize / ThreadPool::capacity;
                
        size_t start = step + index * workItemSize;
        size_t stop = (index == lastProcessorIndex) ? dimension : (start + workItemSize);

        float* pScan = fp32Matrix + step * expandedDimension + start;

        float maxValue = 0;
        int pivotIndex = -1;

        for(size_t i = start; i < stop; ++i, ++pScan)
        {
            float fv = *pScan;
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

        if(workSize < 256)
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

    /*
    int findPivotMt(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;
        
        size_t workSize = dimension - step;

        if(workSize < 64)
        {
            return findPivot(step);
        }

        size_t workItemSize = workSize / ncpu;

        thread_handle thands[ncpu];
        thread_args_findPivot targs[ncpu];        

        for(size_t i = 0, loaded = 0; i < ncpu; ++i)
        {               
            targs[i].expandedDimension = expandedDimension;
            targs[i].step = step;
            targs[i].matrix = fp32Matrix;         

            targs[i].start = step + loaded;
            loaded += workItemSize;
            targs[i].stop = (i < lastProcessorIndex) ? (step + loaded) : dimension;

            targs[i].maxValue = 0;
            targs[i].pivotIndex = -1;

            thands[i] = create_thread(thread_proc_findPivot, (void*)&(targs[i]));
        }

        for(size_t i = 0; i < ncpu; ++i)
        {
            void* retValue;
            join_thread(thands[i], &retValue);

            if(targs[i].maxValue > maxValue)
            {
                maxValue = targs[i].maxValue;
                pivotIndex = targs[i].pivotIndex;
            }
        }

        return pivotIndex;        
    }
    */
    /////////////////////////////////////////
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

    /////////////////////////////////////////
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
};

//-------------------------------------------------------------

#endif
