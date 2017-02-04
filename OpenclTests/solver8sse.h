#ifndef SOLVERSSE_H
#define SOLVERSSE_H

#include <tbb/tbb.h>

using namespace tbb;

#include "threadpool_pi.h"

//-------------------------------------------------------------

#include <xmmintrin.h>

//-------------------------------------------------------------

struct Solver : public ThreadPool
{
    static const size_t cacheLine = 64;

    static const size_t sseAlignment = 16;
    static const size_t sseBaseCount = 4;
    static const size_t sseBlockStride = 16;
    
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

    int* permutations;

    double* fp64MatrixLup;

    double* solution;
    double* iterativeSolution;
    double* residuals;
    
    static const size_t minMtWorksize = 8 * sseBaseCount;

    union TaskItem
    {
        align_as(cacheLine) size_t step;
        char padding[cacheLine - sizeof(size_t)];
    };

    TaskItem* taskItems;

    /////////////////////////////////////////
    Solver() :        
        permutations(NULL),
        fp64MatrixLup(NULL),
        solution(NULL),
        iterativeSolution(NULL),
        residuals(NULL),
        fp32Matrix(NULL),        
        fp64Matrix(NULL),
        fp64Vector(NULL),
        taskItems(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {   
        ThreadPool::stop = true;
        WaitResults();

        ThreadPool::Dispose();

        aligned_free(iterativeSolution);
        aligned_free(solution);
        aligned_free(residuals);

        aligned_free(fp64MatrixLup);

        free(permutations);

        aligned_free(fp32Matrix);     
        aligned_free(fp64Matrix);   
        aligned_free(fp64Vector);     

        aligned_free(taskItems);
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
        expandedDimension = dimension + (extra ? (sseBaseCount - extra) : 0);                
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

        solution = (double*)aligned_alloc(sseAlignment, fp64VectorStride); 
        iterativeSolution = (double*)aligned_alloc(sseAlignment, fp64VectorStride); 
        residuals = (double*)aligned_alloc(sseAlignment, fp64VectorStride); 

        permutations = (int*)malloc(expandedDimension * sizeof(int));

        fp64MatrixLup = (double*)aligned_alloc(sseAlignment, fp64ResourceStride); 

        taskItems = (TaskItem*)aligned_alloc(64, ThreadPool::capacity);

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

            // expand initial matrix to fp64             
        for(size_t row = 0, indexSrc = 0, indexDest = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col, ++indexDest, ++indexSrc)
            {                
                fp64Matrix[indexDest] = (double)(argMatrix[indexSrc]);    
            }

                // skip copying trailing bytes
            indexDest += trail;
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
            int pivotIndex = findPivot(step);            

            if(pivotIndex < 0)
            {
                return false;
            }
            
            permutations[step] = pivotIndex;

            if(pivotIndex != step)
            {
                swapColumns(step, pivotIndex);
            }
            
            divideRowElements(step);

            processMainBlockMt(step);
            //processMainBlock(step);
        }
        
        expandMatrix();

        useLupToSolve(solution, fp64Vector);

        return true;
    }

    /////////////////////////////////////////
    bool Iterate(size_t count)
    {
        if(!Solve())
        {
            return false;
        }

        for(size_t i = 0; i < count; ++i)
        {
            CalcResiduals();

            useLupToSolve(iterativeSolution, residuals);

            for(size_t col = 0; col < dimension; ++col)
            {
                solution[col] += iterativeSolution[col];
            }
        }

        return true;            
    }

    /////////////////////////////////////////
    double CalcResiduals()
    {
        double* pScanRow = fp64Matrix;

        for(size_t row = 0; row < dimension; ++row, pScanRow += expandedDimension)
        {
            double s = 0;

            for(size_t col = 0; col < dimension; ++col)
            {
                s += pScanRow[col] * solution[col];
            }

            residuals[row] = fp64Vector[row] - s;
        }

        double s = 0;

        for(size_t i = 0; i < dimension; ++i)
        {
            double e = residuals[i];

            s += e * e;
        }

        return s;        
    }

protected:
    
    virtual void ProcessTask(int index, std::pair<size_t, size_t>& workItem)
    {        
        kernelProcessMainBlock(taskItems[index].step, workItem);
    }

private:

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
    //*
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
    //*/
    /////////////////////////////////////////
    void chargeKernelProcessMainBlock(size_t step)
    {
        for(size_t i = 0; i < ThreadPool::capacity; ++i)
        {            
            taskItems[i].step = step;            
        }
        
            // make vertical partitioning of [offset, dimension);            
            // each processor gets G * 4 rows of data to operate on
        
        size_t offset = step + 1;

        const size_t granularity = 8;
        const size_t granuledSkip = sseBaseCount * granularity;
        
        for(size_t acc = offset; acc < dimension;)
        {                        
            size_t next = acc + granuledSkip;

            next = (next > dimension) ? dimension : next;

            ThreadPool::items.push(std::pair<size_t, size_t>(acc, next));

            acc = next;
        }              
    }

    void kernelProcessMainBlock1(size_t step, std::pair<size_t, size_t>& workItem)
    {
        size_t offset = step + 1;

        float* pLdRow = fp32Matrix + step * expandedDimension;

        size_t vertScanStart = workItem.first;
        size_t vertScanStop = workItem.second;

        float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;
        
        for(size_t row = vertScanStart; row < vertScanStop; ++row)
        {
            float leadColElement = pScanRow[step];

            for(size_t col = offset; col < dimension; ++col)
            {
                pScanRow[col] -= leadColElement * pLdRow[col];
            }

            pScanRow += expandedDimension;
        }
    }

    void kernelProcessMainBlock(size_t step, std::pair<size_t, size_t>& workItem)
    {
        size_t vertScanStart = workItem.first;
        size_t vertScanStop = workItem.second;

        float* pLdRow = fp32Matrix + step * expandedDimension;

        float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;

        size_t offset = step + 1;

        for(size_t row = vertScanStart; row < vertScanStop; ++row)
        {
            float* p = pScanRow + offset;

            size_t clExtra = (((unsigned long)(pScanRow + offset) & (cacheLine - 1)) >> 2);

            size_t cacheAlignedIndex = clExtra ? (offset - clExtra + (cacheLine >> 2)) : offset;            
            cacheAlignedIndex = (cacheAlignedIndex < dimension) ? cacheAlignedIndex : dimension;
            
            size_t lastCacheAlignedIndex = dimension - (((unsigned long)(pScanRow + dimension) & (cacheLine - 1)) >> 2);            
            
            align_as(cacheLine) float leadColElement = pScanRow[step];

            size_t col = offset;

                // run to cache aligned
                          
            for(; col < cacheAlignedIndex; ++col)
            {
                pScanRow[col] -= leadColElement * pLdRow[col];
            }

            if(col < dimension)
            {
                    // copy element into all 4 words

                __m128 ldCol = _mm_load1_ps(&leadColElement);

                    // do main run
                /*
                for(col = cacheAlignedIndex; col < lastCacheAlignedIndex;)
                {
                    float *p = pScanRow + col;   
                                        
                    __m128 rOut0 = _mm_sub_ps(_mm_load_ps(pScanRow + col), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col)));

                    col += sseBaseCount;

                    __m128 rOut1 = _mm_sub_ps(_mm_load_ps(pScanRow + col), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col)));

                    col += sseBaseCount;

                    __m128 rOut2 = _mm_sub_ps(_mm_load_ps(pScanRow + col), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col)));

                    col += sseBaseCount;

                    __m128 rOut3 = _mm_sub_ps(_mm_load_ps(pScanRow + col), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col)));

                    col += sseBaseCount;

                    _mm_stream_ps(p + 0 * sseBaseCount, rOut0);
                    _mm_stream_ps(p + 1 * sseBaseCount, rOut1);
                    _mm_stream_ps(p + 2 * sseBaseCount, rOut2);
                    _mm_stream_ps(p + 3 * sseBaseCount, rOut3);
                }     
                */

                //*
                for(col = cacheAlignedIndex; col < lastCacheAlignedIndex; col += 4 * sseBaseCount)
                {
                    float* p = pScanRow + col;
                    float* pLd = pLdRow + col;  
                    
                    float* p0 = p + 0 * sseBaseCount;
                    float* p1 = p + 1 * sseBaseCount;
                    float* p2 = p + 2 * sseBaseCount;
                    float* p3 = p + 3 * sseBaseCount;

                    __m128 v10 = _mm_load_ps(p0);
                    __m128 v11 = _mm_load_ps(p1);
                    __m128 v12 = _mm_load_ps(p2);
                    __m128 v13 = _mm_load_ps(p3);

                    __m128 v20 = _mm_load_ps(pLd + 0 * sseBaseCount);
                    __m128 v21 = _mm_load_ps(pLd + 1 * sseBaseCount);
                    __m128 v22 = _mm_load_ps(pLd + 2 * sseBaseCount);
                    __m128 v23 = _mm_load_ps(pLd + 3 * sseBaseCount);

                    _mm_stream_ps(p0, _mm_sub_ps(v10, _mm_mul_ps(ldCol, v20)));
                    _mm_stream_ps(p1, _mm_sub_ps(v11, _mm_mul_ps(ldCol, v21)));
                    _mm_stream_ps(p2, _mm_sub_ps(v12, _mm_mul_ps(ldCol, v22)));
                    _mm_stream_ps(p3, _mm_sub_ps(v13, _mm_mul_ps(ldCol, v23)));                    
                }     
                //*/
                    // calc tail 

                for(; col < dimension; ++col)
                {
                    pScanRow[col] -= leadColElement * pLdRow[col];
                }                    
            }
            
            pScanRow += expandedDimension;
        }
    }

        // SSE version
    /*    
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
    */

    //*
    void processMainBlockMt(size_t step)
    {
        size_t workSize = dimension - step;

        if(workSize >= minMtWorksize)
        {
            chargeKernelProcessMainBlock(step);        
            
            WaitResults();

            Recharge();            
        }
        else
        {
            processMainBlock(step);
        }        
    }
    //*/
    /*
    void processMainBlockMt(size_t step)
    {
        class Apply
        {
            size_t step;
            size_t dimension;
            size_t expandedDimension;
            float *const fp32Matrix;

        public:

            Apply(size_t argStep, size_t argDimension, size_t argExpandedDimension, float* argFp32Matrix) :                 
                step(argStep),
                dimension(argDimension),
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix)
            {}

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t offset = step + 1;
                
                size_t blockAlignedIndex = offset / sseBaseCount;  
                blockAlignedIndex *= sseBaseCount;

                size_t runStart = blockAlignedIndex + ((offset > blockAlignedIndex) ? sseBaseCount : 0);
                
                float* pLdRow = fp32Matrix + step * expandedDimension;

                size_t vertScanStart = workItem.begin();
                size_t vertScanStop = workItem.end();

                float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;
                
                for(size_t row = vertScanStart; row < vertScanStop; ++row, pScanRow += expandedDimension)
                {
                    float leadColElement = pScanRow[step];

                    for(size_t col = offset; col < dimension; ++col)
                    {
                        pScanRow[col] -= leadColElement * pLdRow[col];
                    }
                }                
            } 
        };

        parallel_for(blocked_range<size_t>(step + 1, dimension), Apply(step, dimension, expandedDimension, fp32Matrix));
    }
    */
    /*
    void processMainBlockMt(size_t step)
    {
        class Apply
        {
            size_t step;
            size_t dimension;
            size_t expandedDimension;
            float *const fp32Matrix;

        public:

            Apply(size_t argStep, size_t argDimension, size_t argExpandedDimension, float* argFp32Matrix) :                 
                step(argStep),
                dimension(argDimension),
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix)
            {}

            void operator()(const blocked_range<size_t>& r) const
            {
                float* pLdRow = fp32Matrix + step * expandedDimension;
                
                size_t offset = step + 1;

                    // tmp buffer to manipulate sse blocks

                align_as(16) float bufLdCol[4]; 
                
                    // find nearest block-aligned index

                size_t blockAlignedIndex = offset / 4;        
                blockAlignedIndex *= 4;

                size_t runStart = blockAlignedIndex + ((offset > blockAlignedIndex) ? 4 : 0);
                
                for(size_t i = r.begin(); i != r.end(); ++i)
                {
                    float* pScanRow = fp32Matrix + i * expandedDimension;
                    
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

                    for(; col < dimension; col += 4)
                    {
                        // [col] is block aligned

                        float *p = pScanRow + col;   

                        _mm_store_ps(p, _mm_sub_ps(_mm_load_ps(p), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col))));
                    }            
                }
            } 
        };

        parallel_for(blocked_range<size_t>(step + 1, dimension), Apply(step, dimension, expandedDimension, fp32Matrix));
    }
    */
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

    /////////////////////////////////////////
    void expandMatrix()
    {
        float* pfp32 = fp32Matrix;
        
        for(size_t row = 0; row < dimension; ++row, pfp32 += expandedDimension)
        {            
                // row becomes a column (de-transpose LU matrix)
            
            double* pfp64 = fp64MatrixLup + row;

            for(size_t col = 0; col < dimension; ++col, pfp64 += expandedDimension)
            {
                *pfp64 = (double)(pfp32[col]);
            }
        }        
    }

    /////////////////////////////////////////
    void useLupToSolve(double* x, double* b)
    {        
        memcpy(x, b, fp64VectorSize);

            // permute right-hand part
        
        size_t lastIndex = dimension - 1;

        for(size_t step = 0; step < lastIndex; ++step)
        {
            int permutationIndex = permutations[step];
            
            if(permutationIndex != step)
            {
                double v = x[step];
                x[step] = x[permutationIndex];
                x[permutationIndex] = v;
            }
        }

            // Ly = Pb (in place)
        
        double* pScanRow = fp64MatrixLup + expandedDimension;

        for(size_t row = 1; row < dimension; ++row, pScanRow += expandedDimension)
        {            
            double s = 0;

            for(int col = 0; col < row; ++col)
            {
                s += pScanRow[col] * x[col];
            }

            x[row] -= s;
        }

            // Ux = y
        
        pScanRow -= expandedDimension;

        for(int row = lastIndex; row >= 0; --row, pScanRow -= expandedDimension)
        {
            double s = 0;

            double de = pScanRow[row];

            for(int col = row + 1; col < dimension; ++col)
            {
                s += pScanRow[col] * x[col];
            }

            x[row] -= s;
            x[row] /= de;        
        }
    }
};

//-------------------------------------------------------------

#endif
