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
    //static const size_t sseAlignment = cacheLine;
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
            //int pivotIndex = findPivot(step);            
            
            int pivotIndex = sseFindPivot(step);            

            if(pivotIndex < 0)
            {
                return false;
            }
            
            permutations[step] = pivotIndex;

            if(pivotIndex != step)
            {
                //swapColumns(step, pivotIndex);
                swapColumnsMt(step, pivotIndex);
            }
            
            divideRowElements(step);
            //divideRowElementsMt(step);

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
        class Apply
        {            
            size_t dimension;
            size_t expandedDimension;
            double *const fp64Matrix;
            double *const solution;
            double *const residuals;
            double *const fp64Vector;

        public:

            Apply(size_t argDimension, size_t argExpandedDimension, double* argFp64Matrix, double* argSolution, double* argResiduals, double* argFp64Vector) :                                 
                dimension(argDimension),
                expandedDimension(argExpandedDimension),
                fp64Matrix(argFp64Matrix),
                solution(argSolution),
                residuals(argResiduals),
                fp64Vector(argFp64Vector)                
            {}

            /*
            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t scanStart = workItem.begin();
                size_t scanStop = workItem.end();

                double* pScanRow = fp64Matrix + scanStart * expandedDimension;

                for(size_t row = scanStart; row < scanStop; ++row, pScanRow += expandedDimension)
                {
                    double s = 0;

                    for(size_t col = 0; col < dimension; ++col)
                    {
                        s += pScanRow[col] * solution[col];
                    }

                    residuals[row] = fp64Vector[row] - s;
                }                
            }
            */
            //*
            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t scanStart = workItem.begin();
                size_t scanStop = workItem.end();

                double* pScanRow = fp64Matrix + scanStart * expandedDimension;

                for(size_t row = scanStart; row < scanStop; ++row, pScanRow += expandedDimension)
                {
                    //align_as(sseAlignment) double s[2];
                    align_as(sseAlignment) double s[2] = {0, 0};
                    __m128d* pS = (__m128d*)s;

                    //__m128d sse2 = _mm_set1_pd(0);

                    for(size_t col = 0; col < dimension; col += 2)
                    {
                        //sse2 = _mm_add_pd(sse2, _mm_mul_pd(_mm_load_pd(pScanRow + col), _mm_load_pd(solution + col)));
                        *pS = _mm_add_pd(*pS, _mm_mul_pd(_mm_load_pd(pScanRow + col), _mm_load_pd(solution + col)));
                    }

                    //_mm_store_pd(s, sse2);

                    residuals[row] = fp64Vector[row] - s[0] - s[1];
                }                
            }    
            //*/        
        };

        parallel_for(blocked_range<size_t>(0, dimension, 8 * sseBaseCount), Apply(dimension, expandedDimension, fp64Matrix, solution, residuals, fp64Vector)); 

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
        align_as(sseAlignment) float buf[sseBaseCount]; 
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

    void swapColumnsMt(size_t c1, size_t c2)
    {
        class Apply
        {
            size_t c1;
            size_t c2;            
            size_t dimension;
            size_t expandedDimension;
            float *const fp32Matrix;            

        public:

            Apply(size_t argC1, size_t argC2, size_t argDimension, size_t argExpandedDimension, float* argFp32Matrix) :                 
                c1(argC1),
                c2(argC2),
                dimension(argDimension),
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix)                
            {}
            
            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t scanStart = workItem.begin();
                size_t scanStop = workItem.end();
                
                size_t skipOffset = scanStart * expandedDimension;

                float* pScan1 = fp32Matrix + c1 + skipOffset;
                float* pScan2 = fp32Matrix + c2 + skipOffset;

                for(size_t i = scanStart; i < scanStop; ++i, pScan1 += expandedDimension, pScan2 += expandedDimension)
                {
                    float t = *pScan1;
                    *pScan1 = *pScan2;
                    *pScan2 = t;
                }                
            }
        };

        parallel_for(blocked_range<size_t>(0, dimension, 128 * sseBaseCount), Apply(c1, c2, dimension, expandedDimension, fp32Matrix));
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

    void divideRowElementsMt(size_t step)
    {
        class Apply
        {
            size_t step;
            size_t dimension;
            size_t expandedDimension;
            float *const fp32Matrix;
            float divisor;

        public:

            Apply(size_t argStep, size_t argDimension, size_t argExpandedDimension, float* argFp32Matrix, float argDivisor) :                 
                step(argStep),
                dimension(argDimension),
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix),
                divisor(argDivisor)
            {}

            /*
            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t scanStart = workItem.begin();
                size_t scanStop = workItem.end();

                float* pScan = fp32Matrix + step * expandedDimension + scanStart;
                
                for(size_t i = scanStart; i < scanStop; ++i, ++pScan)
                {
                    *pScan /= divisor;
                }                
            }
            */
            //*
            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t scanStart = workItem.begin();
                size_t scanStop = workItem.end();
                
                    // find nearest block-aligned index
                
                //size_t extra = scanStart % sseBaseCount;

                size_t blockAlignedIndex = scanStart / sseBaseCount;        
                blockAlignedIndex *= sseBaseCount;

                size_t runStart = blockAlignedIndex + ((scanStart > blockAlignedIndex) ? sseBaseCount : 0);                
                //size_t runStart = scanStart + (extra ? (sseBaseCount - extra) : 0);                

                size_t runStop = scanStop / sseBaseCount;        
                runStop *= sseBaseCount;

                float* pScanRow = fp32Matrix + step * expandedDimension;

                //printf("[%u %u %u %u]", scanStart, runStart, runStop, scanStop);

                    // tmp buffer to manipulate sse blocks

                align_as(sseAlignment) float bufDivisor[sseBaseCount]; 

                bufDivisor[0] = divisor;

                    // move to run start

                size_t col = scanStart;

                for(; col < runStart; ++col)
                {
                    pScanRow[col] /= bufDivisor[0];
                }

                    // copy element into all 4 words           
                __m128 sseDivisor = _mm_load1_ps(bufDivisor);

                    // do main run
                
                for(; col < runStop; col += sseBaseCount)
                {
                    // [col] is block aligned

                    float *p = pScanRow + col;
                        
                    _mm_stream_ps(p, _mm_div_ps(_mm_load_ps(p), sseDivisor));
                }               
                
                    // calc tail

                for(; col < scanStop; ++col)
                {
                    pScanRow[col] /= bufDivisor[0];
                }                   
            }
            //*/
        };

        size_t workSize = dimension - step;

        if(workSize >= minMtWorksize)
        {
            float divisor = *(fp32Matrix + step * expandedDimension + step);

            parallel_for(blocked_range<size_t>(step + 1, dimension, 2 * sseBaseCount), Apply(step, dimension, expandedDimension, fp32Matrix, divisor));
        }
        else
        {
            divideRowElements(step);
        }
    }

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

        align_as(sseAlignment) float bufDivisor[sseBaseCount]; 

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
    
    /*
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
    */
    //*
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
                size_t vertScanStart = workItem.begin();
                size_t vertScanStop = workItem.end();

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
                            // calc tail 

                        for(; col < dimension; ++col)
                        {
                            pScanRow[col] -= leadColElement * pLdRow[col];
                        }                    
                    }
                    
                    pScanRow += expandedDimension;
                }
            } 
        };

        //parallel_for(blocked_range<size_t>(step + 1, dimension, 4 * sseBaseCount), Apply(step, dimension, expandedDimension, fp32Matrix));
        parallel_for(blocked_range<size_t>(step + 1, dimension), Apply(step, dimension, expandedDimension, fp32Matrix));
    }
    //*/

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
