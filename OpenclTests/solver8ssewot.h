#ifndef SOLVERSSEWOT_H
#define SOLVERSSEWOT_H

#include <tbb/tbb.h>

#include <math.h>
#include <xmmintrin.h>

#include "memalign.h"

//-------------------------------------------------------------

using namespace tbb;

//-------------------------------------------------------------

struct Solver
{
    size_t dimension;

    size_t expandedDimension;

    size_t extra;
    size_t trail;

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
    
    /////////////////////////////////////////
    Solver() :        
        permutations(NULL),
        fp64MatrixLup(NULL),
        solution(NULL),
        iterativeSolution(NULL),
        residuals(NULL),
        fp32Matrix(NULL),        
        fp64Matrix(NULL),
        fp64Vector(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {   
        aligned_free(iterativeSolution);
        aligned_free(solution);
        aligned_free(residuals);

        aligned_free(fp64MatrixLup);

        free(permutations);

        aligned_free(fp32Matrix);     
        aligned_free(fp64Matrix);   
        aligned_free(fp64Vector);     
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension)
    {
        dimension = useDimension;

        fp32VectorSize = dimension * sizeof(float);        
        fp64VectorSize = dimension * sizeof(double);

        extra = dimension % SSE_BASE_COUNT;        
        expandedDimension = dimension + (extra ? (SSE_BASE_COUNT - extra) : 0);                
        trail = expandedDimension - dimension;
        
        fp32VectorStride = expandedDimension * sizeof(float);        
        fp32ResourceStride = dimension * fp32VectorStride;

        fp64VectorStride = expandedDimension * sizeof(double);
        fp64ResourceStride = dimension * fp64VectorStride;

        fp32Matrix = (float*)aligned_alloc(SSE_ALIGNMENT, fp32ResourceStride); 
        fp64Matrix = (double*)aligned_alloc(SSE_ALIGNMENT, fp64ResourceStride); 

        fp64Vector = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 

        solution = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 
        iterativeSolution = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 
        residuals = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 

        permutations = (int*)malloc(expandedDimension * sizeof(int));

        fp64MatrixLup = (double*)aligned_alloc(SSE_ALIGNMENT, fp64ResourceStride); 

        return true;
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
        float* pSrc = argMatrix;

        float* pFp32 = fp32Matrix;
        double* pFp64 = fp64Matrix;

        for(size_t row = 0; row < dimension; ++row)
        {            
            for(size_t col = 0; col < dimension; ++col)
            {
                float v = pSrc[col];    
                pFp32[col] = v;
                pFp64[col] = (double)v;
            }   

            pSrc += dimension;
            pFp32 += expandedDimension;
            pFp64 += expandedDimension;         
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
            //int pivotIndex = findPivotPp(step);            

            if(pivotIndex < 0)
            {
                return false;
            }
            
            permutations[step] = pivotIndex;

            if(pivotIndex != step)
            {
                //swapRows(step, pivotIndex);                
                swapRowsMt(step, pivotIndex);
            }
            
            //processDivMainBlockMt(step);
            //processDivMainBlockSseMt(step);
            processDivMainBlockCacheSseMt(step);
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
    double CalcResiduals1()
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
        };

        parallel_for
        (
            blocked_range<size_t>(0, dimension, 8 * SSE_BASE_COUNT), 
            Apply(dimension, expandedDimension, fp64Matrix, solution, residuals, fp64Vector)
        ); 

        double s = 0;

        for(size_t i = 0; i < dimension; ++i)
        {
            double e = residuals[i];

            s += e * e;
        }

        return s;          
    }

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
                    align_as(SSE_ALIGNMENT) double s[2] = {0, 0};
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

        parallel_for(blocked_range<size_t>(0, dimension, 8 * SSE_BASE_COUNT), Apply(dimension, expandedDimension, fp64Matrix, solution, residuals, fp64Vector)); 

        double s = 0;

        for(size_t i = 0; i < dimension; ++i)
        {
            double e = residuals[i];

            s += e * e;
        }

        return s;          
    }

private:

    int findPivot(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;
        
        float* pScan = fp32Matrix + step * expandedDimension + step;

        for(size_t row = step; row < dimension; ++row, pScan += expandedDimension)
        {
            float fav = fabs(*pScan);

            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = row;
            }
        }

        return pivotIndex;        
    }

    int findPivotPp(size_t step)
    {
        int pivotIndex = -1;        
        float maxValue = 0;
        
        float* pScan = fp32Matrix + step * expandedDimension + step;
        
        size_t workSize = dimension - step;
        size_t stop = (workSize > 16) ? (step + 16) : dimension;

        for(size_t row = step; row < stop; ++row, pScan += expandedDimension)
        {
            float fav = fabs(*pScan);

            if(fav > maxValue)
            {
                maxValue = fav;
                pivotIndex = row;
            }
        }

        return pivotIndex;        
    }

    void swapRows(size_t r1, size_t r2)
    {
        float* pScan1 = fp32Matrix + r1 * expandedDimension;
        float* pScan2 = fp32Matrix + r2 * expandedDimension;

        for(size_t i = 0; i < dimension; ++i, ++pScan1, ++pScan2)
        {
            float t = *pScan1;
            *pScan1 = *pScan2;
            *pScan2 = t;
        }
    }           

    void swapRowsMt(size_t r1, size_t r2)
    {
        class Apply
        {
            size_t r1;
            size_t r2;            
            size_t expandedDimension;
            float *const fp32Matrix;
            
        public:

            Apply(size_t argR1, size_t argR2, size_t argExpandedDimension, float* argFp32Matrix) :                 
                r1(argR1),
                r2(argR2),                
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix)
            {}

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t colStart = workItem.begin();
                size_t colStop = workItem.end();

                float* pScan1 = fp32Matrix + r1 * expandedDimension + colStart;
                float* pScan2 = fp32Matrix + r2 * expandedDimension + colStart;

                for(size_t i = colStart; i < colStop; ++i, ++pScan1, ++pScan2)
                {
                    float t = *pScan1;
                    *pScan1 = *pScan2;
                    *pScan2 = t;
                }
            }            

        }; // end class Apply

        parallel_for
        (
            blocked_range<size_t>(0, dimension, 32 * SSE_BASE_COUNT),              
            Apply(r1, r2, expandedDimension, fp32Matrix)
        );           
    }           
    
    void divideColElements(size_t step)
    {
        float* pScan = fp32Matrix + step * expandedDimension + step;

        float divisor = *pScan;

        pScan += expandedDimension;

        for(size_t i = step + 1; i < dimension; ++i, pScan += expandedDimension)
        {
            *pScan /= divisor;
        }
    }

    void divideColElementsMt(size_t step)
    {
        class Apply
        {
            size_t step;                       
            size_t expandedDimension;
            float *const fp32Matrix;
            float divisor;
            
        public:

            Apply(size_t argStep, size_t argExpandedDimension, float* argFp32Matrix, float argDivisor) :                 
                step(argStep),                         
                expandedDimension(argExpandedDimension),
                fp32Matrix(argFp32Matrix),
                divisor(argDivisor)
            {}

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t rowStart = workItem.begin();
                size_t rowStop = workItem.end();

                float* pScan = fp32Matrix + rowStart * expandedDimension + step;

                for(size_t i = rowStart; i < rowStop; ++i, pScan += expandedDimension)
                {
                    *pScan /= divisor;
                }
            }            

        }; // end class Apply

        parallel_for
        (
            blocked_range<size_t>(step + 1, dimension, 8 * SSE_BASE_COUNT),              
            Apply(step, expandedDimension, fp32Matrix, fp32Matrix[step * expandedDimension + step])
        );                   
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
                
                for(size_t row = vertScanStart; row < vertScanStop; ++row, pScanRow += expandedDimension)
                {
                    float leadColElement = pScanRow[step];

                    for(size_t col = offset; col < dimension; ++col)
                    {
                        pScanRow[col] -= leadColElement * pLdRow[col];
                    }
                }                
            }            

        }; // end class Apply

        parallel_for
        (
            //blocked_range<size_t>(step + 1, dimension, 8 * SSE_BASE_COUNT), 
            blocked_range<size_t>(step + 1, dimension), 
            Apply(step, dimension, expandedDimension, fp32Matrix)
        );   
    }    

    void processDivMainBlockMt(size_t step)
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

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t vertScanStart = workItem.begin();
                size_t vertScanStop = workItem.end();

                float* pLdRow = fp32Matrix + step * expandedDimension;

                float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;

                size_t offset = step + 1;
                
                for(size_t row = vertScanStart; row < vertScanStop; ++row, pScanRow += expandedDimension)
                {
                    float leadColElement = (pScanRow[step] /= divisor);

                    for(size_t col = offset; col < dimension; ++col)
                    {
                        pScanRow[col] -= leadColElement * pLdRow[col];
                    }
                }                
            }            

        }; // end class Apply

        parallel_for
        (
            //blocked_range<size_t>(step + 1, dimension, 32 * SSE_BASE_COUNT), 
            blocked_range<size_t>(step + 1, dimension), 
            Apply(step, dimension, expandedDimension, fp32Matrix, fp32Matrix[step * expandedDimension + step])
        );   
    }    

    void processDivMainBlockSseMt(size_t step)
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

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t vertScanStart = workItem.begin();
                size_t vertScanStop = workItem.end();

                float* pLdRow = fp32Matrix + step * expandedDimension;

                float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;

                size_t offset = step + 1;
                
                size_t rem = offset % SSE_BASE_COUNT;
                size_t runStart = offset + (rem ? (SSE_BASE_COUNT - rem) : 0);

                for(size_t row = vertScanStart; row < vertScanStop; ++row, pScanRow += expandedDimension)
                {
                    align_as(SSE_ALIGNMENT) float leadColElement = (pScanRow[step] /= divisor);
                    
                    size_t col = offset;

                    for(; col < runStart; ++col)
                    {
                        pScanRow[col] -= leadColElement * pLdRow[col];
                    }

                        // expand leadColElement to mm 4 floats
                    __m128 ldCol = _mm_load1_ps(&leadColElement);

                    for(; col < dimension; col += SSE_BASE_COUNT)
                    {                        
                        float* p = pScanRow + col;

                        _mm_stream_ps(p, _mm_sub_ps(_mm_load_ps(p), _mm_mul_ps(ldCol, _mm_load_ps(pLdRow + col))));                        
                    }
                }                
            }            

        }; // end class Apply

        parallel_for
        (
            //blocked_range<size_t>(step + 1, dimension, 32 * SSE_BASE_COUNT), 
            blocked_range<size_t>(step + 1, dimension), 
            Apply(step, dimension, expandedDimension, fp32Matrix, fp32Matrix[step * expandedDimension + step])
        );   
    }    

    void processDivMainBlockCacheSseMt(size_t step)
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

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t vertScanStart = workItem.begin();
                size_t vertScanStop = workItem.end();

                float* pLdRow = fp32Matrix + step * expandedDimension;

                float* pScanRow = fp32Matrix + vertScanStart * expandedDimension;

                size_t offset = step + 1;
                
                for(size_t row = vertScanStart; row < vertScanStop; ++row, pScanRow += expandedDimension)
                {                    
                    size_t clExtra = (((unsigned long)(pScanRow + offset) & (CACHE_LINE - 1)) >> 2);

                    size_t cacheAlignedIndex = clExtra ? (offset - clExtra + (CACHE_LINE >> 2)) : offset;            
                    cacheAlignedIndex = (cacheAlignedIndex < dimension) ? cacheAlignedIndex : dimension;
                    size_t lastCacheAlignedIndex = dimension - (((unsigned long)(pScanRow + dimension) & (CACHE_LINE - 1)) >> 2);            
                                        
                    align_as(SSE_ALIGNMENT) float leadColElement = (pScanRow[step] /= divisor);
                    
                    size_t col = offset;

                    for(; col < cacheAlignedIndex; ++col)
                    {
                        pScanRow[col] -= leadColElement * pLdRow[col];
                    }

                        // expand leadColElement to mm 4 floats
                    __m128 ldCol = _mm_load1_ps(&leadColElement);

                        // do main run

                    for(; col < lastCacheAlignedIndex; col += 4 * SSE_BASE_COUNT)
                    {                        
                        float* p = pScanRow + col;
                        float* pLd = pLdRow + col;  
                        
                        float* p0 = p + 0 * SSE_BASE_COUNT;
                        float* p1 = p + 1 * SSE_BASE_COUNT;
                        float* p2 = p + 2 * SSE_BASE_COUNT;
                        float* p3 = p + 3 * SSE_BASE_COUNT;

                        __m128 v10 = _mm_load_ps(p0);
                        __m128 v11 = _mm_load_ps(p1);
                        __m128 v12 = _mm_load_ps(p2);
                        __m128 v13 = _mm_load_ps(p3);

                        __m128 v20 = _mm_load_ps(pLd + 0 * SSE_BASE_COUNT);
                        __m128 v21 = _mm_load_ps(pLd + 1 * SSE_BASE_COUNT);
                        __m128 v22 = _mm_load_ps(pLd + 2 * SSE_BASE_COUNT);
                        __m128 v23 = _mm_load_ps(pLd + 3 * SSE_BASE_COUNT);

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
            }            

        }; // end class Apply

        parallel_for
        (
            //blocked_range<size_t>(step + 1, dimension, 128 * SSE_BASE_COUNT), 
            blocked_range<size_t>(step + 1, dimension), 
            Apply(step, dimension, expandedDimension, fp32Matrix, fp32Matrix[step * expandedDimension + step])
        );   
    }    

    /////////////////////////////////////////
    void expandMatrix()
    {
        float* pfp32 = fp32Matrix;
        double* pfp64 = fp64MatrixLup;
        
        for(size_t row = 0; row < dimension; ++row)
        {            
            for(size_t col = 0; col < dimension; ++col)
            {
                pfp64[col] = (double)(pfp32[col]);
            }

            pfp32 += expandedDimension;
            pfp64 += expandedDimension;
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
