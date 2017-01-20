#ifndef SOLVERSSE_H
#define SOLVERSSE_H

#include <malloc.h>
#include <xmmintrin.h>

//-------------------------------------------------------------

struct Solver
{
    size_t sseAlignment;
    size_t sseBaseCount;
    size_t sseBlockStride;

    size_t dimension;

    size_t actualDimension;

    size_t extra;

    size_t sseBlocksCount;
    
    size_t fp32VectorSize;    
    size_t fp64VectorSize;    

    size_t fp32VectorStride;
    size_t fp32ResourceStride;

    size_t fp64VectorStride;
    size_t fp64ResourceStride;

    float* fp32Matrix;    
    double* fp64Matrix;

    /////////////////////////////////////////
    Solver() :
        sseAlignment(16),
        sseBaseCount(4),
        sseBlockStride(16),
        fp32Matrix(NULL),
        fp64Matrix(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {        
        _aligned_free(fp32Matrix);     
        _aligned_free(fp64Matrix);     
    }

    /////////////////////////////////////////
    bool Init(size_t useDimension)
    {
        dimension = useDimension;

        fp32VectorSize = dimension * sizeof(float);        
        fp64VectorSize = dimension * sizeof(double);

        extra = dimension % sseBaseCount;

        actualDimension = (extra == 0) ? dimension : dimension + (sseBaseCount - extra);

        sseBlocksCount = actualDimension / sseBaseCount;

        fp32VectorStride = actualDimension * sizeof(float);        
        fp32ResourceStride = dimension * fp32VectorStride;

        fp64VectorStride = actualDimension * sizeof(double);
        fp64ResourceStride = dimension * fp64VectorStride;

        fp32Matrix = (float*)_aligned_malloc(fp32ResourceStride, sseAlignment); 
        fp64Matrix = (double*)_aligned_malloc(fp64ResourceStride, sseAlignment); 

        return true;
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
            // copy arg matrix data
        
        float* src = argMatrix;
        float* dest = fp32Matrix;
        /*
        char* src = (char*)argMatrix;
        char* dest = (char*)fp32Matrix;

        for(size_t row = 0; row < dimension; ++row, src += fp32VectorSize, dest += fp32VectorStride)
        {
            memcpy(dest, src, fp32VectorSize);
        }        
        */
        //*
        size_t stop = (dimension == actualDimension) ? sseBlocksCount : (sseBlocksCount - 1);        

        __declspec(align(16)) float buf[4]; __m128* pBuf = (__m128*)buf;

        for(size_t row = 0; row < dimension; ++row)
        {
            for(size_t i = 0; i < stop; ++i, src += sseBaseCount, dest += sseBaseCount)
            {
                _mm_store_ps(dest, _mm_loadu_ps(src));
            }
        
            if(extra)
            {
                for(size_t i = 0; i < extra; ++i, ++src)
                {
                    buf[i] = *src;                    
                }

                _mm_store_ps(dest, *pBuf);
                
                dest += sseBaseCount;
            }
        }        
        //*/
            // expand matrix to fp64
    }
};

//-------------------------------------------------------------

#endif
