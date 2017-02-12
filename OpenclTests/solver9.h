#ifndef SOLVER9_H
#define SOLVER9_H

#include <tbb/tbb.h>

#include <math.h>
#include <xmmintrin.h>

#include "memalign.h"

//-------------------------------------------------------------

using namespace tbb;

//-------------------------------------------------------------

struct Tile
{
    size_t tileRows;
    size_t tileCols;

    size_t tileSize;

    float* tile;

    size_t runStop;

    /////////////////////////////////////////
    Tile(size_t argTileRows, size_t argTileCols) :
        tileRows(argTileRows),
        tileCols(argTileCols)
    {
        tileSize = tileRows * tileCols;

        tile = (float*)aligned_alloc(SSE_ALIGNMENT, tileSize * sizeof(float));
        
        runStop = tileRows / SSE_BASE_COUNT;
        runStop *= SSE_BASE_COUNT;
    }
    
    /////////////////////////////////////////
    void ImportFromRowMajor(const float* pMatrixBlock, size_t stride)
    {
            // tile is col major
                
        float* pDest = tile;

        for(size_t col = 0; col < tileCols; ++col)
        {
            const float* pSrc = pMatrixBlock + col;

            for(size_t row = 0; row < tileRows; ++row, ++pDest, pSrc += stride)
            {
                *pDest = *pSrc;                
            }
        }
    }

    /////////////////////////////////////////
    void ExportToRowMajor(float* pMatrixBlock, size_t stride)
    {
        float* pSrc = tile;

        for(size_t col = 0; col < tileCols; ++col)
        {
            float* pDest = pMatrixBlock + col;

            for(size_t row = 0; row < tileRows; ++row, ++pSrc, pDest += stride)
            {
                *pDest = *pSrc;                
            }
        }
    }

    /////////////////////////////////////////
    void ExportToRowMajorFp64(double* pMatrixBlock, size_t stride)
    {
        float* pSrc = tile;

        for(size_t col = 0; col < tileCols; ++col)
        {
            double* pDest = pMatrixBlock + col;

            for(size_t row = 0; row < tileRows; ++row, ++pSrc, pDest += stride)
            {
                *pDest = (double)(*pSrc);                
            }
        }
    }

    /////////////////////////////////////////
    ~Tile()
    {
        aligned_free(tile);
    }

    /////////////////////////////////////////
    int findPivot(size_t offsetVert, size_t offsetHoriz)
    {
        int pivotIndex = -1;        
        float maxValue = 0;
        
        float* pScan = tile + offsetHoriz * tileRows + offsetVert;

        for(size_t row = offsetVert; row < tileRows; ++row, ++pScan)
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

    /////////////////////////////////////////
    void swapRows(size_t r1, size_t r2)
    {
        float* pScan1 = tile + r1;
        float* pScan2 = tile + r2;

        for(size_t i = 0; i < tileCols; ++i, pScan1 += tileRows, pScan2 += tileRows)
        {
            float t = *pScan1;
            *pScan1 = *pScan2;
            *pScan2 = t;
        }
    }   

    /////////////////////////////////////////
    void scaleColumn(size_t offsetVert, size_t offsetHoriz, float divisor)
    {
        float* pScan = tile + offsetHoriz * tileRows + offsetVert;

        for(size_t row = offsetVert; row < tileRows; ++row, ++pScan)
        {
            *pScan /= divisor;
        }        
    }

    /////////////////////////////////////////
    void updateMain1
            (
                size_t offsetVert, 
                size_t offsetHoriz, 
                Tile* leadRowTile, 
                size_t leadRowVert,                  
                const float* leadColBlock
            )
    {   
        size_t leadRowTileStride = leadRowTile->tileRows;

        float* pLeadRow = leadRowTile->tile + offsetHoriz * leadRowTileStride + leadRowVert;     

        for(size_t col = offsetHoriz; col < tileCols; ++col, pLeadRow += leadRowTileStride)
        {
            size_t index = col * tileRows;

            float ldRowValue = *pLeadRow;

            for(size_t row = offsetVert; row < tileRows; ++row)
            {
                tile[index + row] -= leadColBlock[row] * ldRowValue;      
            }                
        }
    }    

    void updateMain
            (
                size_t offsetVert, 
                size_t offsetHoriz, 
                Tile* leadRowTile, 
                size_t leadRowVert,                  
                const float* leadColBlock
            )
    {   
        size_t leadRowTileStride = leadRowTile->tileRows;

        float* pLeadRow = leadRowTile->tile + offsetHoriz * leadRowTileStride + leadRowVert;     

        for(size_t col = offsetHoriz; col < tileCols; ++col, pLeadRow += leadRowTileStride)
        {
            size_t index = col * tileRows;

                // 1. from offsetVert to aligned run start (but no more than to tileRows)
                // 2. from aligned run start to aligned run stop
                // 3. from aligned run stop to tileRows

            size_t extra = offsetVert % SSE_BASE_COUNT;
            size_t runStart = offsetVert + (extra ? (SSE_BASE_COUNT - extra) : 0);
            
            runStart = (runStart < tileRows) ? runStart : tileRows;

            align_as(SSE_ALIGNMENT) float ldRowValue = *pLeadRow;

            for(size_t row = offsetVert; row < runStart; ++row)
            {
                tile[index + row] -= leadColBlock[row] * ldRowValue;      
            }                

            __m128 ldRowValueQuad = _mm_load1_ps(&ldRowValue);

            float* p = tile + index;

            size_t row = runStart;

            for(; row < runStop; row += SSE_BASE_COUNT)
            {                
                _mm_stream_ps(p + row, _mm_sub_ps(_mm_load_ps(p + row), _mm_mul_ps(_mm_load_ps(leadColBlock + row), ldRowValueQuad)));
            }   
                        
            for(; row < tileRows; ++row)
            {
                tile[index + row] -= leadColBlock[row] * ldRowValue;      
            }                
        }
    }        
};

//-------------------------------------------------------------

struct TiledMatrix
{
    static const size_t tileRows = 128;
    static const size_t tileCols = 128;
    
    size_t dimension;

    size_t mainCountHoriz;
    size_t mainCountVert;

    size_t tilesCountHoriz;
    size_t tilesCountVert;
    
    size_t extraHoriz;
    size_t extraVert;

    Tile** tiles;

    /////////////////////////////////////////
    TiledMatrix(size_t argDimension) :
        dimension(argDimension)
    {        
        mainCountHoriz = dimension / tileCols;        
        extraHoriz = dimension - mainCountHoriz * tileCols;
        tilesCountHoriz = mainCountHoriz + (extraHoriz ? 1 : 0);

        mainCountVert = dimension / tileRows;
        extraVert = dimension - mainCountVert * tileRows;
        tilesCountVert = mainCountVert + (extraVert ? 1 : 0);

        tiles = (Tile**)malloc(tilesCountHoriz * tilesCountVert * sizeof(Tile));

        size_t index = 0;

        for(size_t i = 0; i < mainCountVert; ++i)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index] = new Tile(tileRows, tileCols);
            }

            if(extraHoriz)
            {
                tiles[index] = new Tile(tileRows, extraHoriz);
                ++index;    
            }
        }

        if(extraVert)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index] = new Tile(extraVert, tileCols);
            }

            if(extraHoriz)
            {
                tiles[index] = new Tile(extraVert, extraHoriz);                
            }            
        }
    }   

    /////////////////////////////////////////
    ~TiledMatrix()
    {
        size_t index = 0;

        for(size_t i = 0; i < tilesCountVert; ++i)
        {
            for(size_t j = 0; j < tilesCountHoriz; ++j, ++index)
            {
                delete tiles[index];
            }
        }

        free(tiles);
    } 

    /////////////////////////////////////////
    void ImportFromRowMajor(const float* pMatrix, size_t stride)    
    {
        size_t index = 0;

        const float* pSrc = pMatrix;

        size_t horizSkip = mainCountHoriz * tileCols;        
        size_t vertSkip = stride * tileRows;

        for(size_t i = 0; i < mainCountVert; ++i, pSrc += vertSkip)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ImportFromRowMajor(pSrc + j * tileCols, stride);                
            }

            if(extraHoriz)
            {
                tiles[index]->ImportFromRowMajor(pSrc + horizSkip, stride);                
                ++index;    
            }            
        }

        if(extraVert)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ImportFromRowMajor(pSrc + j * tileCols, stride);                
            }

            if(extraHoriz)
            {
                tiles[index]->ImportFromRowMajor(pSrc + horizSkip, stride);                                
            }            
        }        
    }

    /////////////////////////////////////////
    void ExportToRowMajor(float* pMatrix, size_t stride)    
    {
        size_t index = 0;

        float* pDest = pMatrix;

        size_t horizSkip = mainCountHoriz * tileCols;        
        size_t vertSkip = stride * tileRows;

        for(size_t i = 0; i < mainCountVert; ++i, pDest += vertSkip)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ExportToRowMajor(pDest + j * tileCols, stride);                
            }

            if(extraHoriz)
            {
                tiles[index]->ExportToRowMajor(pDest + horizSkip, stride);             
                ++index;    
            }            
        }

        if(extraVert)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ExportToRowMajor(pDest + j * tileCols, stride);                                
            }

            if(extraHoriz)
            {
                tiles[index]->ExportToRowMajor(pDest + horizSkip, stride);                
            }            
        }
    }    

    /////////////////////////////////////////
    void ExportToRowMajorFp64(double* pMatrix, size_t stride)    
    {
        size_t index = 0;

        double* pDest = pMatrix;

        size_t horizSkip = mainCountHoriz * tileCols;        
        size_t vertSkip = stride * tileRows;

        for(size_t i = 0; i < mainCountVert; ++i, pDest += vertSkip)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ExportToRowMajorFp64(pDest + j * tileCols, stride);                
            }

            if(extraHoriz)
            {
                tiles[index]->ExportToRowMajorFp64(pDest + horizSkip, stride);             
                ++index;    
            }            
        }

        if(extraVert)
        {
            for(size_t j = 0; j < mainCountHoriz; ++j, ++index)
            {
                tiles[index]->ExportToRowMajorFp64(pDest + j * tileCols, stride);                                
            }

            if(extraHoriz)
            {
                tiles[index]->ExportToRowMajorFp64(pDest + horizSkip, stride);                
            }            
        }
    }        

    /////////////////////////////////////////
    int findPivot(size_t step)
    {
        size_t tileHorizIndex = step / tileCols;
        size_t tileVertIndex = step / tileRows;

        size_t inTileHorizOffset = step - tileHorizIndex * tileCols;
        size_t inTileVertOffset = step - tileVertIndex * tileRows;

        size_t index = tileVertIndex * tilesCountHoriz + tileHorizIndex;

        int pivotIndex = -1;
        float pivotValue = 0;

            // first tile
        
        Tile* curentTile = tiles[index];
                
        int localPivotIndex = curentTile->findPivot(inTileVertOffset, inTileHorizOffset); 

        if(localPivotIndex > -1)
        {
            float fav = fabs(curentTile->tile[inTileHorizOffset * (curentTile->tileRows) + localPivotIndex]);

            if(fav > pivotValue)
            {
                pivotValue = fav;
                pivotIndex = tileVertIndex * tileRows + localPivotIndex;
            }
        }
            // next tiles in 'tile column'
        
        index += tilesCountHoriz;

        for(size_t i = tileVertIndex + 1; i < tilesCountVert; ++i, index += tilesCountHoriz)
        {
            curentTile = tiles[index];

            localPivotIndex = curentTile->findPivot(0, inTileHorizOffset); 

            if(localPivotIndex > -1)
            {
                float fav = fabs(curentTile->tile[inTileHorizOffset * (curentTile->tileRows) + localPivotIndex]);

                if(fav > pivotValue)
                {
                    pivotValue = fav;
                    pivotIndex = i * tileRows + localPivotIndex;
                }
            }
        }

        return pivotIndex;
    }

    /////////////////////////////////////////
    void swapRows1(size_t r1, size_t r2)
    {
        size_t vertIndex1 = r1 / tileRows;
        size_t vertIndex2 = r2 / tileRows;

        size_t inTileR1 = r1 - vertIndex1 * tileRows;
        size_t inTileR2 = r2 - vertIndex2 * tileRows;

        size_t index1 = vertIndex1 * tilesCountHoriz;
        size_t index2 = vertIndex2 * tilesCountHoriz;

        if(vertIndex1 != vertIndex2)
        {            
            for(size_t i = 0; i < tilesCountHoriz; ++i)
            {
                swapRows(tiles[index1 + i], tiles[index2 + i], inTileR1, inTileR2);
            }    
        }
        else
        {   
            for(size_t i = 0; i < tilesCountHoriz; ++i)
            {
                tiles[index1 + i]->swapRows(inTileR1, inTileR2);
            }    
        }
    }    

    /////////////////////////////////////////
    void swapRows(size_t r1, size_t r2)
    {
        size_t vertIndex1 = r1 / tileRows;
        size_t vertIndex2 = r2 / tileRows;

        size_t inTileR1 = r1 - vertIndex1 * tileRows;
        size_t inTileR2 = r2 - vertIndex2 * tileRows;

        size_t index1 = vertIndex1 * tilesCountHoriz;
        size_t index2 = vertIndex2 * tilesCountHoriz;

        if(vertIndex1 != vertIndex2)
        {      
            class Apply
            {
                TiledMatrix* host;
                size_t index1;
                size_t index2;
                size_t inTileR1;
                size_t inTileR2;

            public:

                Apply(TiledMatrix* argHost, size_t argIndex1, size_t argIndex2, size_t argInTileR1, size_t argInTileR2):
                    host(argHost),
                    index1(argIndex1),
                    index2(argIndex2),
                    inTileR1(argInTileR1),
                    inTileR2(argInTileR2)                                        
                {}

                void operator()(const blocked_range<size_t>& workItem) const
                {
                    size_t start = workItem.begin();
                    size_t stop = workItem.end();

                    Tile** tiles = host->tiles;

                    for(size_t i = start; i < stop; ++i)
                    {
                        host->swapRows(tiles[index1 + i], tiles[index2 + i], inTileR1, inTileR2);
                    }                        
                }
            };         

            parallel_for
            (
                blocked_range<size_t>(0, tilesCountHoriz), 
                Apply(this, index1, index2, inTileR1, inTileR2)
            ); 
        }
        else
        {   
            class Apply
            {
                Tile** tiles;
                size_t index;
                size_t inTileR1;
                size_t inTileR2;

            public:

                Apply
                (
                    Tile** argTiles,
                    size_t argIndex,
                    size_t argInTileR1,
                    size_t argInTileR2                    
                )
                    :

                    tiles(argTiles),
                    index(argIndex),
                    inTileR1(argInTileR1),
                    inTileR2(argInTileR2)                                        
                {}

                void operator()(const blocked_range<size_t>& workItem) const
                {
                    size_t start = workItem.begin();
                    size_t stop = workItem.end();

                    for(size_t i = start; i < stop; ++i)
                    {
                        tiles[index + i]->swapRows(inTileR1, inTileR2);
                    }                        
                }
            };         

            parallel_for
            (
                blocked_range<size_t>(0, tilesCountHoriz), 
                Apply(tiles, index1, inTileR1, inTileR2)
            ); 
        }
    }    

    /////////////////////////////////////////
    void scaleColumn(size_t step)
    {
        size_t leadRowTileVertIndex = step / tileRows;
        size_t inLeadRowTileVertOffset = step - leadRowTileVertIndex * tileRows;
        
        size_t tileHorizIndex = step / tileCols;
        size_t inTileHorizOffset = step - tileHorizIndex * tileCols;

        size_t leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + tileHorizIndex;

        Tile* leadTile = tiles[leadRowTileIndex];

        float divisor = leadTile->tile[inTileHorizOffset * (leadTile->tileRows) + inLeadRowTileVertOffset];

            //

        size_t offset = step + 1;

        size_t tileVertIndex = offset / tileRows;
        size_t inTileVertOffset = offset - tileVertIndex * tileRows;

            //

        size_t index = tileVertIndex * tilesCountHoriz + tileHorizIndex;

        Tile* curentTile = tiles[index];

            // first tile

        curentTile->scaleColumn(inTileVertOffset, inTileHorizOffset, divisor);

            // next tiles in 'tile column'

        index += tilesCountHoriz;

        for(size_t i = tileVertIndex + 1; i < tilesCountVert; ++i, index += tilesCountHoriz)
        {            
            tiles[index]->scaleColumn(0, inTileHorizOffset, divisor);
        }
    }

    /////////////////////////////////////////
    void updateMain1(size_t step)
    {
        size_t leadColTileHorizIndex = step / tileCols;
        size_t leadRowTileVertIndex = step / tileRows;

        size_t inLeadColTileHorizOffset = step - leadColTileHorizIndex * tileCols;
        size_t inLeadRowTileVertOffset = step - leadRowTileVertIndex * tileRows;

            //

        size_t offset = step + 1;

        size_t tileHorizIndex = offset / tileCols;
        size_t tileVertIndex = offset / tileRows;

        size_t inTileHorizOffset = offset - tileHorizIndex * tileCols;
        size_t inTileVertOffset = offset - tileVertIndex * tileRows;

            //

        size_t index = tileVertIndex * tilesCountHoriz + tileHorizIndex;

        size_t leadColTileIndex = tileVertIndex * tilesCountHoriz + leadColTileHorizIndex;
        size_t leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + tileHorizIndex;

        Tile* curentTile = tiles[index];

        Tile* leadColTile = tiles[leadColTileIndex];
        float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);

        Tile* leadRowTile = tiles[leadRowTileIndex];

            // first tile

        curentTile->updateMain(inTileVertOffset, inTileHorizOffset, leadRowTile, inLeadRowTileVertOffset, leadColBlock);

            //

        size_t bulkStartHoriz = tileHorizIndex + 1;
        size_t bulkStartVert = tileVertIndex + 1;
        
            // next tiles in row

        ++index;
        ++leadRowTileIndex;

        for(size_t i = bulkStartHoriz; i < tilesCountHoriz; ++i, ++index, ++leadRowTileIndex)
        {
            curentTile = tiles[index];
            leadRowTile = tiles[leadRowTileIndex];

            curentTile->updateMain(inTileVertOffset, 0, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
        }     

            // next tiles in col

        leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + tileHorizIndex;
        leadRowTile = tiles[leadRowTileIndex];
        
        index += tileHorizIndex;
        
        leadColTileIndex += tilesCountHoriz;

        for(size_t i = bulkStartVert; i < tilesCountVert; ++i, index += tilesCountHoriz, leadColTileIndex += tilesCountHoriz) 
        {
            curentTile = tiles[index];

            leadColTile = tiles[leadColTileIndex];
            float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);

            curentTile->updateMain(0, inTileHorizOffset, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
        }   

            // main tiles bulk
                
        leadColTileIndex = bulkStartVert * tilesCountHoriz + leadColTileHorizIndex;

        for(size_t i = bulkStartVert; i < tilesCountVert; ++i, leadColTileIndex += tilesCountHoriz)
        {            
            leadColTile = tiles[leadColTileIndex];
            float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);
            
            leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + bulkStartHoriz;

            for(size_t j = bulkStartHoriz; j < tilesCountHoriz; ++j, ++leadRowTileIndex)
            {                
                leadRowTile = tiles[leadRowTileIndex];

                index = i * tilesCountHoriz + j;
                curentTile = tiles[index];
                
                curentTile->updateMain(0, 0, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
            }
        }    
    }

    void updateMain(size_t step)
    {
        size_t leadColTileHorizIndex = step / tileCols;
        size_t leadRowTileVertIndex = step / tileRows;

        size_t inLeadColTileHorizOffset = step - leadColTileHorizIndex * tileCols;
        size_t inLeadRowTileVertOffset = step - leadRowTileVertIndex * tileRows;

            //

        size_t offset = step + 1;

        size_t tileHorizIndex = offset / tileCols;
        size_t tileVertIndex = offset / tileRows;

        size_t inTileHorizOffset = offset - tileHorizIndex * tileCols;
        size_t inTileVertOffset = offset - tileVertIndex * tileRows;

            //

        size_t index = tileVertIndex * tilesCountHoriz + tileHorizIndex;

        size_t leadColTileIndex = tileVertIndex * tilesCountHoriz + leadColTileHorizIndex;
        size_t leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + tileHorizIndex;

        Tile* curentTile = tiles[index];

        Tile* leadColTile = tiles[leadColTileIndex];
        float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);

        Tile* leadRowTile = tiles[leadRowTileIndex];

            // first tile

        curentTile->updateMain(inTileVertOffset, inTileHorizOffset, leadRowTile, inLeadRowTileVertOffset, leadColBlock);

            //

        size_t bulkStartHoriz = tileHorizIndex + 1;
        size_t bulkStartVert = tileVertIndex + 1;
        
            // next tiles in row

        ++index;
        ++leadRowTileIndex;

        for(size_t i = bulkStartHoriz; i < tilesCountHoriz; ++i, ++index, ++leadRowTileIndex)
        {
            curentTile = tiles[index];
            leadRowTile = tiles[leadRowTileIndex];

            curentTile->updateMain(inTileVertOffset, 0, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
        }     

            // next tiles in col

        leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + tileHorizIndex;
        leadRowTile = tiles[leadRowTileIndex];
        
        index += tileHorizIndex;
        
        leadColTileIndex += tilesCountHoriz;

        for(size_t i = bulkStartVert; i < tilesCountVert; ++i, index += tilesCountHoriz, leadColTileIndex += tilesCountHoriz) 
        {
            curentTile = tiles[index];

            leadColTile = tiles[leadColTileIndex];
            float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);

            curentTile->updateMain(0, inTileHorizOffset, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
        }   

            // main tiles bulk
        
        class Apply
        {
            Tile** tiles;
            size_t tilesCountHoriz; 
            size_t leadColTileHorizIndex; 
            size_t inLeadColTileHorizOffset; 
            size_t inLeadRowTileVertOffset;
            size_t leadRowTileVertIndex;
            size_t bulkStartHoriz;                

        public:

            Apply
            (
                Tile** argTiles,
                size_t argTilesCountHoriz, 
                size_t argLeadColTileHorizIndex, 
                size_t argInLeadColTileHorizOffset, 
                size_t argInLeadRowTileVertOffset,
                size_t argLeadRowTileVertIndex,
                size_t argBulkStartHoriz                
            )
                :                    
                tiles(argTiles),
                tilesCountHoriz(argTilesCountHoriz),
                leadColTileHorizIndex(argLeadColTileHorizIndex),
                inLeadColTileHorizOffset(argInLeadColTileHorizOffset),
                inLeadRowTileVertOffset(argInLeadRowTileVertOffset),
                leadRowTileVertIndex(argLeadRowTileVertIndex),
                bulkStartHoriz(argBulkStartHoriz)  
            {}

            void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t startIndex = workItem.begin();
                size_t stopIndex = workItem.end();

                size_t leadColTileIndex = startIndex * tilesCountHoriz + leadColTileHorizIndex;

                for(size_t i = startIndex; i < stopIndex; ++i, leadColTileIndex += tilesCountHoriz)
                {                                
                    Tile* leadColTile = tiles[leadColTileIndex];
                    float* leadColBlock = leadColTile->tile + inLeadColTileHorizOffset * (leadColTile->tileRows);
                    
                    size_t leadRowTileIndex = leadRowTileVertIndex * tilesCountHoriz + bulkStartHoriz;

                    for(size_t j = bulkStartHoriz; j < tilesCountHoriz; ++j, ++leadRowTileIndex)
                    {                
                        Tile* leadRowTile = tiles[leadRowTileIndex];

                        size_t index = i * tilesCountHoriz + j;
                        Tile* curentTile = tiles[index];
                        
                        curentTile->updateMain(0, 0, leadRowTile, inLeadRowTileVertOffset, leadColBlock);
                    }
                }    
            }
        };

        parallel_for
        (
            blocked_range<size_t>(bulkStartVert, tilesCountVert), 
            Apply
            (
                tiles, 
                tilesCountHoriz, 
                leadColTileHorizIndex, 
                inLeadColTileHorizOffset, 
                inLeadRowTileVertOffset, 
                leadRowTileVertIndex, 
                bulkStartHoriz
            )
        ); 
    }

    /////////////////////////////////////////
    // debug
    void printMatrix()
    {
        for(size_t i = 0; i < tilesCountVert; ++i)        
        {
            size_t vert = tiles[i * tilesCountHoriz]->tileRows;

            for(size_t row = 0; row < vert; ++row)
            {
                for(size_t j = 0; j < tilesCountHoriz; ++j)
                {
                    Tile* currentTile = tiles[i * tilesCountHoriz + j];        

                    size_t horiz = currentTile->tileCols;

                    for(size_t col = 0; col < horiz; ++col)
                    {
                        printf("%+7.2f ", currentTile->tile[col * vert + row]);
                    }

                    printf("|");
                }

                printf("\n");
            }

            printf("\n");
        }

        printf("====\n");
    }

private:

    /////////////////////////////////////////
    void swapRows(Tile* t1, Tile* t2, size_t r1, size_t r2)
    {
            // t1 and t2 are in the same 'tile column'
        
        size_t width = t1->tileCols;

        size_t t1stride = t1->tileRows;
        size_t t2stride = t2->tileRows;

        float* pScan1 = t1->tile + r1;
        float* pScan2 = t2->tile + r2;

        for(size_t i = 0; i < width; ++i, pScan1 += t1stride, pScan2 += t2stride)
        {            
            float t = *pScan1;
            *pScan1 = *pScan2;
            *pScan2 = t;
        }                
    }    
};

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
         
    double* fp64Matrix;

    double* fp64Vector;

    int* permutations;

    double* fp64MatrixLup;

    double* solution;
    double* iterativeSolution;
    double* residuals;

    TiledMatrix* fp32TiledMatrix;

    /////////////////////////////////////////
    Solver() :        
        permutations(NULL),
        fp64MatrixLup(NULL),
        solution(NULL),
        iterativeSolution(NULL),
        residuals(NULL),               
        fp64Matrix(NULL),
        fp64Vector(NULL),
        fp32TiledMatrix(NULL)
    {}

    /////////////////////////////////////////
    void Dispose()
    {   
        aligned_free(iterativeSolution);
        aligned_free(solution);
        aligned_free(residuals);

        aligned_free(fp64MatrixLup);

        free(permutations);
            
        aligned_free(fp64Matrix);   
        aligned_free(fp64Vector);     

        delete fp32TiledMatrix;
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
        
        fp64Matrix = (double*)aligned_alloc(SSE_ALIGNMENT, fp64ResourceStride); 

        fp64Vector = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 

        solution = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 
        iterativeSolution = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 
        residuals = (double*)aligned_alloc(SSE_ALIGNMENT, fp64VectorStride); 

        permutations = (int*)malloc(expandedDimension * sizeof(int));

        fp64MatrixLup = (double*)aligned_alloc(SSE_ALIGNMENT, fp64ResourceStride); 

        fp32TiledMatrix = new TiledMatrix(dimension);

        return true;
    }

    /////////////////////////////////////////
    // debug
    void printMatrix(float* argMatrix)
    {
        for(size_t row = 0; row < dimension; ++row)
        {
            for(size_t col = 0; col < dimension; ++col)
            {
                printf("%6.2f ", argMatrix[row * dimension + col]);
            }

            printf("\n");
        }

        printf("\n");
    }

    /////////////////////////////////////////
    void useMatrix(float* argMatrix)
    {
        //printMatrix(argMatrix);

        fp32TiledMatrix->ImportFromRowMajor(argMatrix, dimension);            
        fp32TiledMatrix->ExportToRowMajorFp64(fp64Matrix, expandedDimension);   
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
        //fp32TiledMatrix->printMatrix();     

        for(size_t step = 0; step < dimension - 1; ++step)        
        {                                    
            int pivotIndex = fp32TiledMatrix->findPivot(step);            
            
            if(pivotIndex < 0)
            {
                return false;
            }
            
            permutations[step] = pivotIndex;

        //printf("{%d %d}\n", step, pivotIndex);

            if(pivotIndex != step)
            {
                fp32TiledMatrix->swapRows(step, pivotIndex);                
            }
            
            //if(step == 0)
                //fp32TiledMatrix->printMatrix();     

            fp32TiledMatrix->scaleColumn(step);                          

            //if(step == 0)
                //fp32TiledMatrix->printMatrix();     

            fp32TiledMatrix->updateMain(step);            

            //if(step == 0)
                //fp32TiledMatrix->printMatrix();                 
        }

        permutations[dimension - 1] = dimension - 1;

        fp32TiledMatrix->ExportToRowMajorFp64(fp64MatrixLup, expandedDimension);

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

private:

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
