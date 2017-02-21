#ifndef SOLVER9BLASAVX1_H
#define SOLVER9BLASAVX1_H

#include <tbb/tbb.h>

#include <math.h>
//#include <xmmintrin.h>
#include <immintrin.h>

#include <mkl.h>

#include "memalign.h"

#define TILE_SIDE 128

//-------------------------------------------------------------

using namespace tbb;

//-------------------------------------------------------------

struct Tile
{
    size_t tileRows;
    size_t tileCols;

    float* tile;

    /////////////////////////////////////////
    Tile(size_t argTileRows, size_t argTileCols) :
        tileRows(argTileRows),
        tileCols(argTileCols),
        tile((float*)aligned_alloc(CACHE_LINE, tileRows * tileCols * sizeof(float)))
    {}
    
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
    void ExtractRow(size_t rowIndex, float* dest)
    {
        float* pSrc = tile + rowIndex; 

        for(size_t col = 0; col < tileCols; ++col, pSrc += tileRows)
        {
            dest[col] = *pSrc;
        }
    }

    /////////////////////////////////////////
    int findPivotLeftmost()
    {
        int pivotIndex = -1;        
        float maxValue = 0;
        
        float* pScan = tile;

        for(size_t row = 0; row < tileRows; ++row, ++pScan)
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
    void scaleRightmost(float divisor)
    {
        float* pScan = tile + (tileCols - 1) * tileRows;

        if(tileRows == TILE_SIDE)
        {
            __m256 qdiv = _mm256_set1_ps(divisor);

            for(size_t row = 0; row < tileRows; row += AVX_BASE_COUNT)
            {   
                float* p = pScan + row;             
                
                _mm256_store_ps(p, _mm256_div_ps(*(__m256*)p, qdiv));
            }
        }
        else
        {
            for(size_t row = 0; row < tileRows; ++row, ++pScan)
            {
                *pScan /= divisor;
            }        
        }
    }

    /////////////////////////////////////////
    int updateWithScaleAndNextPivotSearch(size_t offsetVert, size_t offsetHoriz, const float* leadRowBlock)
    {
        // assert
        // offsetHoriz > 1
        // offsetHoriz < tileCols

        // leadRowBlock is continguous (extracted from col major into temporary buffer)

            //

        size_t leadColOffset = offsetHoriz - 1;

        float* leadCol = tile + leadColOffset * tileRows;

        float divisor = leadRowBlock[leadColOffset];

        scaleColumn(offsetVert, leadCol, divisor);

            // process leftmost col of update area, with next pivot search

        int pivotIndex = -1;        
        float maxValue = 0;

        float* column = leadCol + tileRows;

        float ldRowValue = leadRowBlock[offsetHoriz];

        for(size_t i = offsetVert; i < tileRows; ++i)
        {
            float v = (column[i] -= leadCol[i] * ldRowValue);

            float fav = fabs(v);

            if(fav > maxValue)
            {                
                maxValue = fav;
                pivotIndex = i;                
            }            
        }   
                        
            // process other cols of update area
        
        column += tileRows;

        processColumns(column, offsetHoriz + 1, offsetVert, leadRowBlock, leadCol);

        return pivotIndex; 
    }

    /////////////////////////////////////////
    void solveTriangle(const Tile* triTile)
    {   
        cblas_strsm
        (
            CblasColMajor, 
            CblasLeft, 
            CblasLower, 
            CblasNoTrans, 
            CblasUnit, 
            (MKL_INT)tileRows, 
            (MKL_INT)tileCols, 
            1, 
            triTile->tile, 
            (MKL_INT)tileRows, 
            tile, 
            (MKL_INT)tileRows
        );
    }

    /////////////////////////////////////////
    void subtractProduct(const Tile* left, const Tile* top)
    {
        size_t leftTileRows = left->tileRows;
        size_t leftTileCols = left->tileCols;
        //size_t topTileRows = top->tileRows; == leftTileCols
        
        cblas_sgemm 
        (
            CblasColMajor, 
            CblasNoTrans, 
            CblasNoTrans, 
            (MKL_INT)leftTileRows, 
            (MKL_INT)tileCols, 
            (MKL_INT)leftTileCols, 
            -1, 
            left->tile, 
            (MKL_INT)leftTileRows, 
            top->tile, 
            (MKL_INT)leftTileCols, 
            1, 
            tile, 
            (MKL_INT)leftTileRows
        );        
    }       

private:

    /////////////////////////////////////////
    inline void scaleColumn(size_t offsetVert, float* leadCol, float divisor)
    {
        if(tileRows == TILE_SIDE)
        {
            size_t extra = offsetVert % AVX_BASE_COUNT;
            size_t runStart = offsetVert + (extra ? (AVX_BASE_COUNT - extra) : 0);
            
            size_t i = offsetVert;

            for(; i < runStart; ++i)
            {
                leadCol[i] /= divisor;
            }
            
            __m256 qdiv = _mm256_set1_ps(divisor);

            for(; i < tileRows; i += AVX_BASE_COUNT)
            {   
                float* p = leadCol + i;             

                _mm256_store_ps(p, _mm256_div_ps(*(__m256*)p, qdiv));
            }
        }
        else
        {
            for(size_t i = offsetVert; i < tileRows; ++i)
            {
                leadCol[i] /= divisor;
            }
        }
    }

    /////////////////////////////////////////
    inline void processColumns
                (
                    float* columnData, 
                    size_t colStart, 
                    size_t offsetVert, 
                    const float* leadRowBlock, 
                    const float* leadCol
                )
    {
        if(tileRows == TILE_SIDE)
        {
            size_t extra = offsetVert % AVX_BASE_COUNT;
            size_t runStart = offsetVert + (extra ? (AVX_BASE_COUNT - extra) : 0);
                        
            for(size_t j = colStart; j < tileCols; ++j, columnData += tileRows)
            {
                float ldRowValue = leadRowBlock[j];

                size_t i = offsetVert;

                for(; i < runStart; ++i)
                {
                    columnData[i] -= leadCol[i] * ldRowValue;
                }

                __m256 qLdRowValue = _mm256_set1_ps(ldRowValue);

                for(; i < tileRows; i += AVX_BASE_COUNT)
                {
                    float* p = columnData + i;             
                    
                    _mm256_store_ps(p, _mm256_sub_ps(*(__m256*)p, _mm256_mul_ps(*(__m256*)(leadCol + i), qLdRowValue)));
                }   
            }            
        }
        else
        {
            for(size_t j = colStart; j < tileCols; ++j, columnData += tileRows)
            {
                float ldRowValue = leadRowBlock[j];

                for(size_t i = offsetVert; i < tileRows; ++i)
                {
                    columnData[i] -= leadCol[i] * ldRowValue;
                }   
            }            
        }
    }

    inline void processColumns1
            (
                float* columnData, 
                size_t colStart, 
                size_t offsetVert, 
                const float* leadRowBlock, 
                const float* leadCol
            )
    {
        size_t height = tileRows - offsetVert;

        const float* x = leadCol + offsetVert;

        float* y = columnData + offsetVert;

        for(size_t j = colStart; j < tileCols; ++j, y += tileRows)
        {            
            // y = -leadRowElement * x + y

            cblas_saxpy((MKL_INT)height, -leadRowBlock[j], x, 1, y, 1);
        }            
    }    
};

//-------------------------------------------------------------

struct TiledMatrix
{
    static const size_t tileRows = TILE_SIDE;
    static const size_t tileCols = tileRows;
    
    size_t dimension;

    size_t mainCountHoriz;
    size_t mainCountVert;

    size_t tilesCountHoriz;
    size_t tilesCountVert;
    
    size_t extraHoriz;
    size_t extraVert;

    Tile** tiles;

    float* leadRowBlock;

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

        leadRowBlock = (float*)aligned_alloc(CACHE_LINE, tileCols * sizeof(float));
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

        aligned_free(leadRowBlock);        
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
    bool factorizePanel(size_t panelDiagIndex, int* permutations)
    {           
        size_t topTileIndex = panelDiagIndex * tilesCountHoriz + panelDiagIndex;

        size_t diagSteps = tiles[topTileIndex]->tileCols;

            // find pivot in leftmost col

        class ReduceLeftmostPivot
        {
            Tile** tiles;
            size_t panelDiagIndex;
            size_t tilesCountHoriz;

        public:

            float pivotValue;
            int pivotIndex;

            ReduceLeftmostPivot
                (
                    Tile** argTiles, 
                    size_t argPanelDiagIndex, 
                    size_t argTilesCountHoriz
                ) : 
                tiles(argTiles),
                panelDiagIndex(argPanelDiagIndex),
                tilesCountHoriz(argTilesCountHoriz),
                pivotValue(0),
                pivotIndex(-1)
            {}

            ReduceLeftmostPivot(ReduceLeftmostPivot& x, split) : 
                tiles(x.tiles),
                panelDiagIndex(x.panelDiagIndex),
                tilesCountHoriz(x.tilesCountHoriz),
                pivotValue(0),
                pivotIndex(-1)
            {}
            
            inline void operator()(const blocked_range<size_t>& workItem)
            {
                float pv = pivotValue;
                int pi = pivotIndex;
                
                size_t start = workItem.begin();
                size_t stop = workItem.end();

                size_t index = start * tilesCountHoriz + panelDiagIndex;

                for(size_t i = start; i != stop; ++i, index += tilesCountHoriz)
                {
                    Tile* currentTile = tiles[index];

                    int localPivotIndex = currentTile->findPivotLeftmost(); 

                    if(localPivotIndex > -1)
                    {
                        float fav = fabs(currentTile->tile[localPivotIndex]);

                        if(fav > pv)
                        {
                            pv = fav;
                            pi = i * tileRows + localPivotIndex;
                        }
                    }
                }    

                pivotValue = pv;
                pivotIndex = pi;                            
            }

            void join(const ReduceLeftmostPivot& x)
            {
                if(x.pivotValue > pivotValue)
                {
                    pivotValue = x.pivotValue;
                    pivotIndex = x.pivotIndex; 
                }
            }
        };

        int pivotIndex = -1;
        float pivotValue = 0;

        Tile* currentTile;

        ReduceLeftmostPivot rlmp(tiles, panelDiagIndex, tilesCountHoriz);

        parallel_reduce(blocked_range<size_t>(panelDiagIndex, tilesCountVert), rlmp);

        pivotIndex = rlmp.pivotIndex;
        pivotValue = rlmp.pivotValue;

        if(pivotIndex < 0)
        {
            return false;            
        }

            //

        size_t topRowIndex = panelDiagIndex * tileRows;

        permutations[topRowIndex] = pivotIndex;

        if(pivotIndex != topRowIndex)
        {            
            swapRowsInPanel(panelDiagIndex, topRowIndex, pivotIndex);
        }

            // go on diagonal of top tile

        for(size_t diag = 1, matrixRow = topRowIndex + 1; diag < diagSteps; ++diag, ++matrixRow)
        {
            currentTile = tiles[topTileIndex];

            currentTile->ExtractRow(diag - 1, leadRowBlock);

            // update all tiles in 'tile column'

                // search next pivot while updating
    
            class ReduceNextPivotAndUpdate
            {
                size_t diag;
                Tile** tiles;
                size_t panelDiagIndex;
                size_t tilesCountHoriz;
                const float* leadRowBlock;

            public:

                float pivotValue;
                int pivotIndex;

                ReduceNextPivotAndUpdate
                    (
                        size_t argDiag,
                        Tile** argTiles, 
                        size_t argPanelDiagIndex, 
                        size_t argTilesCountHoriz,
                        const float* argLeadRowBlock                    
                    ) :
                    diag(argDiag),
                    tiles(argTiles),
                    panelDiagIndex(argPanelDiagIndex),
                    tilesCountHoriz(argTilesCountHoriz),
                    leadRowBlock(argLeadRowBlock),
                    pivotValue(0),
                    pivotIndex(-1)
                {}

                ReduceNextPivotAndUpdate(ReduceNextPivotAndUpdate& x, split) : 
                    diag(x.diag),
                    tiles(x.tiles),
                    panelDiagIndex(x.panelDiagIndex),
                    tilesCountHoriz(x.tilesCountHoriz),
                    leadRowBlock(x.leadRowBlock),
                    pivotValue(0),
                    pivotIndex(-1)
                {}
                
                inline void operator()(const blocked_range<size_t>& workItem)
                {
                    float pv = pivotValue;
                    int pi = pivotIndex;
                    
                    size_t start = workItem.begin();
                    size_t stop = workItem.end();

                    size_t index = start * tilesCountHoriz + panelDiagIndex;

                    for(size_t i = start; i != stop; ++i, index += tilesCountHoriz)
                    {
                        Tile* currentTile = tiles[index];

                        int localPivotIndex = currentTile->updateWithScaleAndNextPivotSearch(((i > panelDiagIndex) ? 0 : diag), diag, leadRowBlock);

                        if(localPivotIndex > -1)
                        {
                            float fav = fabs(currentTile->tile[diag * (currentTile->tileRows) + localPivotIndex]);

                            if(fav > pv)
                            {
                                pv = fav;
                                pi = i * tileRows + localPivotIndex;
                            }
                        }
                    }    

                    pivotValue = pv;
                    pivotIndex = pi;                            
                }

                void join(const ReduceNextPivotAndUpdate& x)
                {
                    if(x.pivotValue > pivotValue)
                    {
                        pivotValue = x.pivotValue;
                        pivotIndex = x.pivotIndex; 
                    }
                }                
            };

            pivotIndex = -1;
            pivotValue = 0;

            ReduceNextPivotAndUpdate rnpau(diag, tiles, panelDiagIndex, tilesCountHoriz, leadRowBlock);

            parallel_reduce(blocked_range<size_t>(panelDiagIndex, tilesCountVert), rnpau);

            pivotIndex = rnpau.pivotIndex;
            pivotValue = rnpau.pivotValue;

            if(pivotIndex < 0)
            {
                return false;
            }

            permutations[matrixRow] = pivotIndex;

            if(pivotIndex != matrixRow)
            {                
                swapRowsInPanel(panelDiagIndex, matrixRow, pivotIndex);
            }
        }
            
            // scale rightmost col
        
        currentTile = tiles[topTileIndex];

        float divisor = currentTile->tile[(currentTile->tileCols * currentTile->tileRows) - 1];

        class ApplyScaleRightmost
        {
            Tile** tiles;
            size_t panelDiagIndex;
            size_t tilesCountHoriz;
            float divisor;

        public:

            ApplyScaleRightmost
                (
                    Tile** argTiles,
                    size_t argPanelDiagIndex,
                    size_t argTilesCountHoriz,
                    float argDivisor
                ) : 
                tiles(argTiles),
                panelDiagIndex(argPanelDiagIndex),
                tilesCountHoriz(argTilesCountHoriz),
                divisor(argDivisor)
            {}

            inline void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t start = workItem.begin();
                size_t stop = workItem.end();

                size_t index = start * tilesCountHoriz + panelDiagIndex;

                for(size_t i = start; i < stop; ++i, index += tilesCountHoriz)
                {
                    tiles[index]->scaleRightmost(divisor);
                }                        
            }
        };
        
        parallel_for
        (
            blocked_range<size_t>(panelDiagIndex + 1, tilesCountVert), 
            ApplyScaleRightmost(tiles, panelDiagIndex, tilesCountHoriz, divisor)
        ); 

        return true;
    }

    /////////////////////////////////////////
    void triangleSolve(size_t panelDiagIndex, const int* permutations)
    {   
        class Apply
        {
            TiledMatrix* host;
            size_t panelDiagIndex;
            size_t tilesCountHoriz;            
            Tile** tiles;
            size_t rowsCount;
            const int* permutations;
            size_t permutStartIndex;
            size_t permutStopIndex;
                        
        public:

            Apply(TiledMatrix* argHost, size_t argPanelDiagIndex, size_t argTilesCountHoriz, Tile** argTiles, size_t argRowsCount, const int* argPermutations) :

                host(argHost),
                panelDiagIndex(argPanelDiagIndex), 
                tilesCountHoriz(argTilesCountHoriz),                         
                tiles(argTiles),
                rowsCount(argRowsCount),
                permutations(argPermutations)
            {                
                permutStartIndex = panelDiagIndex * rowsCount;
                permutStopIndex = permutStartIndex + rowsCount;                
            }    

            inline void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t start = workItem.begin();
                size_t stop = workItem.end();

                Tile* triTile = tiles[panelDiagIndex * tilesCountHoriz + panelDiagIndex];

                for(size_t i = start, index = panelDiagIndex * tilesCountHoriz + start; i < stop; ++i, ++index)
                {     
                        // make rows permutations
                    
                    for(size_t rowIndex = permutStartIndex; rowIndex < permutStopIndex; ++rowIndex)
                    {
                        int pi = permutations[rowIndex];

                        if(pi != rowIndex)
                        {
                            host->swapRowsInPanel(i, rowIndex, pi);
                        }                        
                    }
                        
                        // 
                    tiles[index]->solveTriangle(triTile);
                }
            }
        };

        parallel_for
        (
            blocked_range<size_t>(panelDiagIndex + 1, tilesCountHoriz), 
            Apply(this, panelDiagIndex, tilesCountHoriz, tiles, tileRows, permutations)
        ); 
    }

    /////////////////////////////////////////
    void updateTrailingSubmatrix(size_t panelDiagIndex)
    {
        size_t offset = panelDiagIndex + 1;
        
        size_t rowScanIndex = offset * tilesCountHoriz;

        size_t topStart = panelDiagIndex * tilesCountHoriz;

        class Apply
        {
            size_t panelDiagIndex;
            size_t tilesCountHoriz;
            Tile** tiles;
                        
        public:

            Apply(size_t argPanelDiagIndex, size_t argTilesCountHoriz, Tile** argTiles) :

                panelDiagIndex(argPanelDiagIndex),
                tilesCountHoriz(argTilesCountHoriz),
                tiles(argTiles)
            {}    

            inline void operator()(const blocked_range<size_t>& workItem) const
            {
                size_t start = workItem.begin();
                size_t stop = workItem.end();

                size_t offset = panelDiagIndex + 1;
                
                size_t rowScanIndex = start * tilesCountHoriz;

                size_t topStart = panelDiagIndex * tilesCountHoriz;

                for(size_t i = start; i < stop; ++i, rowScanIndex += tilesCountHoriz)
                {
                    Tile* left = tiles[rowScanIndex + panelDiagIndex];

                    for(size_t j = offset; j < tilesCountHoriz; ++j)
                    {                
                        tiles[rowScanIndex + j]->subtractProduct(left, tiles[topStart + j]);
                    }
                }
            }
        };

        parallel_for
        (
            blocked_range<size_t>(offset, tilesCountVert), 
            Apply(panelDiagIndex, tilesCountHoriz, tiles)
        ); 
    }

    /////////////////////////////////////////
    void updateTrailingSubmatrix1(size_t panelDiagIndex)
    {
        size_t offset = panelDiagIndex + 1;
        
        size_t rowScanIndex = offset * tilesCountHoriz;

        size_t topStart = panelDiagIndex * tilesCountHoriz;

        for(size_t i = offset; i < tilesCountVert; ++i, rowScanIndex += tilesCountHoriz)
        {
            Tile* left = tiles[rowScanIndex + panelDiagIndex];

            for(size_t j = offset; j < tilesCountHoriz; ++j)
            {                
                tiles[rowScanIndex + j]->subtractProduct(left, tiles[topStart + j]);
            }
        }        
    }
    
    /////////////////////////////////////////
    void swapRowsLeft(int* permutations)
    {
        for(size_t i = 1, permutStartIndex = tileRows; i < tilesCountVert; ++i, permutStartIndex += tileRows)        
        {
            size_t permutStopIndex = permutStartIndex + tiles[i * tilesCountHoriz]->tileRows;

            for(size_t j = 0; j < i; ++j)
            {
                // to do
                // if lengthy, may be try parallel?

                for(size_t rowIndex = permutStartIndex; rowIndex < permutStopIndex; ++rowIndex)
                {
                    int pi = permutations[rowIndex];

                    if(pi != rowIndex)
                    {
                        swapRowsInPanel(j, rowIndex, pi);
                    }                        
                }
            }
        }
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
    void swapRowsInPanel(size_t panelIndex, size_t r1, size_t r2)
    {        
        size_t vertIndex1 = r1 / tileRows;
        size_t vertIndex2 = r2 / tileRows;

        size_t inTileR1 = r1 - vertIndex1 * tileRows;
        size_t inTileR2 = r2 - vertIndex2 * tileRows;

        size_t index1 = vertIndex1 * tilesCountHoriz + panelIndex;
        size_t index2 = vertIndex2 * tilesCountHoriz + panelIndex;

        Tile* t1 = tiles[index1];

        if(vertIndex1 != vertIndex2)
        {            
            Tile* t2 = tiles[index2];

            size_t width = t1->tileCols;

            size_t t1stride = t1->tileRows;
            size_t t2stride = t2->tileRows;

            float* pScan1 = t1->tile + inTileR1;
            float* pScan2 = t2->tile + inTileR2;

            for(size_t i = 0; i < width; ++i, pScan1 += t1stride, pScan2 += t2stride)
            {   
                float t = *pScan1;
                *pScan1 = *pScan2;
                *pScan2 = t;
            }                            
        }
        else
        {
            t1->swapRows(inTileR1, inTileR2);    
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

        extra = dimension % AVX_BASE_COUNT;        
        expandedDimension = dimension + (extra ? (AVX_BASE_COUNT - extra) : 0);                
        trail = expandedDimension - dimension;
        
        fp32VectorStride = expandedDimension * sizeof(float);        
        fp32ResourceStride = dimension * fp32VectorStride;

        fp64VectorStride = expandedDimension * sizeof(double);
        fp64ResourceStride = dimension * fp64VectorStride;
        
        fp64Matrix = (double*)aligned_alloc(AVX_ALIGNMENT, fp64ResourceStride); 

        fp64Vector = (double*)aligned_alloc(AVX_ALIGNMENT, fp64VectorStride); 

        solution = (double*)aligned_alloc(AVX_ALIGNMENT, fp64VectorStride); 
        iterativeSolution = (double*)aligned_alloc(AVX_ALIGNMENT, fp64VectorStride); 
        residuals = (double*)aligned_alloc(AVX_ALIGNMENT, fp64VectorStride); 

        permutations = (int*)malloc(expandedDimension * sizeof(int));

        fp64MatrixLup = (double*)aligned_alloc(AVX_ALIGNMENT, fp64ResourceStride); 

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

            // to do: use tilesCount (square matrices)

        size_t diagCount = fp32TiledMatrix->tilesCountHoriz;

        for(size_t step = 0; step < diagCount; ++step)
        {
            if(fp32TiledMatrix->factorizePanel(step, permutations))
            {                
                if(step < diagCount - 1)
                {                    
                    fp32TiledMatrix->triangleSolve(step, permutations);                    
                    fp32TiledMatrix->updateTrailingSubmatrix(step);                
                }
            }
            else
            {
                return false;
            }
        }

        fp32TiledMatrix->swapRowsLeft(permutations);

        //fp32TiledMatrix->printMatrix();

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
            blocked_range<size_t>(0, dimension, 8 * AVX_BASE_COUNT), 
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

    /////////////////////////////////////////
    double CalcResiduals1()
    {
        memcpy(residuals, fp64Vector, fp64VectorSize);

        cblas_dgemv 
        (
            CblasRowMajor, 
            CblasNoTrans, 
            dimension, 
            dimension, 
            -1, 
            fp64Matrix, 
            (MKL_INT)expandedDimension, 
            solution, 
            1, 
            1, 
            residuals, 
            1
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
    void useLupToSolve1(double* x, double* b)
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
        //*
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
        //*/
        /*
        cblas_dtrsm 
        (
            CblasRowMajor, 
            CblasLeft, 
            CblasLower, 
            CblasNoTrans, 
            CblasUnit, 
            (MKL_INT)dimension, 
            1, 
            1, 
            fp64MatrixLup, 
            (MKL_INT)expandedDimension, 
            x, 
            1
        );
        */
        cblas_dtrsm 
        (
            CblasRowMajor, 
            CblasLeft, 
            CblasUpper, 
            CblasNoTrans, 
            CblasNonUnit, 
            (MKL_INT)dimension, 
            1, 
            1, 
            fp64MatrixLup, 
            (MKL_INT)expandedDimension, 
            x, 
            1
        );
    }    
};

//-------------------------------------------------------------

#endif
