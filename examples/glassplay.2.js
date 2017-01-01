//------------------------------------------------------------------------------

function fillTestSystem(knownX, matrix, vector)
{
    var N = vector.length;

    for(var i = 0, index = 0; i < N; ++i)
    {
        var s = 0;
        
        for(var j = 0; j < N; ++j, ++index)
        {
            var a = matrix[index] = Math.floor(Math.random() * 10);
            
            s += a * knownX[j];
        }
        
        vector[i] = s;
    }
}

//------------------------------------------------------------------------------

function augmentMatrix(matrix, vector, augmentedMatrix)
{
    var N = vector.length;

        // write matrix cols as rows of augmented matrix
        // (transposed)
    
    for(var i = 0, augIndex = 0; i < N; ++i)
    {
        for(var j = 0; j < N; ++j, ++augIndex)
        {
            augmentedMatrix[augIndex] = matrix[j * N + i]; 
        }
    } 
    
        // write vector B as last row of augmented matrix
    
    for(var i = 0; i < N; ++i, ++augIndex)
    {
        augmentedMatrix[augIndex] = vector[i]; 
    }
}

//------------------------------------------------------------------------------

function printAugmentedMatrix(dim, augmentedMatrix)
{
    for(var i = 0; i < dim; ++i)
    {
        var tmp = [];
        
        for(var j = 0; j <= dim; ++j)
        {
            tmp.push(augmentedMatrix[j * dim + i]);
        }

        console.log(tmp);        
    }
}

//------------------------------------------------------------------------------

function findLeadIndex(valuesRow, length)
{
    for(var i = 0; i < length; ++i)
    {
        if(Math.abs(valuesRow[i]) > 0)
        {
            return i;
        }
    }
    
    return -1;
}

//------------------------------------------------------------------------------

function combineAugmentedMatrix(dim, doneLeft, dataLast, dataPrevious)
{
    // last col from dataLast, prev-last from dataPrevious, and cycle
    
    var dataDest = dataLast;
    var dataSrc = dataPrevious;

    for(var col = doneLeft - 1; col >= 0; col -= 2)
    {
        for(var row = 0; row <= dim; ++row)
        {
            var index = row * dim + col;
            
            dataDest[index] = dataSrc[index];
        }
    }
}

//------------------------------------------------------------------------------

function eliminateRect(dim, left, top, augmentedMatrix, leadIndex)
{
    var rowOffset;
    var rowLeftOffset;
    var rowLeadOffset;
    var topOffset;
    
    var currentOffset;
    
    var tmp;
        
        // divide leading col by its leading element; swap cols, if needed
        
    var leadDivisor;
        
    for(var row = top; row <= dim; ++row)
    {
        rowOffset = row * dim;
        
        rowLeadOffset = rowOffset + leadIndex;
        
        if(row === top)
        {
            topOffset = rowOffset;
            leadDivisor = augmentedMatrix[rowLeadOffset];
        }
        
        augmentedMatrix[rowLeadOffset] /= leadDivisor;
        
        if(leadIndex !== left)
        {
            rowLeftOffset = rowOffset + left;
            
            tmp = augmentedMatrix[rowLeftOffset];
            
            augmentedMatrix[rowLeftOffset] = augmentedMatrix[rowLeadOffset];
            
            augmentedMatrix[rowLeadOffset] = tmp;
        }
    }
        
        // subtract every col from leading col
    
    var divisor;
    
    for(var col = left + 1; col < dim; ++col)
    {
        divisor = augmentedMatrix[topOffset + col];
        
        if(Math.abs(divisor) > 0)
        {
            for(var row = top; row <= dim; ++row)
            {
                rowOffset = row * dim;
                
                rowLeftOffset = rowOffset + left;   
                currentOffset = rowOffset + col;
                
                augmentedMatrix[currentOffset] = augmentedMatrix[rowLeftOffset] - augmentedMatrix[currentOffset] / divisor;
            }
        }
    }
}

//------------------------------------------------------------------------------

function continueElimination(dim, left, top, augmentedMatrix)
{
    for(; top < dim; ++top)
    {
        var leadIndex = -1;
        
        var offset = top * dim;
        
        for(var i = left, index = offset + left; i < dim; ++i, ++index)
        {
            var m = augmentedMatrix[index];
            
            if(Math.abs(m) > 0)
            {
                leadIndex = i;
                break;
            }
        }
        
        if(leadIndex < 0)
        {
            continue;
        }
        
        eliminateRect(dim, left, top, augmentedMatrix, leadIndex);
        
        ++left;
    }
}

//------------------------------------------------------------------------------

function doGaussBackSteps(dim, augmentedMatrix, solution)
{
    var lastRowOffset = dim * dim;
    
    for(var col = dim - 1; col >= 0; --col)
    {
        var b = augmentedMatrix[lastRowOffset + col];
        
        for(var i = dim - 1; i > col; --i)
        {
            b -= augmentedMatrix[i * dim + col] * solution[i];
        }
        
        solution[col] = b;    
    }
}

//------------------------------------------------------------------------------

function calcErrorSquare(matrix, x, vector)
{
    var dim = x.length;

    var errSquare = 0;
    
    var index = 0;
    
    for(var i = 0; i < dim; ++i)
    {
        var s = 0;
        
        for(var j = 0; j < dim; ++j)
        {
            s += matrix[index] * x[j];
                        
            ++index;
        }
        
        var err = (vector[i] - s);
        
        errSquare += (err * err);
    }
    
    return errSquare;
}

//------------------------------------------------------------------------------

$(document).ready(() => 
{
    var glass = new Glass();
    
        //
        
    //const dim = 4;
    const dim = 100;
    const aug_height = dim + 1;
    
    const minTargetSquare = 16 * 16;
    
        // initialize test linear equations system
        
    var knownX = new Float32Array(dim);

    knownX.forEach((v, index, arr) => {arr[index] = Math.floor(Math.random() * 10)});
    
    var matrixSrc = [   0,  3,  -1, 8,
                        10, -1, 2,  0,
                        -1,	11, -1, 3,
                        2,	-1, 10, -1  ];
    
    //var matrix = new Float32Array(matrixSrc);

    var matrix = new Float32Array(dim * dim);
    
    //var vector = new Float32Array([35, 14, 30, 26]);

    var vector = new Float32Array(dim);   

    fillTestSystem(knownX, matrix, vector); 
    
    var t1 = Date.now();
    
    var transferMatrix = new Float32Array(dim * aug_height);
    
    augmentMatrix(matrix, vector, transferMatrix);
    
    var transferMatrixConverted = new Uint8Array(transferMatrix.buffer);

        // create ping-pong resources
    
    var pp = [];
    
    var ppIndexInput = 0;
    var ppIndexOutput = 1;

    for(var i = 0; i < 2; ++i)
    {
        var r = glass.createResource(dim, aug_height, glass.gl.RGBA, glass.gl.UNSIGNED_BYTE);

        pp[i] = {resource: r, target: glass.createOutputTarget(r)};
        
        glass.fillResource(pp[i].resource, transferMatrixConverted);
    }

        //

    var kernelSource = 
            
        glass.buildDefines({'WIDTH' : dim, 'HEIGHT' : aug_height}) + 
        $('#commonKernelRoutines').text() + 
        $('#eliminator3').text();
    
    var kernel = glass.createKernel(kernelSource);
    
    if(!kernel)
    {
        console.log(glass.errorLog);
    }

    glass.setKernel(kernel);
    
    glass.linkInputSampler(0, 'matrix');

    var leftTopLoc = glass.gl.getUniformLocation(kernel, 'leftTop');
    var leadIndexLoc = glass.gl.getUniformLocation(kernel, 'leadIndex');
    
        //
        
    var backData = [];
    
    for(var i = 0; i < 2; ++i)
    {   
        var resourcePack = {};
        
        resourcePack.float = new Float32Array(dim * aug_height);
        resourcePack.ubyte = new Uint8Array(resourcePack.float.buffer);
        
        backData[i] = resourcePack;
    }
        //
    
    var doneTop;    
    var doneLeft;
    
    var leadIndex = findLeadIndex(transferMatrix, dim);

    for(var top = 0, left = 0; top < dim;)
    {
        var dispatchWidth = dim - left;
        var dispatchHeight = aug_height - top;

        glass.gl.uniform2f(leftTopLoc, left, top);
        glass.gl.uniform1f(leadIndexLoc, leadIndex);

        glass.attachInputResource(0, pp[ppIndexInput].resource);
        glass.setOutputTarget(pp[ppIndexOutput].target);
        
        glass.dispatchKernel(left, top, dispatchWidth, dispatchHeight);
        
        doneTop = top;
        doneLeft = left;
        
        ++top;
        
        if(leadIndex >= 0)
        {
            ++left;    
        }
        //*
        if(dispatchWidth * dispatchHeight < minTargetSquare)
        {
            break;    
        }
        //*/
        if(top < dim)
        {
            var readBackLength = dim - left;
            
            var activeTargetBackup = backData[ppIndexOutput];
            
            glass.readBack(activeTargetBackup.ubyte, left, top, readBackLength, 1);

            leadIndex = findLeadIndex(activeTargetBackup.float, readBackLength);
            
            if(leadIndex >= 0)
            {
                leadIndex += left;
            }
            
            ppIndexInput = 1 - ppIndexInput;
            ppIndexOutput = 1 - ppIndexOutput;
        }
    }

    glass.setOutputTarget(pp[ppIndexOutput].target);
    glass.readBack(backData[0].ubyte, 0, 0, dim, aug_height);
    
    glass.setOutputTarget(pp[1 - ppIndexOutput].target);
    glass.readBack(backData[1].ubyte, 0, 0, dim, aug_height);  

    combineAugmentedMatrix(dim, doneLeft, backData[0].float, backData[1].float)
    
        // debug count diagonal
    
    console.log('done: ' + doneLeft + ', ' + doneTop);
    
    var units = 0;
    
    for(var i = 0; i < dim; ++i)
    {
        if(backData[0].float[i * dim + i] === 1)
        {
            ++units;
        }
        else
        {
            break;
        }
    }
    
    console.log(units);
    
    //printAugmentedMatrix(dim, backData[0].float);
    
        // continue on CPU
        
    continueElimination(dim, left, top, backData[0].float);
    
    var solution = new Float32Array(dim);
    
    doGaussBackSteps(dim, backData[0].float, solution);
    
    var t2 = Date.now();
    
    console.log('Done in ' + (t2 - t1) + ' ms');
    
    console.log(solution);
    console.log(knownX);
    
    var serr = 0;
    
    for(var i = 0; i < dim; ++i)
    {
        var e = solution[i] - knownX[i];
        
        serr += e * e;
    }
    
    console.log(serr);
    
    console.log(calcErrorSquare(matrix, solution, vector));
    
    //printAugmentedMatrix(dim, backData[0].float);
});

//------------------------------------------------------------------------------