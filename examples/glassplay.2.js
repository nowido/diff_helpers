//------------------------------------------------------------------------------

function GlassSolver(glass, dimension)
{
    var solver = this;
    
    solver.glass = glass;
        
        // use dimensions
        
    solver.dimension = dimension;
    var aug_height = solver.aug_height = dimension + 1;
    
    var surfaceSize = solver.surfaceSize = dimension * aug_height;
    
        // create dimension-dependend typed arrays
    
    solver.x = new Float64Array(dimension);
    solver.partialMatrix = new Float64Array(surfaceSize);
    
        // create ping-pong resources
    
    solver.pp = [];
    
    for(var i = 0; i < 2; ++i)
    {
        var r = glass.createResource(dimension, aug_height, glass.gl.RGBA, glass.gl.UNSIGNED_BYTE);
        
        solver.pp[i] = {resource: r, target: glass.createOutputTarget(r)};
    }
    
        // create GPU program stuff

    var kernelSource = 
            
        glass.buildDefines({'WIDTH' : dimension, 'HEIGHT' : aug_height}) + 
        $('#commonKernelRoutines').text() + 
        $('#eliminator3').text();
    
    var kernel = glass.createKernel(kernelSource);
    
    if(!kernel)
    {
        console.log(glass.errorLog);
    }

    glass.setKernel(kernel);
    
    glass.linkInputSampler(0, 'matrix');

    solver.leftTopLoc = glass.gl.getUniformLocation(kernel, 'leftTop');
    solver.leadIndexLoc = glass.gl.getUniformLocation(kernel, 'leadIndex');
    
        // create readback surfaces (one if them also used for initial data transfer)
        
    solver.backData = [];
    
    for(var i = 0; i < 2; ++i)
    {   
        var resourcePack = {};
        
        resourcePack.float = new Float32Array(surfaceSize);
        resourcePack.ubyte = new Uint8Array(resourcePack.float.buffer);
        
        solver.backData[i] = resourcePack;
    }
    
        //
        
    solver.minTargetSquare = 300 * 300;
}

//------------------------------------------------------------------------------

GlassSolver.prototype.useMatrix = function(matrix)
{
    this.matrix = matrix;
}

//------------------------------------------------------------------------------

GlassSolver.prototype.useVector = function(vector)
{
    this.vector = vector;
}

//------------------------------------------------------------------------------

GlassSolver.prototype.augmentMatrix = function()
{
    var solver = this;
    
    var dimension = solver.dimension;
    
    var matrix = solver.matrix;
    var vector = solver.vector;
    
    var augmentedMatrix = solver.backData[0].float;

        // write matrix cols as rows of augmented matrix
        // (transposed)
    
    for(var i = 0, augIndex = 0; i < dimension; ++i)
    {
        for(var j = 0; j < dimension; ++j, ++augIndex)
        {
            augmentedMatrix[augIndex] = matrix[j * dimension + i]; 
        }
    } 
    
        // write vector B as last row of augmented matrix
    
    for(var i = 0; i < dimension; ++i, ++augIndex)
    {
        augmentedMatrix[augIndex] = vector[i]; 
    }
}

//------------------------------------------------------------------------------

GlassSolver.prototype.combineAugmentedMatrix = function(doneLeft)
{
    var solver = this;
    
    var dimension = solver.dimension;
    
        // take last col from last rendered data, prev-last col from prev-last rendered data,
        // and so on cycle

    var dataDest = solver.backData[0].float;
    var dataSrc = solver.backData[1].float;
    
    var index;
    
    for(var col = doneLeft - 1; col >= 0; col -= 2)
    {
        for(var row = 0; row <= dimension; ++row)
        {
            index = row * dimension + col;
            
            dataDest[index] = dataSrc[index];
        }
    }
    
        // copy rendered single-precision data to double-precision tmp resource
        
        // to do: no need to copy full matrix
        
    var partialMatrix = solver.partialMatrix;
    
    var surfaceSize = solver.surfaceSize;

    for(index = 0; index < surfaceSize; ++index)
    {
        partialMatrix[index] = dataDest[index];
    }
}

//------------------------------------------------------------------------------

GlassSolver.prototype.findLeadIndex = function(valuesRow, length)
{
    var leadIndex = -1;
    
    var maxElement;
    
    for(var i = 0; i < length; ++i)
    {
        var el = Math.abs(valuesRow[i]);
        
        if(el > 0)
        {
            if((leadIndex < 0) || (el > maxElement))
            {
                leadIndex = i;
                maxElement = el;
            }
        }
    }
    
    return leadIndex;
}

//------------------------------------------------------------------------------

GlassSolver.prototype.eliminateRect = function(left, top, leadIndex)
{
    var solver = this;
    
    var dimension = solver.dimension;
    
    var augmentedMatrix = solver.partialMatrix;
    
    var rowOffset;
    var rowLeftOffset;
    var rowLeadOffset;
    var topOffset;
    
    var currentOffset;
    
    var tmp;
        
        // divide leading col by its leading element; swap cols, if needed
        
    var leadDivisor;
        
    for(var row = top; row <= dimension; ++row)
    {
        rowOffset = row * dimension;
        
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
    
    for(var col = left + 1; col < dimension; ++col)
    {
        divisor = augmentedMatrix[topOffset + col];
        
        if(Math.abs(divisor) > 0)
        {
            for(var row = top; row <= dimension; ++row)
            {
                rowOffset = row * dimension;
                
                rowLeftOffset = rowOffset + left;   
                currentOffset = rowOffset + col;
                
                augmentedMatrix[currentOffset] = augmentedMatrix[rowLeftOffset] - augmentedMatrix[currentOffset] / divisor;
            }
        }
    }
}

GlassSolver.prototype.continueElimination = function(left, top)
{
    var solver = this;
    
    var dimension = solver.dimension;
    
    var augmentedMatrix = solver.partialMatrix;
    
    for(; top < dimension; ++top)
    {
        var leadIndex = -1;
        
        var maxElement;
        
        var offset = top * dimension;
        
        for(var i = left, index = offset + left; i < dimension; ++i, ++index)
        {
            var el = Math.abs(augmentedMatrix[index]);
            
            if(el > 0)
            {
                if((leadIndex < 0) || (el > maxElement))
                {
                    leadIndex = i;    
                    maxElement = el;
                }
            }
        }
        
        if(leadIndex < 0)
        {
            continue;
        }
        
        solver.eliminateRect(left, top, leadIndex);
        
        ++left;
    }
}

//------------------------------------------------------------------------------

GlassSolver.prototype.doGaussBackSteps = function()
{
    var solver = this;
    
    var dimension = solver.dimension;
    
    var augmentedMatrix = solver.partialMatrix;
    
    var x = solver.x;
    
    var lastRowOffset = dimension * dimension;
    
    for(var col = dimension - 1; col >= 0; --col)
    {
        var b = augmentedMatrix[lastRowOffset + col];
        
        for(var i = dimension - 1; i > col; --i)
        {
            b -= augmentedMatrix[i * dimension + col] * x[i];
        }
        
        x[col] = b;    
    }
}

//------------------------------------------------------------------------------

GlassSolver.prototype.solve = function()
{
    var solver = this;    
    
    var glass = solver.glass;
    
        // cache dimension values 
        
    var dimension = solver.dimension;
    var aug_height = solver.aug_height;
    
        // cache uniform locations
        
    var leftTopLoc = solver.leftTopLoc;
    var leadIndexLoc = solver.leadIndexLoc;
        
        // cache GPU memory-map resources
        
    var pp = solver.pp;
    var backData = solver.backData;
    
        // cache other stuff
        
    var minTargetSquare = solver.minTargetSquare;
    
        // push initial data to GPU
    
    solver.augmentMatrix();

    for(var i = 0; i < 2; ++i)
    {
        glass.fillResource(solver.pp[i].resource, backData[0].ubyte);
    }

        // main dispatch cycle
        
    var ppIndexInput = 0;
    var ppIndexOutput = 1;

    var doneTop;    
    var doneLeft;
    
    var leadIndex = solver.findLeadIndex(backData[0].float, dimension);

    for(var top = 0, left = 0; top < dimension;)
    {
        var dispatchWidth = dimension - left;
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
        
        if(dispatchWidth * dispatchHeight < minTargetSquare)
        {
            break;    
        }
        
        if(top < dimension)
        {
            var readBackLength = dimension - left;
            
            var activeTargetBackup = backData[ppIndexOutput];
            
            glass.readBack(activeTargetBackup.ubyte, left, top, readBackLength, 1);

            leadIndex = solver.findLeadIndex(activeTargetBackup.float, readBackLength);
            
            if(leadIndex >= 0)
            {
                leadIndex += left;
            }
            
            ppIndexInput = 1 - ppIndexInput;
            ppIndexOutput = 1 - ppIndexOutput;
        }
    }
    
    glass.readBack(backData[0].ubyte, 0, 0, dimension, aug_height);
    
    glass.setOutputTarget(pp[1 - ppIndexOutput].target);
    glass.readBack(backData[1].ubyte, 0, 0, dimension, aug_height);  

    solver.combineAugmentedMatrix(doneLeft);
    
        // continue rest of computations on CPU

    solver.continueElimination(left, top);

    solver.doGaussBackSteps();
}

//------------------------------------------------------------------------------

GlassSolver.prototype.calcErrorSquare = function()
{
    var solver = this;    

    var dimension = solver.dimension;

    var matrix = solver.matrix;
    var vector = solver.vector;
    var x = solver.x;

    var errSquare = 0;
    
    var index = 0;
    
    for(var i = 0; i < dimension; ++i)
    {
        var s = 0;
        
        for(var j = 0; j < dimension; ++j)
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

GlassSolver.prototype.calcResiduals = function(residuals)
{
    var solver = this;    

    var dimension = solver.dimension;

    var matrix = solver.matrix;
    var vector = solver.vector;
    var x = solver.x;

    var errSquare = 0;
    
    var index = 0;
    
    for(var i = 0; i < dimension; ++i)
    {
        var s = 0;
        
        for(var j = 0; j < dimension; ++j)
        {
            s += matrix[index] * x[j];
                        
            ++index;
        }
        
        var err = residuals[i] = (vector[i] - s);
        
        errSquare += (err * err);
    }
    
    return errSquare;
}

//------------------------------------------------------------------------------

GlassSolver.prototype.iterate = function(count)
{
    var solver = this;    

    var dimension = solver.dimension;
    
    var vector = solver.vector;
    var x = solver.x;
    
    var solution = new Float32Array(dimension);
    var residuals = new Float32Array(dimension);

    solver.solve();  
    
    for(var j = 0; j < dimension; ++j)
    {
        solution[j] = x[j];
    }
    
    var errsq;
    
    for(var i = 0; i < count - 1; ++i)
    {
        errsq = solver.calcResiduals(residuals);
        
        console.log('-- ' + errsq);
        
        solver.useVector(residuals);
        
        solver.solve();
        
        for(var j = 0; j < dimension; ++j)
        {
             x[j] += solution[j];
             
             solution[j] = x[j];
        }
        
        solver.useVector(vector);
    }
    
    errsq = solver.calcResiduals(residuals);
    
    console.log('-- ' + errsq);
}

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

$(document).ready(() => 
{
    var glass = new Glass();
    
        //
        
    //const dim = 4;
    const dim = 1000;
    const aug_height = dim + 1;

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
    
        //
            
    var solver = new GlassSolver(glass, dim);
    
    solver.useMatrix(matrix);
    solver.useVector(vector);
    
    var t1 = Date.now();

    solver.iterate(3);    
    
    var t2 = Date.now();
    
    $(document.body).append('<p>Done in ' + (t2 - t1) + ' ms</p>');
    
    var serr = 0;

    for(var i = 0; i < dim; ++i)
    {
        var e = solver.x[i] - knownX[i];

        serr += e * e;
    }
    
    console.log(serr);
    
    $(document.body).append('<p>' + serr + '</p>');
});

//------------------------------------------------------------------------------
