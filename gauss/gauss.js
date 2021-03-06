//------------------------------------------------------------------------------

function Solver(dimension)
{
    var solver = this;
    
    solver.dimension = dimension;
    
    solver.x = new Float64Array(dimension);
}

Solver.prototype.useMatrix = function(matrix)
{
    this.matrix = new Float64Array(matrix);
}

Solver.prototype.useVector = function(vector)
{
    this.vector = new Float64Array(vector);
}

Solver.prototype.solve = function()
{
    var solver = this;
    
    var dim = solver.dimension;
    
        // forward
        
    for(var row = 0, col = 0; col < dim; ++col)
    {
        var leadingRow = solver.findLeadingRow(row, col);    
        
        if(leadingRow < 0)
        {
            continue;
        }
        else if(leadingRow !== row)
        {
            solver.swapRows(row, leadingRow, col);  
            solver.eliminate1(row, col);
        }
        else
        {
            solver.eliminate2(row, col);        
        }
        
        ++row;
    }
    
        // backward
        
    var matrix = solver.matrix;    
    var vector = solver.vector;    
    var x = solver.x;
    
    for(var row = dim - 1; row >= 0; --row)    
    {
        var s = 0;
        
        for(var col = dim - 1, offset = row * dim + col; col > row; --col, --offset)
        {
            s += matrix[offset] * x[col];
        }
        
        x[row] = vector[row] - s;
    }
}

Solver.prototype.findLeadingRow = function(top, left)
{
    var solver = this;
    
    var dim = solver.dimension;
    
    var matrix = solver.matrix;

    var leadingRowIndex = -1;
    
    var maxElement;
    
    for(var row = top; row < dim; ++row)
    {
        var v = Math.abs(matrix[row * dim + left]);
        
        if(v > 0)
        {
            if((leadingRowIndex < 0) || (v > maxElement))
            {
                leadingRowIndex = row;    
                maxElement = v;
            }
        }
    }
    
    return leadingRowIndex;
}

Solver.prototype.swapRows = function(row, foundLeadingRow, left)
{
    var solver = this;
    
    var dim = solver.dimension;

    var matrix = solver.matrix;
    
    var offset = row * dim + left;
    var leadingOffset = foundLeadingRow * dim + left;
    
    var leadingDivisor = matrix[leadingOffset];
    
    var tmp;
    
    for(var i = left; i < dim; ++i, ++offset, ++leadingOffset)
    {
        tmp = matrix[leadingOffset];
        matrix[leadingOffset] = matrix[offset];
        matrix[offset] = tmp / leadingDivisor;
    }
    
    var vector = solver.vector;
    
    tmp = vector[foundLeadingRow];
    vector[foundLeadingRow] = vector[row];
    vector[row] = tmp / leadingDivisor;
}

Solver.prototype.eliminate1 = function(top, left)
{
    var solver = this;
    
    var dim = solver.dimension;

    var matrix = solver.matrix;
        
        // top row was already divided by its lead, now process lower rows 
    
    var topOffset = top * dim;    
    
    for(var row = top + 1; row < dim; ++row)
    {
        var offset = row * dim + left;
        
        var element = matrix[offset];
        
        if(Math.abs(element) > 0)
        {
            for(var col = left; col < dim; ++col, ++offset)
            {
                matrix[offset] = matrix[topOffset + col] - matrix[offset] / element;    
            }
            
            solver.vector[row] = solver.vector[top] - solver.vector[row] / element;
        }
    }
}

Solver.prototype.eliminate2 = function(top, left)
{
    var solver = this;
    
    var dim = solver.dimension;

    var matrix = solver.matrix;
        
        // divide top row by leading element
    
    var topOffset = top * dim;    
    
    var offset = topOffset + left;

    var divisor = matrix[offset];
    
    for(var i = left; i < dim; ++i, ++offset)
    {
        matrix[offset] /= divisor;
    }
    
    solver.vector[top] /= divisor;
    
        // process lower rows
    
    for(var row = top + 1; row < dim; ++row)
    {
        offset = row * dim + left;
        
        var element = matrix[offset];
        
        if(Math.abs(element) > 0)
        {
            for(var col = left; col < dim; ++col, ++offset)
            {
                matrix[offset] = matrix[topOffset + col] - matrix[offset] / element;    
            }
            
            solver.vector[row] = solver.vector[top] - solver.vector[row] / element;
        }
    }
}

Solver.prototype.calcErrorSquare = function()
{
    var solver = this;
    
    var dim = solver.dimension;
    
    var matrix = solver.matrix;
    var vector = solver.vector;
    
    var x = solver.x;    
    
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

Solver.prototype.calcResiduals = function(residuals)
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

Solver.prototype.iterate = function(count)
{
    var solver = this;    

    var dimension = solver.dimension;
    
    var vector = solver.vector;
    var x = solver.x;
    
    var solution = new Float64Array(dimension);
    var residuals = new Float64Array(dimension);

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

$(document).ready(() => 
{
    const dim = 1000;
    
    var knownX = new Float64Array(dim);

    knownX.forEach((v, index, arr) => {arr[index] = Math.floor(Math.random() * 10)});

    var matrix = new Float64Array(dim * dim);
    
    //var matrix = new Float64Array([10, 0, 2, 0, -1, 0, -1, 3, 2, 0, 10, -1, 0, 0, -1, 8]);
    
    var vector = new Float64Array(dim);
    
    //var vector = new Float64Array([6, 25, -11, 15]);
    
    fillTestSystem(knownX, matrix, vector);        
    
    var t1 = Date.now();
    
    var slv = new Solver(dim);

    slv.useMatrix(matrix);
    slv.useVector(vector);
    
    //slv.iterate(10);
    slv.solve();
    
    var t2 = Date.now();

    $(document.body).append('<p>Done in ' + (t2 - t1) + ' ms</p>');

    console.log('Done in ' + (t2 - t1) + ' ms');

    /*
    var logMatrix = [];
        
    var offset = 0;
    
    for(var i = 0; i < dim; ++i)
    {
        var tmp = [];
        
        for(var j = 0; j < dim; ++j, ++offset)
        {
            tmp[j] = slv.matrix[offset];
        }
        
        logMatrix[i] = tmp;
    }
    
    //console.log(logMatrix);
    //console.log(slv.x);
    */
    
    var errSq = slv.calcErrorSquare();
    
    console.log(errSq);    
    
    $(document.body).append('<p>' + errSq + '</p>');
});

//------------------------------------------------------------------------------
