//------------------------------------------------------------------------------

function Solver(dimension)
{
    var solver = this;
    
    solver.dimension = dimension;
    
    solver.pp = [];
    solver.ppIndex = 0;
}

Solver.prototype.useMatrix = function(matrix)
{
    this.matrix = matrix;
}

Solver.prototype.useVector = function(vector)
{
    this.vector = vector;
}

Solver.prototype.useSeed = function(seed)
{
    var solver = this;
    
    solver.pp[solver.ppIndex] = seed;
    solver.pp[1 - solver.ppIndex] = new Float64Array(solver.dimension);
}

Solver.prototype.solve = function(count, eps)
{
    var solver = this;
    
    var dim = solver.dimension;
    
    var matrix = solver.matrix;
    var vector = solver.vector;
    
    for(var i = 0; i < count; ++i)
    {
        var xk = solver.pp[solver.ppIndex];
        var xk1 = solver.pp[1 - solver.ppIndex];
        
        for(var j = 0, diagIndex = 0; j < dim; ++j, diagIndex += (dim + 1))
        {
            var edp = solver.exclDotProduct(matrix, xk, j);    
            xk1[j] = (vector[j] - edp) / matrix[diagIndex];
        }
        
        solver.ppIndex = 1 - solver.ppIndex;
    }
}

Solver.prototype.exclDotProduct = function(matrix, xk, index)
{
    var dim = xk.length;

    var s = 0;
    
    for(var i = 0, matrixIndex = dim * index; i < dim; ++i, ++matrixIndex)
    {
        s += ((i !== index) ? matrix[matrixIndex] * xk[i] : 0);
    }
    
    return s;
}

Solver.prototype.calcErrorSquare = function()
{
    var solver = this;
    
    var dim = solver.dimension;
    
    var matrix = solver.matrix;
    var vector = solver.vector;
    
    var xk = solver.pp[solver.ppIndex];    
    
    var errSquare = 0;
    
    var index = 0;
    
    for(var i = 0; i < dim; ++i)
    {
        var s = 0;
        
        for(var j = 0; j < dim; ++j)
        {
            s += matrix[index] * xk[j];
                        
            ++index;
        }
        
        var err = (vector[i] - s);
        
        errSquare += (err * err);
    }
    
    return errSquare;
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
            var r = Math.floor(Math.random() * 10);
            
            var a = matrix[index] = ((i !== j) ? r : (r + 1) * 20);
            
            s += a * knownX[j];
        }
        
        vector[i] = s;
    }
}

//------------------------------------------------------------------------------

function main()
{
    const dim = 10;
    
    var knownX = new Float64Array(dim);

    knownX.forEach((v, index, arr) => {arr[index] = Math.floor(Math.random() * 10)});

    var matrix = new Float64Array(dim * dim);
    //var matrix = new Float32Array([10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -4, 0, 3, -1, 8]);
    var vector = new Float64Array(dim);
    //var vector = new Float32Array([6, 25, -11, 15]);
    
    fillTestSystem(knownX, matrix, vector);        
    
    var slv = new Solver(dim);
    
    slv.useSeed(knownX);
    
    slv.useMatrix(matrix);
    slv.useVector(vector);
    
    console.log(slv.calcErrorSquare());
    
    var seed = new Float64Array(dim);

    seed.forEach((v, index, arr) => {arr[index] = Math.floor(Math.random() * 10)});
    
    slv.useSeed(seed);
    
    console.log(slv.calcErrorSquare());
    
    slv.solve(17);
    
    console.log(slv.calcErrorSquare());
}

//------------------------------------------------------------------------------