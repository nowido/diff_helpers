//------------------------------------------------------------------------------

function FloatFromUbytes(conversionPlace)
{
    // conversion place b0, b1, b2, b3
    
    console.log(conversionPlace.b0.toString(16));
    console.log(conversionPlace.b1.toString(16));
    console.log(conversionPlace.b2.toString(16));
    console.log(conversionPlace.b3.toString(16));
    
    var sign = (conversionPlace.b3 > 127);
    
    console.log('sign: ' + sign);
    
    var expHigh = conversionPlace.b3 - (sign ? 128 : 0);
    var expLow = (conversionPlace.b2 > 127) ? 1 : 0;
    
    var expSh = expHigh * 2 + expLow;
    
    console.log('exp sh: ' + expSh);
    
    var exp = expSh - 127;
    
    var denorm = (expSh === 0);
    var inf = (expSh === 255);
    
    console.log('exp: ' + exp + (denorm ? ' (denorm)' : '') + (inf ? ' inf' : ''));
    
    var mantHigh = conversionPlace.b2 - ((conversionPlace.b2 > 127) ? 128 : 0);
    var mantMid = conversionPlace.b1;
    var mantLow = conversionPlace.b0;
    
    var unit23 = 8388608;
    
    var mant = (mantHigh * 65536 + mantMid * 256 + mantLow) / unit23;
    
    mant += denorm ? 0 : 1;
    exp += denorm ? 1 : 0;
    
    console.log(mant);
    
    return Math.pow(2, exp) * mant * (sign ? -1 : 1);
}

function UbytesFromFloat(fv, conversionPlace)
{
    // conversion place b0, b1, b2, b3
    
    var sign = (fv < 0);
    
    fv *= (sign ? -1 : 1);
    
    console.log('sign: ' + sign);
    
    var exp = (fv !== 0) ? Math.floor(Math.log(fv) / Math.log(2)) : -127;
    
    var expSh = exp + 127;
    
    console.log('exp sh: ' + expSh);
    
    var denorm = (expSh === 0);
    var inf = (expSh === 255);
    
    console.log('exp: ' + exp + (denorm ? ' (denorm)' : '') + (inf ? ' inf' : ''));
    
    var expHigh = Math.floor(expSh / 2);
    var expLow = expSh % 2;
    
    var mant = fv / Math.pow(2, exp);
    
    console.log(mant);
    
    mant *= denorm ? 0.5 : 1;
    mant -= denorm ? 0 : 1;
    
    var mantInt = Math.floor(mant * 8388608);
    
    conversionPlace.b3 = expHigh + (sign ? 128 : 0);
    
    var mantHigh = Math.floor(mantInt / 65536);
    
    conversionPlace.b2 = mantHigh + ((expLow > 0) ? 128 : 0);
    
    var mantMid = Math.floor((mantInt - mantHigh * 65536) / 256);
    
    conversionPlace.b1 = mantMid; 
    
    conversionPlace.b0 = Math.floor(mantInt - mantHigh * 65536 - mantMid * 256);
    
    console.log(conversionPlace.b0.toString(16));
    console.log(conversionPlace.b1.toString(16));
    console.log(conversionPlace.b2.toString(16));
    console.log(conversionPlace.b3.toString(16));
}

//------------------------------------------------------------------------------

function augmentMatrix(matrix, vector, augmentedMatrix)
{
    var N = vector.length;

    var leadIndex = -1;

        // write matrix cols as rows of augmented matrix
        // (transposed)
    
    for(var i = 0, augIndex = 0; i < N; ++i)
    {
        for(var j = 0; j < N; ++j, ++augIndex)
        {
            var m = matrix[j * N + i];

            augmentedMatrix[augIndex] = m; 

            if((i == 0) && (leadIndex < 0) && (Math.abs(m) > 0))
            {
                leadIndex = j;    
            }
        }
    } 
    
        // write vector B as last row of augmented matrix
    
    for(var i = 0; i < N; ++i, ++augIndex)
    {
        augmentedMatrix[augIndex] = vector[i]; 
    }
    
    return leadIndex;
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
    var tt = new Float32Array(1);
    
    tt[0] = 20;
    
    var ttb = new Uint8Array(tt.buffer);
    
    var conversionPlace = 
    {
        b0: ttb[0],
        b1: ttb[1],
        b2: ttb[2],
        b3: ttb[3]
    };
    
    console.log(FloatFromUbytes(conversionPlace));
    
    UbytesFromFloat(tt[0], conversionPlace);
    
        //
    
    var glass = new Glass();
    
        //
        
    const dim = 4;
    const aug_height = dim + 1;
    
    var matrixSrc = [   0,  3,  -1, 8,
                        10, -1, 2,  0,
                        -1,	11, -1, 3,
                        2,	-1, 10, -1  ];

    var matrix = new Float32Array(matrixSrc);

    var vector = new Float32Array([35, 14, 30, 26]);

    var transferMatrix = new Float32Array(dim * aug_height);
    
    var leadIndex = augmentMatrix(matrix, vector, transferMatrix);
    
    var transferMatrixConverted = new Uint8Array(transferMatrix.buffer);

        // create ping-pong resources
    
    var pp = [];
    
    var ppIndexInput = 0;
    var ppIndexOutput = 1;

    for(var i = 0; i < 2; ++i)
    {
        var r = glass.createResource(dim, aug_height, glass.gl.RGBA, glass.gl.UNSIGNED_BYTE);

        pp[i] = {resource: r, target: glass.createOutputTarget(r)};
    }
    
    glass.fillResource(pp[ppIndexInput].resource, transferMatrixConverted);
    
        //

    var kernelSource = 
            
        glass.buildDefines({'WIDTH' : dim, 'HEIGHT' : aug_height}) + 
        
        $('#commonKernelRoutines').text() + 
        
        $('#justTransfer').text();
    
    var kernel = glass.createKernel(kernelSource);
    
    if(!kernel)
    {
        console.log(glass.errorLog);
    }

    glass.setKernel(kernel);
    
    glass.linkInputSampler(0, 'matrix');

    glass.attachInputResource(0, pp[ppIndexInput].resource);
    glass.setOutputTarget(pp[ppIndexOutput].target);
    
    glass.dispatchKernel(0, 0, dim, aug_height);
    
    var dataRaw = new Uint8Array(dim * aug_height * 4);
    
    glass.readBack(dataRaw, 0, 0, dim, aug_height); 
    
    var dataConverted = new Float32Array(dataRaw.buffer);

    printAugmentedMatrix(dim, dataConverted);
});

//------------------------------------------------------------------------------