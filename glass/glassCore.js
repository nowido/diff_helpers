//------------------------------------------------------------------------------

function Glass(canvas)
{
    var glass = this;
    
    if(canvas !== undefined)
    {
        glass.userCanvasAttached = true;
    }
    else
    {
        canvas = document.createElement('canvas');
        
        canvas.width = 1;
        canvas.height = 1;
    }
    
    glass.gl = canvas.getContext('webgl');
    
    var gl = glass.gl;
    
        // frequently used extensions stuff
    
    if(!gl.getExtension('OES_texture_float'))
    {
        console.log('OES_texture_float not supported');
    }
    
    if(!gl.getExtension('OES_texture_float_linear'))
    {
        console.log('OES_texture_float_linear not supported');
    }

        // define 'all cover' buffer (Z-pattern of 2 triangles)
    
    glass.vbViewCover = gl.createBuffer();
    
    gl.bindBuffer(gl.ARRAY_BUFFER, glass.vbViewCover);
    
    gl.bufferData
    (
        gl.ARRAY_BUFFER, 
        new Float32Array([-1, +1, +1, +1, -1, -1, +1, -1, -1, -1, +1, +1]), 
        gl.STATIC_DRAW
    );
    
        // define 'all cover' vertex shader
    
    glass.vsViewCover = glass.createShader
    (
        gl.VERTEX_SHADER, 
        'attribute vec2 v;void main(){gl_Position = vec4(v.x, v.y, 0.0, 1.0);}'
    );
}

//------------------------------------------------------------------------------

Object.defineProperty(Glass.prototype, 'canvas', 
{
    get: function()
    {
        return (this.gl && this.userCanvasAttached) ? this.gl.canvas : null;
    }
});

//------------------------------------------------------------------------------

Glass.prototype.resize = function(width, height)
{
    var glass = this;
    
    var canvas = glass.canvas;
    
    if(canvas)
    {
        canvas.width = width;
        canvas.height = height;
        
        var gl = glass.gl;
        
        gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    }
}

//------------------------------------------------------------------------------

Glass.prototype.createShader = function(shaderType, shaderSource)
{
    var glass = this;
    
    var gl = glass.gl;
    
    var shader = gl.createShader(shaderType);
    
    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    
    if(!gl.getShaderParameter(shader, gl.COMPILE_STATUS))
    {
        glass.errorLog = gl.getShaderInfoLog(shader);
        
        gl.deleteShader(shader);
        
        shader = null;
    }
    
    return shader;
}

//------------------------------------------------------------------------------

Glass.prototype.createProgram = function(vertexShader, fragmentShader)
{
    var glass = this;
    
    var gl = glass.gl;
    
    var program = gl.createProgram();
    
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    
    gl.linkProgram(program);
    
    if(!gl.getProgramParameter(program, gl.LINK_STATUS)) 
    {
        glass.errorLog = gl.getProgramInfoLog(program);    
        
        gl.deleteProgram(program);
        
        program = null;
    }
    
    return program;
}

//------------------------------------------------------------------------------

Glass.prototype.buildDefines = function(macro)
{
    return Object.keys(macro).map(key => ('#define ' + key + ' ' + macro[key])).join('\n');
}

//------------------------------------------------------------------------------

Glass.prototype.createKernel = function(kernelSource)
{
    var glass = this;
    
    var fs = glass.createShader(glass.gl.FRAGMENT_SHADER, kernelSource);
    
    if(fs)
    {
        return glass.createProgram(glass.vsViewCover, fs); 
    }
    
    return null;    
}

//------------------------------------------------------------------------------

Glass.prototype.setKernel = function(kernel)
{
    var glass = this;

    glass.gl.useProgram(kernel);
    
    glass.kernel = kernel;
}

//------------------------------------------------------------------------------

Glass.prototype.dispatchKernel = function(left, top, width, height)
{
    var glass = this;
    
    var gl = glass.gl;

    gl.bindBuffer(gl.ARRAY_BUFFER, glass.vbViewCover);
    
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 2 * 4, 0);
    
    var resource = glass.outputTarget.resource;
    
    gl.viewport
    (
        left ? left : 0, 
        top ? top : 0, 
        width ? width : resource.width, 
        height ? height : resource.height
    );
    
    gl.drawArrays(gl.TRIANGLES, 0, 2 * 3);
}

//------------------------------------------------------------------------------

Glass.prototype.createResource = function(width, height, format, type, dataBuffer)
{
    var glass = this;
    
    var gl = glass.gl;
    
    var texture = gl.createTexture();
    
    gl.bindTexture(gl.TEXTURE_2D, texture);
    
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    
    //gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    //gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    
    gl.texImage2D
    (
        gl.TEXTURE_2D, 
        0, 
        format, 
        width, 
        height, 
        0, 
        format, 
        type, 
        dataBuffer ? dataBuffer : null
    );

    return {texture: texture, width: width, height: height, format: format, type: type};
}

//------------------------------------------------------------------------------

Glass.prototype.fillResource = function(resource, dataBuffer)
{
    var glass = this;
    
    var gl = glass.gl;
   
    gl.bindTexture(gl.TEXTURE_2D, resource.texture);
    
    gl.texImage2D
    (
        gl.TEXTURE_2D, 
        0, 
        resource.format, 
        resource.width, 
        resource.height, 
        0, 
        resource.format, 
        resource.type, 
        dataBuffer
    );
}

//------------------------------------------------------------------------------

Glass.prototype.createOutputTarget = function(resource)
{
    return {frameBuffer: this.gl.createFramebuffer(), resource: resource};
}

//------------------------------------------------------------------------------

Glass.prototype.setOutputTarget = function(outputTarget)
{
    var glass = this;
    
    var gl = glass.gl;
    
    gl.bindFramebuffer(gl.FRAMEBUFFER, outputTarget.frameBuffer);
    
    var resource = outputTarget.resource;
    
    gl.framebufferTexture2D
    (
        gl.FRAMEBUFFER, 
        gl.COLOR_ATTACHMENT0, 
        gl.TEXTURE_2D, 
        resource.texture, 
        0
    );
    
    glass.outputTarget = outputTarget;
}

//------------------------------------------------------------------------------

Glass.prototype.readBack = function(dataBuffer, left, top, width, height)
{
    var glass = this;
    
    var gl = glass.gl;
    
    var resource = glass.outputTarget.resource;
    
    gl.readPixels
    (
        left ? left : 0, 
        top ? top : 0, 
        width ? width : resource.width, 
        height ? height : resource.height, 
        resource.format, 
        resource.type, 
        dataBuffer
    );
}

//------------------------------------------------------------------------------

Glass.prototype.attachInputResource = function(index, resource)
{
    var glass = this;
    
    var gl = glass.gl;

    gl.activeTexture(gl.TEXTURE0 + index);
    
    gl.bindTexture(gl.TEXTURE_2D, resource.texture);
}

//------------------------------------------------------------------------------

Glass.prototype.linkInputSampler = function(index, samplerName)
{
    var glass = this;
    
    var gl = glass.gl;
    
    var samplerLocation = gl.getUniformLocation(glass.kernel, samplerName);
    
    gl.uniform1i(samplerLocation, index);
}

//------------------------------------------------------------------------------
