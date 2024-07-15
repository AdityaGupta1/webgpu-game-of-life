const canvas = document.querySelector("canvas");

if (!navigator.gpu) {
    throw new Error("WebGPU not supported on this browser.");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
    throw new Error("No appropriate GPUAdapter found.");
}

const device = await adapter.requestDevice();

const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
    device: device,
    format: canvasFormat,
});

const vertices = new Float32Array([
    -0.8, -0.8, // Triangle 1 (Blue)
    0.8, -0.8,
    0.8,  0.8,

    -0.8, -0.8, // Triangle 2 (Red)
    0.8,  0.8,
    -0.8,  0.8,
]);

const vertexBuffer = device.createBuffer({
    label: "cell vertices",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
});

device.queue.writeBuffer(vertexBuffer, /*bufferOffset=*/0, vertices);

const vertexBufferLayout = {
    arrayStride: 8,
    attributes: [{
        format: "float32x2",
        offset: 0,
        shaderLocation: 0, // shader layout pos
    }],
};

const cellVertShader = device.createShaderModule({
    label: "cell vert shader",
    code: `
@vertex
fn vertMain(@location(0) pos: vec2f) -> @builtin(position) vec4f {
    return vec4f(pos, 0, 1);
}
    `
});

const cellFragShader = device.createShaderModule({
    label: "cell frag shader",
    code: `
@fragment
fn fragMain() -> @location(0) vec4f {
    return vec4(1, 0, 1, 1);
}
    `
});

const cellPipeline = device.createRenderPipeline({
    label: "cell pipeline",
    layout: "auto",
    vertex: {
        module: cellVertShader,
        entryPoint: "vertMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: cellFragShader,
        entryPoint: "fragMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

const encoder = device.createCommandEncoder();

const pass = encoder.beginRenderPass({
    colorAttachments: [{
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: [0, 0, 0, 1],
        storeOp: "store",
    }]
});

pass.setPipeline(cellPipeline);
pass.setVertexBuffer(0, vertexBuffer);
pass.draw(vertices.length / 2);

pass.end();

const commandBuffer = encoder.finish();

device.queue.submit([commandBuffer]);
