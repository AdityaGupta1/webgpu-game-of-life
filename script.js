const GRID_SIZE_X = 64;
const GRID_SIZE_Y = 64;
const CELL_PADDING = 0.1;

const canvas = document.querySelector("canvas");

if (!navigator.gpu)
{
    throw new Error("WebGPU not supported on this browser.");
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter)
{
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
    CELL_PADDING, CELL_PADDING,
    1 - CELL_PADDING, CELL_PADDING,
    1 - CELL_PADDING, 1 - CELL_PADDING,

    CELL_PADDING, CELL_PADDING,
    1 - CELL_PADDING, 1 - CELL_PADDING,
    CELL_PADDING, 1 - CELL_PADDING,
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

const gridSizeUniformArray = new Uint32Array([GRID_SIZE_X, GRID_SIZE_Y]);
const gridSizeUniformBuffer = device.createBuffer({
  label: "grid size uniform",
  size: gridSizeUniformArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(gridSizeUniformBuffer, 0, gridSizeUniformArray);

const cellPaddingUniformArray = new Float32Array([CELL_PADDING]);
const cellPaddingUniformBuffer = device.createBuffer({
  label: "cell padding uniform",
  size: cellPaddingUniformArray.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(cellPaddingUniformBuffer, 0, cellPaddingUniformArray);

function createCellStatesStorageBuffer(idx)
{
    return device.createBuffer({
        label: "cell states " + idx,
        size: cellStatesStorageArray.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
}

const cellStatesStorageArray = new Uint32Array(GRID_SIZE_X * GRID_SIZE_Y);
const cellStatesStorageBuffers = [
    createCellStatesStorageBuffer(0),
    createCellStatesStorageBuffer(1)
];

for (let i = 0; i < cellStatesStorageArray.length; i += 3)
{
    cellStatesStorageArray[i] = 1;
}
device.queue.writeBuffer(cellStatesStorageBuffers[0], 0, cellStatesStorageArray);

for (let i = 0; i < cellStatesStorageArray.length; ++i)
{
    cellStatesStorageArray[i] = i % 2;
}
device.queue.writeBuffer(cellStatesStorageBuffers[1], 0, cellStatesStorageArray);

const cellShaderModule = device.createShaderModule({
    label: "cell shader",
    code: `
@group(0) @binding(0) var<uniform> gridSize: vec2u;
@group(0) @binding(1) var<uniform> cellPadding: f32;

@group(1) @binding(0) var<storage> cellStates: array<u32>;

struct VertexInput {
    @location(0) pos: vec2f,
    @builtin(instance_index) instanceIdx: u32,
};

struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) cellGridPos: vec2u,
};

@vertex
fn vertMain(in: VertexInput) -> VertexOutput {
    let cellGridPos = vec2u(in.instanceIdx % gridSize.x, in.instanceIdx / gridSize.x);
    let inverseGridSize = 1 / vec2f(gridSize);

    var vertPos = ((vec2f(cellGridPos) + in.pos) * inverseGridSize) * 2 - 1;
    vertPos *= 1 - 2 * (cellPadding * inverseGridSize);

    let cellState = f32(cellStates[in.instanceIdx]);

    var out: VertexOutput;
    out.pos = vec4(vertPos * cellState, 0, 1);
    out.cellGridPos = cellGridPos;
    return out;
}

struct FragInput {
    @location(0) @interpolate(flat) cellGridPos: vec2u,
};

struct FragOutput {
    @location(0) color: vec4f,
}

@fragment
fn fragMain(in: FragInput) -> FragOutput {
    let normalizedCellPos = vec2f(in.cellGridPos) / vec2f(gridSize);
    var color = vec4f(normalizedCellPos, 1 - normalizedCellPos.x, 1);

    var out: FragOutput;
    out.color = color;
    return out;
}
    `
});

const cellPipeline = device.createRenderPipeline({
    label: "cell pipeline",
    layout: "auto",
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertMain",
        buffers: [vertexBufferLayout]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragMain",
        targets: [{
            format: canvasFormat
        }]
    }
});

const gridUniformsBindGroup = device.createBindGroup({
    label: "grid uniforms bind group",
    layout: cellPipeline.getBindGroupLayout(0),
    entries: [
        {
            binding: 0,
            resource: { buffer: gridSizeUniformBuffer }
        },
        {
            binding: 1,
            resource: { buffer: cellPaddingUniformBuffer }
        }
    ],
});

function createCellStatesBindGroup(idx)
{
    return device.createBindGroup({
        label: "cell states bind group " + idx,
        layout: cellPipeline.getBindGroupLayout(1),
        entries: [
            {
                binding: 0,
                resource: { buffer: cellStatesStorageBuffers[idx] }
            }
        ]
    })
}

const cellStatesBindGroups = [
    createCellStatesBindGroup(0),
    createCellStatesBindGroup(1)
];

const UPDATE_INTERVAL = 250;
let step = 0;

function draw()
{
    ++step;

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

    pass.setBindGroup(0, gridUniformsBindGroup);
    pass.setBindGroup(1, cellStatesBindGroups[step % 2]);

    pass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y);

    pass.end();

    const commandBuffer = encoder.finish();

    device.queue.submit([commandBuffer]);
}

setInterval(draw, UPDATE_INTERVAL);
