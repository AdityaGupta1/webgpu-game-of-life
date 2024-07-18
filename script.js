const GRID_SIZE_X = 512;
const GRID_SIZE_Y = 512;
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
    attributes: [
        {
            format: "float32x2",
            offset: 0,
            shaderLocation: 0, // shader layout pos
        }
    ],
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

for (let i = 0; i < cellStatesStorageArray.length; ++i)
{
    cellStatesStorageArray[i] = Math.random() > 0.6 ? 1 : 0;
}
device.queue.writeBuffer(cellStatesStorageBuffers[0], 0, cellStatesStorageArray);

const cellShaderModule = device.createShaderModule({
    label: "cell vert/frag shader",
    code: `
@group(0) @binding(0) var<uniform> gridSize: vec2u;
@group(0) @binding(1) var<uniform> cellPadding: f32;

@group(1) @binding(0) var<storage> cellStates: array<u32>;

struct VertexInput
{
    @location(0) pos: vec2f,
    @builtin(instance_index) instanceIdx: u32,
};

struct VertexOutput
{
    @builtin(position) pos: vec4f,
    @location(0) @interpolate(flat) cellGridPos: vec2u,
};

@vertex
fn vertMain(in: VertexInput) -> VertexOutput
{
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

struct FragInput
{
    @location(0) @interpolate(flat) cellGridPos: vec2u,
};

struct FragOutput
{
    @location(0) color: vec4f,
}

@fragment
fn fragMain(in: FragInput) -> FragOutput 
{
    let normalizedCellPos = vec2f(in.cellGridPos) / vec2f(gridSize);
    var color = vec4f(normalizedCellPos, 1 - normalizedCellPos.x, 1);

    var out: FragOutput;
    out.color = color;
    return out;
}
    `
});

const WORKGROUP_SIZE_X = 8;
const WORKGROUP_SIZE_Y = 8;

const simulationShaderModule = device.createShaderModule({
  label: "simulation compute shader",
  code: `
@group(0) @binding(0) var<uniform> gridSize: vec2u;

@group(1) @binding(0) var<storage> inCellStates: array<u32>;
@group(1) @binding(1) var<storage, read_write> outCellStates: array<u32>;

fn cellGridPosToIdx(cellGridPos: vec2u) -> u32
{
    return (cellGridPos.y % gridSize.y) * gridSize.x + (cellGridPos.x % gridSize.x);
}

fn isCellActive(cellGridPos: vec2u, dx: i32, dy: i32) -> u32 {
    return inCellStates[cellGridPosToIdx(cellGridPos + vec2u(u32(dx), u32(dy)) + gridSize)]; // add gridSize to compensate for underflow
}

@compute
@workgroup_size(${WORKGROUP_SIZE_X}, ${WORKGROUP_SIZE_Y})
fn computeMain(@builtin(global_invocation_id) cellGridPos: vec3u)
{
    if (cellGridPos.x >= gridSize.x || cellGridPos.y >= gridSize.y)
    {
        return;
    }

    let numActiveNeighbors = isCellActive(cellGridPos.xy, -1, -1)
                           + isCellActive(cellGridPos.xy,  0, -1)
                           + isCellActive(cellGridPos.xy,  1, -1)
                           + isCellActive(cellGridPos.xy, -1,  0)
                           + isCellActive(cellGridPos.xy,  1,  0)
                           + isCellActive(cellGridPos.xy, -1,  1)
                           + isCellActive(cellGridPos.xy,  0,  1)
                           + isCellActive(cellGridPos.xy,  1,  1);

    let thisGridIdx = cellGridPosToIdx(cellGridPos.xy);

    switch numActiveNeighbors
    {
        case 2:
        {
            outCellStates[thisGridIdx] = inCellStates[thisGridIdx];
        }
        case 3:
        {
            outCellStates[thisGridIdx] = 1;
        }
        default:
        {
            outCellStates[thisGridIdx] = 0;
        }
    }
}
    `
});

const gridUniformsBindGroupLayout = device.createBindGroupLayout({
    label: "grid uniforms bind group layout",
    entries: [
        { // gridSize
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
            buffer: { type: "uniform" }
        }, 
        { // cellPadding
            binding: 1,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "uniform" }
        }
    ]
});

const cellStatesBindGroupLayout = device.createBindGroupLayout({
    label: "cell states bind group layout",
    entries: [
        { // inCellState
            binding: 0,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage"}
        }, 
        { // outCellState
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage"}
        }
    ]
});

const gridUniformsBindGroup = device.createBindGroup({
    label: "grid uniforms bind group",
    layout: gridUniformsBindGroupLayout,
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
        layout: cellStatesBindGroupLayout,
        entries: [
            {
                binding: 0,
                resource: { buffer: cellStatesStorageBuffers[idx] }
            },
            {
                binding: 1,
                resource: { buffer: cellStatesStorageBuffers[1 - idx] }
            }
        ]
    })
}

const cellStatesBindGroups = [
    createCellStatesBindGroup(0),
    createCellStatesBindGroup(1)
];

const pipelineLayout = device.createPipelineLayout({
    label: "pipeline layout",
    bindGroupLayouts: [ gridUniformsBindGroupLayout, cellStatesBindGroupLayout ]
});

const cellRenderPipeline = device.createRenderPipeline({
    label: "cell render pipeline",
    layout: pipelineLayout,
    vertex: {
        module: cellShaderModule,
        entryPoint: "vertMain",
        buffers: [ vertexBufferLayout ]
    },
    fragment: {
        module: cellShaderModule,
        entryPoint: "fragMain",
        targets: [
            {
                format: canvasFormat
            }
        ]
    }
});

const simulationComputePipeline = device.createComputePipeline({
    label: "simulation compute pipeline",
    layout: pipelineLayout,
    compute: {
        module: simulationShaderModule,
        entryPoint: "computeMain",
    }
});

function calculateNumWorkgroups(n, workgroupSize)
{
    return (n + workgroupSize - 1) / workgroupSize;
}

const UPDATE_INTERVAL = 10;
let step = 0;

function draw()
{
    const encoder = device.createCommandEncoder();

    // run render pass first so the initial state gets rendered as well (rather than starting rendering after the first step of simulation)
    const renderPass = encoder.beginRenderPass({
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                clearValue: [0, 0, 0, 1],
                storeOp: "store",
            }
        ]
    });

    renderPass.setPipeline(cellRenderPipeline);
    renderPass.setVertexBuffer(0, vertexBuffer);

    renderPass.setBindGroup(0, gridUniformsBindGroup);
    renderPass.setBindGroup(1, cellStatesBindGroups[step % 2]);

    renderPass.draw(vertices.length / 2, GRID_SIZE_X * GRID_SIZE_Y);

    renderPass.end();

    const computePass = encoder.beginComputePass();

    computePass.setPipeline(simulationComputePipeline);
    computePass.setBindGroup(0, gridUniformsBindGroup);
    computePass.setBindGroup(1, cellStatesBindGroups[step % 2]);

    computePass.dispatchWorkgroups(
        calculateNumWorkgroups(GRID_SIZE_X, WORKGROUP_SIZE_X), 
        calculateNumWorkgroups(GRID_SIZE_Y, WORKGROUP_SIZE_Y)
    );

    computePass.end();

    const commandBuffer = encoder.finish();

    device.queue.submit([commandBuffer]);
    
    ++step;
}

setInterval(draw, UPDATE_INTERVAL);
