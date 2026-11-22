//! Jenny no mundo: le o modelo Bedrock (jenny/jenny.bin, feito do .geo.json) e
//! desenha os ossos direto na GPU, no ponto `native_world_drawn`.
use super::native_render::WorldDrawn;
use glam::{DVec3, Mat4, Vec3, Vec4};
use std::sync::Mutex;

const MODEL: &[u8] = include_bytes!("jenny/jenny.bin");
const TEXTURE: &[u8] = include_bytes!("jenny/jenny.rgba");
const TEX_SIZE: u32 = 1024;
/// Bedrock gira com sinal oposto a regra da mao direita.
const SIGN: f32 = -1.0;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const VERTEX_STRIDE: u64 = 32;

const SHADER: &str = r#"
struct U {
    vp: mat4x4<f32>,
    offset: vec4<f32>,
    sizes: vec4<f32>,
    dsize: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var scene_depth: texture_depth_2d;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) n: vec3<f32>,
};

@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) n: vec3<f32>) -> VOut {
    var o: VOut;
    o.pos = u.vp * vec4<f32>(p + u.offset.xyz, 1.0);
    o.uv = uv;
    o.n = n;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let c = textureSample(tex, samp, i.uv);
    let maxd = u.dsize.xy - vec2<f32>(1.0, 1.0);
    let dp = clamp(floor(i.pos.xy * u.dsize.xy / u.sizes.xy), vec2<f32>(0.0, 0.0), maxd);
    let s = textureLoad(scene_depth, vec2<i32>(dp), 0);
    if (i.pos.z < s - 0.00002) {
        discard;
    }
    if (c.a < 0.5) {
        discard;
    }
    let l = clamp(dot(normalize(i.n), normalize(vec3<f32>(0.3, 0.9, 0.4))), 0.0, 1.0);
    let shade = 0.62 + 0.38 * l;
    return vec4<f32>(c.rgb * shade, 1.0);
}
"#;

struct Bone {
    parent: i32,
    pivot: Vec3,
    rot: Vec3,
    first: usize,
    count: usize,
}

struct Model {
    bones: Vec<Bone>,
    /// pos(3) uv(2) normal(3), em blocos, na pose de repouso.
    verts: Vec<[f32; 8]>,
}

fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4;
    v
}

fn rd_f32(b: &[u8], p: &mut usize) -> f32 {
    f32::from_bits(rd_u32(b, p))
}

fn parse_model() -> Model {
    let mut p = 4;
    let nb = rd_u32(MODEL, &mut p) as usize;
    let mut bones = Vec::with_capacity(nb);
    for _ in 0..nb {
        let parent = rd_u32(MODEL, &mut p) as i32;
        let pivot = Vec3::new(rd_f32(MODEL, &mut p), rd_f32(MODEL, &mut p), rd_f32(MODEL, &mut p));
        let rot = Vec3::new(rd_f32(MODEL, &mut p), rd_f32(MODEL, &mut p), rd_f32(MODEL, &mut p));
        let first = rd_u32(MODEL, &mut p) as usize;
        let count = rd_u32(MODEL, &mut p) as usize;
        let name_len = MODEL[p] as usize;
        p += 1 + name_len;
        bones.push(Bone { parent, pivot, rot, first, count });
    }
    let total = rd_u32(MODEL, &mut p) as usize;
    let mut verts = Vec::with_capacity(total);
    for _ in 0..total {
        let mut v = [0.0f32; 8];
        for x in v.iter_mut() {
            *x = rd_f32(MODEL, &mut p);
        }
        verts.push(v);
    }
    Model { bones, verts }
}

fn bone_matrices(model: &Model) -> Vec<Mat4> {
    let mut world: Vec<Mat4> = Vec::with_capacity(model.bones.len());
    for b in &model.bones {
        let r = Mat4::from_rotation_x(SIGN * b.rot.x.to_radians())
            * Mat4::from_rotation_y(SIGN * b.rot.y.to_radians())
            * Mat4::from_rotation_z(SIGN * b.rot.z.to_radians());
        let local = Mat4::from_translation(b.pivot) * r * Mat4::from_translation(-b.pivot);
        let m = if b.parent >= 0 { world[b.parent as usize] * local } else { local };
        world.push(m);
    }
    world
}

/// Vertices ja na pose, virados `yaw` radianos em Y, em bytes prontos pro buffer.
fn build_vertices(model: &Model, yaw: f32) -> Vec<u8> {
    let mats = bone_matrices(model);
    let turn = Mat4::from_rotation_y(yaw);
    let mut out: Vec<u8> = Vec::with_capacity(model.verts.len() * VERTEX_STRIDE as usize);
    for (i, b) in model.bones.iter().enumerate() {
        let m = turn * mats[i];
        for v in &model.verts[b.first..b.first + b.count] {
            let pos = m.transform_point3(Vec3::new(v[0], v[1], v[2]));
            let n = m.transform_vector3(Vec3::new(v[5], v[6], v[7])).normalize_or_zero();
            for x in [pos.x, pos.y, pos.z, v[3], v[4], n.x, n.y, n.z] {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    out
}

fn build_indices(vertex_count: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(vertex_count / 4 * 6 * 4);
    for q in 0..(vertex_count / 4) as u32 {
        let b = q * 4;
        for i in [b, b + 1, b + 2, b, b + 2, b + 3] {
            out.extend_from_slice(&i.to_le_bytes());
        }
    }
    out
}

struct Gpu {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    tex_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    depth_view: Option<(wgpu::TextureView, [u32; 2])>,
}

struct State {
    model: Model,
    gpu: Option<Gpu>,
    anchor: Option<DVec3>,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn texture_format_for(target: wgpu::TextureFormat) -> wgpu::TextureFormat {
    // Se o alvo guarda cor linear (sRGB ou float), a textura tem que ser lida como sRGB.
    if target.is_srgb() || !matches!(target, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Bgra8Unorm) {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    }
}

fn make_gpu(world: &WorldDrawn<'_>, model: &Model) -> Gpu {
    let device = world.device;
    let queue = world.queue;
    let format = world.color_format;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Jenny"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Jenny bindings"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Jenny"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let attributes = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 20, shader_location: 2 },
    ];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Jenny"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: VERTEX_STRIDE,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &attributes,
            })],
        },
        primitive: Default::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Jenny uniform"),
        size: 96,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let vertices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Jenny vertices"),
        size: model.verts.len() as u64 * VERTEX_STRIDE,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let index_bytes = build_indices(model.verts.len());
    let indices = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Jenny indices"),
        size: index_bytes.len() as u64,
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&indices, 0, &index_bytes);

    let extent = wgpu::Extent3d { width: TEX_SIZE, height: TEX_SIZE, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Jenny texture"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: texture_format_for(format),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        TEXTURE,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(TEX_SIZE * 4),
            rows_per_image: Some(TEX_SIZE),
        },
        extent,
    );
    let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Jenny sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    Gpu {
        format,
        pipeline,
        layout,
        uniform,
        vertices,
        indices,
        index_count: (model.verts.len() / 4 * 6) as u32,
        tex_view,
        sampler,
        depth_view: None,
    }
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// Chamado todo quadro com o mundo ja desenhado.
pub(super) fn draw(world: &mut WorldDrawn<'_>) {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let state = guard.get_or_insert_with(|| State { model: parse_model(), gpu: None, anchor: None });

    // Posicao: 3 blocos na frente da camera, virada pra ela, perto do chao.
    let camera = world.camera;
    let too_far = match state.anchor {
        Some(a) => (a - camera).length() > 48.0,
        None => true,
    };
    let rebuild = too_far;
    let mut yaw = 0.0f32;
    if too_far {
        let inv = world.view_proj.inverse();
        let p = inv * Vec4::new(0.0, 0.0, 0.5, 1.0);
        let mut dir = Vec3::new(p.x / p.w, 0.0, p.z / p.w);
        if !dir.is_finite() || dir.length_squared() < 1e-8 {
            dir = Vec3::new(0.0, 0.0, -1.0);
        }
        let dir = dir.normalize();
        state.anchor = Some(camera + DVec3::new(dir.x as f64, 0.0, dir.z as f64) * 3.0 - DVec3::new(0.0, 1.62, 0.0));
        // O rosto do modelo olha pra -Z; queremos que olhe pra camera (-dir).
        yaw = (-dir.x).atan2(-dir.z);
    }
    let anchor = match state.anchor {
        Some(a) => a,
        None => return,
    };

    let needs_gpu = match &state.gpu {
        Some(g) => g.format != world.color_format,
        None => true,
    };
    if needs_gpu {
        state.gpu = Some(make_gpu(world, &state.model));
    }
    let State { model, gpu, .. } = state;
    let gpu = match gpu {
        Some(g) => g,
        None => return,
    };
    if rebuild || needs_gpu {
        let bytes = build_vertices(model, yaw);
        world.queue.write_buffer(&gpu.vertices, 0, &bytes);
    }

    // Profundidade propria, pra Jenny nao se atravessar.
    let size = world.size;
    let depth_ok = matches!(&gpu.depth_view, Some((_, s)) if *s == size);
    if !depth_ok {
        let texture = world.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Jenny depth"),
            size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        gpu.depth_view = Some((texture.create_view(&wgpu::TextureViewDescriptor::default()), size));
    }
    let depth_view = match &gpu.depth_view {
        Some((v, _)) => v,
        None => return,
    };

    let offset = (anchor - camera).as_vec3();
    let mut u: Vec<u8> = Vec::with_capacity(96);
    push_f32s(&mut u, &world.jittered.to_cols_array());
    push_f32s(&mut u, &[offset.x, offset.y, offset.z, 0.0]);
    push_f32s(&mut u, &[size[0] as f32, size[1] as f32, 0.0, 0.0]);
    push_f32s(&mut u, &[world.depth_size[0] as f32, world.depth_size[1] as f32, 0.0, 0.0]);
    world.queue.write_buffer(&gpu.uniform, 0, &u);

    let bind_group = world.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Jenny"),
        layout: &gpu.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&gpu.tex_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(world.depth) },
        ],
    });

    let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Jenny"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: world.color_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: depth_view,
            depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Discard }),
            stencil_ops: None,
        }),
        ..Default::default()
    });
    pass.set_pipeline(&gpu.pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.set_vertex_buffer(0, gpu.vertices.slice(..));
    pass.set_index_buffer(gpu.indices.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..gpu.index_count, 0, 0..1);
}
