//! Personagens no mundo: le os modelos Bedrock (chars/*.bin, feitos dos .geo.json) e
//! desenha os ossos direto na GPU, no ponto `native_world_drawn`.
use super::native_render::WorldDrawn;
use glam::{DVec3, Mat4, Vec3, Vec4};
use std::sync::Mutex;

struct CharDef {
    model: &'static [u8],
    texture: &'static [u8],
    anim_idle: &'static [u8],
    anim_walk: &'static [u8],
}

/// Ordem na fila (esquerda -> direita): Jenny, Ellie, Bia, Allie, Slime.
const CHARS: [CharDef; 5] = [
    CharDef { model: include_bytes!("chars/jenny.bin"), texture: include_bytes!("chars/jenny.rgba"), anim_idle: include_bytes!("chars/jenny_idle.anim"), anim_walk: include_bytes!("chars/jenny_walk.anim") },
    CharDef { model: include_bytes!("chars/ellie.bin"), texture: include_bytes!("chars/ellie.rgba"), anim_idle: include_bytes!("chars/ellie_idle.anim"), anim_walk: include_bytes!("chars/ellie_walk.anim") },
    CharDef { model: include_bytes!("chars/bia.bin"), texture: include_bytes!("chars/bia.rgba"), anim_idle: include_bytes!("chars/bia_idle.anim"), anim_walk: include_bytes!("chars/bia_walk.anim") },
    CharDef { model: include_bytes!("chars/allie.bin"), texture: include_bytes!("chars/allie.rgba"), anim_idle: include_bytes!("chars/allie_idle.anim"), anim_walk: include_bytes!("chars/allie_walk.anim") },
    CharDef { model: include_bytes!("chars/slime.bin"), texture: include_bytes!("chars/slime.rgba"), anim_idle: include_bytes!("chars/slime_idle.anim"), anim_walk: include_bytes!("chars/slime_walk.anim") },
];
const TEX_SIZE: u32 = 1024;
/// Distancia da camera e espaco entre personagens (em blocos).
const SPAWN_DIST: f64 = 5.0;
const SPACING: f32 = 1.8;
/// Passeio: raio (blocos) em volta do lugar de cada uma e velocidade (blocos/s).
const WANDER_RADIUS: f32 = 1.4;
const WALK_SPEED: f32 = 1.0;
/// Bedrock gira com sinal oposto a regra da mao direita.
const SIGN: f32 = -1.0;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const VERTEX_STRIDE: u64 = 44;

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
    @location(2) tint: vec3<f32>,
};

@vertex
fn vs(@location(0) p: vec3<f32>, @location(1) uv: vec2<f32>, @location(2) n: vec3<f32>, @location(3) t: vec3<f32>) -> VOut {
    var o: VOut;
    o.pos = u.vp * vec4<f32>(p + u.offset.xyz, 1.0);
    o.uv = uv;
    o.n = n;
    o.tint = t;
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
    var rgb = c.rgb;
    // Camadas "armor" sao pintadas de ciano e tingidas na hora (como o mod faz).
    if (c.r < 0.59 && c.g > 0.55 && c.b > 0.51 && (c.g - c.r) > 0.16) {
        let lum = c.r * 0.3 + c.g * 0.59 + c.b * 0.11;
        rgb = vec3<f32>(lum, lum, lum) * i.tint * 1.6;
    }
    let l = clamp(dot(normalize(i.n), normalize(vec3<f32>(0.3, 0.9, 0.4))), 0.0, 1.0);
    let shade = 0.62 + 0.38 * l;
    return vec4<f32>(rgb * shade, 1.0);
}
"#;

struct Bone {
    parent: i32,
    pivot: Vec3,
    rot: Vec3,
    first: usize,
    count: usize,
}

struct Anim {
    frames: usize,
    fps: f32,
    /// (indice do osso, frames de [rot xyz, pos xyz]); rot em graus, pos em blocos.
    tracks: Vec<(usize, Vec<[f32; 6]>)>,
}

struct Model {
    idle: Option<Anim>,
    walk: Option<Anim>,
    bones: Vec<Bone>,
    /// pos(3) uv(2) normal(3) tint(3), em blocos, na pose de repouso.
    verts: Vec<[f32; 11]>,
}

fn rd_u32(b: &[u8], p: &mut usize) -> u32 {
    let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]);
    *p += 4;
    v
}

fn rd_f32(b: &[u8], p: &mut usize) -> f32 {
    f32::from_bits(rd_u32(b, p))
}

fn parse_model(data: &[u8]) -> Model {
    let mut p = 4;
    let nb = rd_u32(data, &mut p) as usize;
    let mut bones = Vec::with_capacity(nb);
    for _ in 0..nb {
        let parent = rd_u32(data, &mut p) as i32;
        let pivot = Vec3::new(rd_f32(data, &mut p), rd_f32(data, &mut p), rd_f32(data, &mut p));
        let rot = Vec3::new(rd_f32(data, &mut p), rd_f32(data, &mut p), rd_f32(data, &mut p));
        let first = rd_u32(data, &mut p) as usize;
        let count = rd_u32(data, &mut p) as usize;
        let name_len = data[p] as usize;
        p += 1 + name_len;
        bones.push(Bone { parent, pivot, rot, first, count });
    }
    let total = rd_u32(data, &mut p) as usize;
    let mut verts = Vec::with_capacity(total);
    for _ in 0..total {
        let mut v = [0.0f32; 11];
        for x in v.iter_mut() {
            *x = rd_f32(data, &mut p);
        }
        verts.push(v);
    }
    Model { idle: None, walk: None, bones, verts }
}

fn parse_anim(data: &[u8]) -> Option<Anim> {
    if data.len() < 16 || &data[0..4] != b"JAN1" {
        return None;
    }
    let mut p = 4;
    let frames = rd_u32(data, &mut p) as usize;
    let fps = rd_f32(data, &mut p);
    let nt = rd_u32(data, &mut p) as usize;
    if frames == 0 || fps <= 0.0 {
        return None;
    }
    let mut tracks = Vec::with_capacity(nt);
    for _ in 0..nt {
        let bone = rd_u32(data, &mut p) as usize;
        let mut fr: Vec<[f32; 6]> = Vec::with_capacity(frames);
        for _ in 0..frames {
            let mut a = [0.0f32; 6];
            for x in a.iter_mut() {
                *x = rd_f32(data, &mut p);
            }
            fr.push(a);
        }
        tracks.push((bone, fr));
    }
    Some(Anim { frames, fps, tracks })
}

struct Pose {
    t_idle: f32,
    t_walk: f32,
    /// 0 = parada (idle), 1 = andando (walk).
    w: f32,
}

fn add_pose(a: &Anim, t: f32, weight: f32, drot: &mut [Vec3], dpos: &mut [Vec3]) {
    if weight <= 0.0 {
        return;
    }
    let n = drot.len();
    let f = (t * a.fps).rem_euclid(a.frames as f32);
    let i0 = (f.floor() as usize).min(a.frames - 1);
    let i1 = (i0 + 1) % a.frames;
    let k = f - i0 as f32;
    for (bone, fr) in &a.tracks {
        if *bone >= n {
            continue;
        }
        let p = &fr[i0];
        let q = &fr[i1];
        let l = |j: usize| p[j] + (q[j] - p[j]) * k;
        drot[*bone] += Vec3::new(l(0), l(1), l(2)) * weight;
        dpos[*bone] += Vec3::new(l(3), l(4), l(5)) * weight;
    }
}

fn bone_matrices(model: &Model, pose: &Pose) -> Vec<Mat4> {
    let n = model.bones.len();
    let mut drot = vec![Vec3::ZERO; n];
    let mut dpos = vec![Vec3::ZERO; n];
    let w = if model.walk.is_some() { pose.w.clamp(0.0, 1.0) } else { 0.0 };
    if let Some(a) = &model.idle {
        add_pose(a, pose.t_idle, 1.0 - w, &mut drot, &mut dpos);
    }
    if let Some(a) = &model.walk {
        add_pose(a, pose.t_walk, w, &mut drot, &mut dpos);
    }
    let mut world: Vec<Mat4> = Vec::with_capacity(n);
    for (i, b) in model.bones.iter().enumerate() {
        // Bedrock: ordem ZYX (X aplicado primeiro). A animacao soma na rotacao de repouso.
        let rot = b.rot + drot[i];
        let r = Mat4::from_rotation_z(SIGN * rot.z.to_radians())
            * Mat4::from_rotation_y(SIGN * rot.y.to_radians())
            * Mat4::from_rotation_x(SIGN * rot.x.to_radians());
        let local = Mat4::from_translation(dpos[i]) * Mat4::from_translation(b.pivot) * r * Mat4::from_translation(-b.pivot);
        let m = if b.parent >= 0 { world[b.parent as usize] * local } else { local };
        world.push(m);
    }
    world
}

/// Passeio de cada personagem (so no cliente, no plano, sem colisao).
struct Wander {
    x: f32,
    z: f32,
    tx: f32,
    tz: f32,
    yaw: f32,
    walking: bool,
    wait: f32,
    w: f32,
    wt: f32,
    rng: u32,
}

impl Wander {
    fn new(yaw: f32, i: usize) -> Wander {
        Wander {
            x: 0.0,
            z: 0.0,
            tx: 0.0,
            tz: 0.0,
            yaw,
            walking: false,
            wait: 1.0 + i as f32 * 0.7,
            w: 0.0,
            wt: 0.0,
            rng: 0x9E37_79B9u32.wrapping_mul(i as u32 + 1) | 1,
        }
    }
}

fn rnd(w: &mut Wander) -> f32 {
    w.rng ^= w.rng << 13;
    w.rng ^= w.rng >> 17;
    w.rng ^= w.rng << 5;
    (w.rng & 0x00FF_FFFF) as f32 / 16_777_216.0
}

fn turn_toward(cur: f32, want: f32, max: f32) -> f32 {
    let tau = std::f32::consts::TAU;
    let mut d = (want - cur) % tau;
    if d > std::f32::consts::PI {
        d -= tau;
    }
    if d < -std::f32::consts::PI {
        d += tau;
    }
    cur + d.clamp(-max, max)
}

fn step_wander(w: &mut Wander, dt: f32, home_yaw: f32) {
    if w.walking {
        let dx = w.tx - w.x;
        let dz = w.tz - w.z;
        let d = (dx * dx + dz * dz).sqrt();
        if d < 0.08 {
            w.walking = false;
            w.wait = 2.0 + rnd(w) * 5.0;
        } else {
            let step = (WALK_SPEED * dt).min(d);
            w.x += dx / d * step;
            w.z += dz / d * step;
            w.yaw = turn_toward(w.yaw, dx.atan2(dz), 6.0 * dt);
        }
    } else {
        w.yaw = turn_toward(w.yaw, home_yaw, 3.0 * dt);
        w.wait -= dt;
        if w.wait <= 0.0 {
            let a = rnd(w) * std::f32::consts::TAU;
            let r = (0.5 + rnd(w) * 0.5) * WANDER_RADIUS;
            w.tx = a.cos() * r;
            w.tz = a.sin() * r;
            w.walking = true;
        }
    }
    let goal = if w.walking { 1.0 } else { 0.0 };
    w.w += (goal - w.w) * (dt / 0.25).min(1.0);
    if w.w > 0.01 {
        w.wt += dt;
    }
}

/// Vertices ja na pose, virados `yaw` radianos em Y, em bytes prontos pro buffer.
fn build_vertices(model: &Model, yaw: f32, pose: &Pose) -> Vec<u8> {
    let mats = bone_matrices(model, pose);
    let turn = Mat4::from_rotation_y(yaw);
    let mut out: Vec<u8> = Vec::with_capacity(model.verts.len() * VERTEX_STRIDE as usize);
    for (i, b) in model.bones.iter().enumerate() {
        let m = turn * mats[i];
        for v in &model.verts[b.first..b.first + b.count] {
            let pos = m.transform_point3(Vec3::new(v[0], v[1], v[2]));
            let n = m.transform_vector3(Vec3::new(v[5], v[6], v[7])).normalize_or_zero();
            for x in [pos.x, pos.y, pos.z, v[3], v[4], n.x, n.y, n.z, v[8], v[9], v[10]] {
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

struct Shared {
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    depth_view: Option<(wgpu::TextureView, [u32; 2])>,
}

struct CharGpu {
    uniform: wgpu::Buffer,
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    tex_view: wgpu::TextureView,
}

struct State {
    models: Vec<Model>,
    shared: Option<Shared>,
    chars: Vec<CharGpu>,
    anchor: Option<DVec3>,
    right: Vec3,
    yaw: f32,
    start: std::time::Instant,
    last_t: f32,
    wander: Vec<Wander>,
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

fn make_gpu(world: &WorldDrawn<'_>, models: &[Model]) -> (Shared, Vec<CharGpu>) {
    let device = world.device;
    let queue = world.queue;
    let format = world.color_format;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Personagens"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Personagens bindings"),
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
        label: Some("Personagens"),
        bind_group_layouts: &[Some(&layout)],
        ..Default::default()
    });
    let attributes = [
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 32, shader_location: 3 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 12, shader_location: 1 },
        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 20, shader_location: 2 },
    ];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Personagens"),
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
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Personagens sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let mut chars: Vec<CharGpu> = Vec::with_capacity(models.len());
    for (i, model) in models.iter().enumerate() {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Personagem uniform"),
            size: 112,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vertices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Personagem vertices"),
            size: model.verts.len() as u64 * VERTEX_STRIDE,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_bytes = build_indices(model.verts.len());
        let indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Personagem indices"),
            size: index_bytes.len() as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&indices, 0, &index_bytes);

        let extent = wgpu::Extent3d { width: TEX_SIZE, height: TEX_SIZE, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Personagem texture"),
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
            CHARS[i].texture,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(TEX_SIZE * 4),
                rows_per_image: Some(TEX_SIZE),
            },
            extent,
        );
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        chars.push(CharGpu {
            uniform,
            vertices,
            indices,
            index_count: (model.verts.len() / 4 * 6) as u32,
            tex_view,
        });
    }

    (Shared { format, pipeline, layout, sampler, depth_view: None }, chars)
}

fn push_f32s(out: &mut Vec<u8>, values: &[f32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// A esfera em volta da personagem esta dentro do campo de visao?
fn visible(vp: &Mat4, c: Vec3, r: f32) -> bool {
    let m = vp.to_cols_array();
    let row = |i: usize| Vec4::new(m[i], m[4 + i], m[8 + i], m[12 + i]);
    let r3 = row(3);
    for p in [r3 + row(0), r3 - row(0), r3 + row(1), r3 - row(1)] {
        let d = p.x * c.x + p.y * c.y + p.z * c.z + p.w;
        let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        if d < -r * len {
            return false;
        }
    }
    true
}

/// Chamado todo quadro com o mundo ja desenhado.
pub(super) fn draw(world: &mut WorldDrawn<'_>) {
    let mut guard = match STATE.lock() {
        Ok(g) => g,
        Err(_) => return,
    };
    let state = guard.get_or_insert_with(|| State {
        models: CHARS
            .iter()
            .map(|c| {
                let mut m = parse_model(c.model);
                m.idle = parse_anim(c.anim_idle);
                m.walk = parse_anim(c.anim_walk);
                m
            })
            .collect(),
        shared: None,
        chars: Vec::new(),
        anchor: None,
        right: Vec3::X,
        yaw: 0.0,
        start: std::time::Instant::now(),
        last_t: 0.0,
        wander: Vec::new(),
    });
    let t = state.start.elapsed().as_secs_f32();

    // Posicao: fila de personagens na frente da camera, viradas pra ela, no chao.
    let camera = world.camera;
    let too_far = match state.anchor {
        Some(a) => (a - camera).length() > 48.0,
        None => true,
    };
    if too_far {
        let inv = world.view_proj.inverse();
        let p = inv * Vec4::new(0.0, 0.0, 0.5, 1.0);
        let mut dir = Vec3::new(p.x / p.w, 0.0, p.z / p.w);
        if !dir.is_finite() || dir.length_squared() < 1e-8 {
            dir = Vec3::new(0.0, 0.0, -1.0);
        }
        let dir = dir.normalize();
        state.anchor = Some(camera + DVec3::new(dir.x as f64, 0.0, dir.z as f64) * SPAWN_DIST - DVec3::new(0.0, 1.62, 0.0));
        // Direita de quem olha na direcao `dir`.
        state.right = Vec3::new(-dir.z, 0.0, dir.x);
        // O rosto do modelo olha pra -Z; queremos que olhe pra camera (-dir).
        state.yaw = (-dir.x).atan2(-dir.z);
    }
    let anchor = match state.anchor {
        Some(a) => a,
        None => return,
    };

    let dt = (t - state.last_t).clamp(0.0, 0.1);
    state.last_t = t;
    let home_yaw = state.yaw;
    if too_far || state.wander.len() != CHARS.len() {
        state.wander = (0..CHARS.len()).map(|i| Wander::new(home_yaw, i)).collect();
    }
    for w in state.wander.iter_mut() {
        step_wander(w, dt, home_yaw);
    }

    let needs_gpu = match &state.shared {
        Some(s) => s.format != world.color_format,
        None => true,
    };
    if needs_gpu {
        let (shared, chars) = make_gpu(world, &state.models);
        state.shared = Some(shared);
        state.chars = chars;
    }
    let rebuild = too_far || needs_gpu;
    let State { models, shared, chars, right, wander, .. } = state;
    let shared = match shared {
        Some(s) => s,
        None => return,
    };
    if rebuild {
        for (i, model) in models.iter().enumerate() {
            let bytes = build_vertices(model, home_yaw, &Pose { t_idle: t, t_walk: 0.0, w: 0.0 });
            world.queue.write_buffer(&chars[i].vertices, 0, &bytes);
        }
    }

    // Quais personagens estao na tela (fora dela nao gasta GPU nenhuma).
    let count = chars.len();
    let mut offsets: Vec<Vec3> = Vec::with_capacity(count);
    let mut shown: Vec<usize> = Vec::with_capacity(count);
    for i in 0..count {
        let slide = (i as f32 - (count as f32 - 1.0) / 2.0) * SPACING;
        let pos = anchor + DVec3::new((right.x * slide + wander[i].x) as f64, 0.0, (right.z * slide + wander[i].z) as f64);
        let offset = (pos - camera).as_vec3();
        if visible(&world.view_proj, offset + Vec3::new(0.0, 1.0, 0.0), 1.6) {
            shown.push(i);
        }
        offsets.push(offset);
    }
    if shown.is_empty() {
        return;
    }
    // Animacao: refaz a pose so dos personagens que aparecem.
    for &i in &shown {
        let wd = &wander[i];
        let bytes = build_vertices(&models[i], wd.yaw, &Pose { t_idle: t, t_walk: wd.wt, w: wd.w });
        world.queue.write_buffer(&chars[i].vertices, 0, &bytes);
    }

    // Profundidade propria, pra elas nao se atravessarem.
    let size = world.size;
    let depth_ok = matches!(&shared.depth_view, Some((_, s)) if *s == size);
    if !depth_ok {
        let texture = world.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Personagens depth"),
            size: wgpu::Extent3d { width: size[0].max(1), height: size[1].max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        shared.depth_view = Some((texture.create_view(&wgpu::TextureViewDescriptor::default()), size));
    }
    let depth_view = match &shared.depth_view {
        Some((v, _)) => v,
        None => return,
    };

    let mut groups: Vec<wgpu::BindGroup> = Vec::with_capacity(shown.len());
    for &i in &shown {
        let offset = offsets[i];
        let mut u: Vec<u8> = Vec::with_capacity(112);
        push_f32s(&mut u, &world.jittered.to_cols_array());
        push_f32s(&mut u, &[offset.x, offset.y, offset.z, 0.0]);
        push_f32s(&mut u, &[size[0] as f32, size[1] as f32, 0.0, 0.0]);
        push_f32s(&mut u, &[world.depth_size[0] as f32, world.depth_size[1] as f32, 0.0, 0.0]);
        world.queue.write_buffer(&chars[i].uniform, 0, &u);
        groups.push(world.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Personagem"),
            layout: &shared.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: chars[i].uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&chars[i].tex_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&shared.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(world.depth) },
            ],
        }));
    }

    let mut pass = world.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Personagens"),
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
    pass.set_pipeline(&shared.pipeline);
    for (k, &i) in shown.iter().enumerate() {
        pass.set_bind_group(0, &groups[k], &[]);
        pass.set_vertex_buffer(0, chars[i].vertices.slice(..));
        pass.set_index_buffer(chars[i].indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..chars[i].index_count, 0, 0..1);
    }
}
