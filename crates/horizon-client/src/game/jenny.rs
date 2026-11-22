//! /jenny: monta a Jenny parada na frente do jogador (11 pecas item_display).
use super::{
    Game,
    native_entities::NativeDisplay,
    native_server::{NativeCommand, NativeServerPlayer},
};
use horizon_world::nbt::{Compound, Tag};

pub(super) fn register(commands: &mut Vec<NativeCommand>) {
    commands.push(NativeCommand::new("jenny"));
}

// MATH-BEGIN
const PARTS: [(&str, [f64; 3], [f64; 4]); 11] = [
    ("fapcraft:jenny/torso", [0.00000, 16.16370, -1.80300], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/hip", [0.00000, 16.16208, -0.10120], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/leg_l", [2.21076, 11.65702, -0.04706], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/leg_r", [-2.21076, 11.65702, -0.04706], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/shin_l", [1.70143, 6.93085, 0.00000], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/shin_r", [-1.70143, 6.93085, 0.00000], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/head", [0.00000, 23.95000, 0.00000], [0.000000, 0.000000, 0.000000, 1.000000]),
    ("fapcraft:jenny/arm_l", [4.16530, 23.70800, 0.19800], [0.000000, 0.000000, 0.173648, 0.984808]),
    ("fapcraft:jenny/arm_r", [-3.94030, 23.70800, 0.19800], [0.000000, 0.000000, -0.173648, 0.984808]),
    ("fapcraft:jenny/lower_arm_l", [6.37976, 17.61620, -0.00200], [0.296137, 0.052217, 0.165611, 0.939228]),
    ("fapcraft:jenny/lower_arm_r", [-6.25737, 17.63429, -0.00000], [0.925417, -0.163176, -0.059391, 0.336824]),
];

fn qmul(a: [f64; 4], b: [f64; 4]) -> [f64; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}

fn rot_y(v: [f64; 3], b: f64) -> [f64; 3] {
    let (s, c) = b.sin_cos();
    [v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c]
}

fn layout(origin: [f64; 3], player_yaw_deg: f64) -> Vec<(&'static str, [f64; 3], [f32; 4])> {
    let beta = -player_yaw_deg.to_radians();
    let qy = [0.0, (beta / 2.0).sin(), 0.0, (beta / 2.0).cos()];
    PARTS
        .iter()
        .map(|(name, piv, q)| {
            let r = rot_y([piv[0] / 16.0, piv[1] / 16.0, piv[2] / 16.0], beta);
            let pos = [origin[0] + r[0], origin[1] + r[1], origin[2] + r[2]];
            let qt = qmul(qy, *q);
            (*name, pos, [qt[0] as f32, qt[1] as f32, qt[2] as f32, qt[3] as f32])
        })
        .collect()
}
// MATH-END

#[allow(dead_code)]
impl Game<'_> {
    pub(super) fn jenny_command(
        &mut self,
        player: &NativeServerPlayer,
        command: &str,
        ran: &mut bool,
    ) {
        if command.split_whitespace().next() != Some("jenny") {
            return;
        }
        *ran = true;
        let psi = (player.yaw as f64).to_radians();
        let origin = [
            player.position[0] - psi.sin() * 3.0,
            player.position[1],
            player.position[2] + psi.cos() * 3.0,
        ];
        let dim = player.dimension.clone();
        for (model, pos, q) in layout(origin, player.yaw as f64) {
            let mut comps = Compound::new();
            comps.insert("minecraft:item_model".into(), Tag::String(model.into()));
            let mut item = Compound::new();
            item.insert("id".into(), Tag::String("minecraft:stick".into()));
            item.insert("count".into(), Tag::Int(1));
            item.insert("components".into(), Tag::Compound(comps));
            let mut nbt = Compound::new();
            nbt.insert("item".into(), Tag::Compound(item));
            if let Some(id) = self.native_summon(&dim, "minecraft:item_display", pos, nbt) {
                let display = NativeDisplay {
                    left_rotation: q,
                    ..NativeDisplay::default()
                };
                let _ = self.native_set_display(&dim, id, &display);
            }
        }
    }
}
