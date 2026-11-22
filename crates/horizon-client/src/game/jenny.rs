//! Teste do /jenny: um bloco flutuando na frente do jogador.
use super::{
    Game,
    native_entities::NativeDisplay,
    native_server::{NativeCommand, NativeServerPlayer},
};
use horizon_world::nbt::{Compound, Tag};

pub(super) fn register(commands: &mut Vec<NativeCommand>) {
    commands.push(NativeCommand::new("jenny"));
}

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
        let yaw = (player.yaw as f64).to_radians();
        let pos = [
            player.position[0] - yaw.sin() * 2.0,
            player.position[1] + 1.0,
            player.position[2] + yaw.cos() * 2.0,
        ];
        let mut item = Compound::new();
        item.insert("id".into(), Tag::String("minecraft:diamond_block".into()));
        item.insert("count".into(), Tag::Int(1));
        let mut nbt = Compound::new();
        nbt.insert("item".into(), Tag::Compound(item));
        let dim = player.dimension.clone();
        if let Some(id) = self.native_summon(&dim, "minecraft:item_display", pos, nbt) {
            let display = NativeDisplay {
                scale: [0.8; 3],
                ..NativeDisplay::default()
            };
            let _ = self.native_set_display(&dim, id, &display);
        }
    }
}
