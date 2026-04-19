"""Thin code emitters for a future native Puffer Ocean backend."""

from rddl2puffer.backends.puffer_c.emit_binding_c import emit_binding_c
from rddl2puffer.backends.puffer_c.emit_env_c import emit_env_c
from rddl2puffer.backends.puffer_c.emit_env_h import emit_env_header
from rddl2puffer.backends.puffer_c.emit_ini import emit_ini
from rddl2puffer.ir.nodes import IRProgram


def render_env_bundle(program: IRProgram, env_name: str) -> dict[str, str]:
    """Render the current minimal Puffer backend bundle into memory."""

    return {
        f"ocean/{env_name}/{env_name}.h": emit_env_header(program, env_name),
        f"ocean/{env_name}/{env_name}.c": emit_env_c(program, env_name),
        f"ocean/{env_name}/binding.c": emit_binding_c(program, env_name),
        f"config/{env_name}.ini": emit_ini(program, env_name),
    }


__all__ = ["emit_binding_c", "emit_env_c", "emit_env_header", "emit_ini", "render_env_bundle"]
