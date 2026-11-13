"""Render a short GitHub-friendly animation of local self-routing."""

from __future__ import annotations

import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from controller import SimulationController
from models import ChordalFunnel8Architecture, NeuronState, RequestStatus


WIDTH, HEIGHT = 960, 600
FRAMES_PER_TICK = 2
TOTAL_TICKS = 25
FRAME_DURATION_MS = 100
OUTPUT = ROOT / "assets" / "algorithm-demo.gif"


def _font(size: int, bold: bool = False):
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = _font(25, bold=True)
FONT_SUBTITLE = _font(14)
FONT_LABEL = _font(11, bold=True)
FONT_SMALL = _font(12)
FONT_TINY = _font(10)


def _point(node: dict) -> tuple[int, int]:
    return 55 + int(float(node["x"]) * 650), 100 + int(float(node["y"]) * 430)


def _hex(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def _mix(color: str, amount: float) -> tuple[int, int, int]:
    r, g, b = _hex(color)
    return tuple(max(0, min(255, int(channel * amount))) for channel in (r, g, b))


def _draw_background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#09111c")
    draw = ImageDraw.Draw(image)
    for y in range(HEIGHT):
        ratio = y / max(1, HEIGHT - 1)
        color = (8 + int(5 * ratio), 16 + int(8 * ratio), 27 + int(13 * ratio))
        draw.line((0, y, WIDTH, y), fill=color)
    draw.rounded_rectangle((22, 20, WIDTH - 22, HEIGHT - 20), radius=18, outline="#20364d", width=2)
    draw.rounded_rectangle((738, 86, WIDTH - 42, HEIGHT - 55), radius=14, fill="#0d1b2b", outline="#20364d", width=1)
    return image


def _render(architecture: ChordalFunnel8Architecture, phase: int) -> Image.Image:
    image = _draw_background()
    draw = ImageDraw.Draw(image)
    topology = architecture.get_topology_info()
    nodes = {tuple(node["address"]): node for node in topology["nodes"]}
    positions = {address: _point(node) for address, node in nodes.items()}

    draw.text((48, 36), "НС АЦП", font=FONT_TITLE, fill="#e7f3ff")
    draw.text((183, 42), "Chordal-Funnel-8 · local self-routing", font=FONT_SUBTITLE, fill="#81a6c9")

    for edge in topology["edges"]:
        source = tuple(edge["from"])
        target = tuple(edge["to"])
        if source not in positions or target not in positions:
            continue
        color = "#28506c" if edge.get("type") == "primary" else "#1d3c52"
        draw.line((*positions[source], *positions[target]), fill=color, width=2)

    # Draw a soft glow for every active route before the crisp route itself.
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for request in architecture.all_requests:
        if not request.path:
            continue
        points = [positions[address] for address in request.path if address in positions]
        if len(points) > 1:
            glow_draw.line(points, fill=(*_hex(request.color), 130), width=12, joint="curve")
    image = Image.alpha_composite(image.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(7))).convert("RGB")
    draw = ImageDraw.Draw(image)

    for request in architecture.all_requests:
        points = [positions[address] for address in request.path if address in positions]
        if len(points) > 1:
            draw.line(points, fill=request.color, width=4, joint="curve")
        for address in request.path[:-1]:
            x, y = positions[address]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=_mix(request.color, 0.65))

    for address, node in nodes.items():
        x, y = positions[address]
        neuron = architecture.get_neuron(address)
        radius = 11 if node.get("is_input") else 9
        fill = "#f6c453" if node.get("is_input") else "#18364d"
        outline = "#4f89b6"
        if neuron is not None and neuron.state == NeuronState.FAULTY:
            fill, outline = "#a9324f", "#ff7591"
        elif neuron is not None and neuron.state == NeuronState.BUSY and neuron.current_request is not None:
            fill, outline = neuron.current_request.color, "#f7fbff"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
        if neuron is not None and neuron.state == NeuronState.FAULTY:
            draw.line((x - 5, y - 5, x + 5, y + 5), fill="#ffd1d9", width=2)
            draw.line((x + 5, y - 5, x - 5, y + 5), fill="#ffd1d9", width=2)

    active_requests = [request for request in architecture.all_requests if request.status == RequestStatus.IN_PROGRESS]
    for request in active_requests:
        if not request.path:
            continue
        x, y = positions[request.path[-1]]
        pulse = 12 + int(4 * (0.5 + 0.5 * math.sin((phase + request.id) * math.pi)))
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.ellipse((x - pulse, y - pulse, x + pulse, y + pulse), outline=(*_hex(request.color), 180), width=3)
        image = Image.alpha_composite(image.convert("RGBA"), overlay.filter(ImageFilter.GaussianBlur(1))).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text((x + 13, y - 7), f"#{request.id}", font=FONT_TINY, fill="#ecf7ff")

    tick = architecture.current_tick
    completed = architecture.completed_count
    rejected = architecture.rejected_count
    active = len(active_requests)
    faulty = sum(1 for neuron in architecture.all_neurons() if neuron.state == NeuronState.FAULTY)

    draw.text((766, 112), "LOCAL ROUTING", font=FONT_LABEL, fill="#7be0c4")
    draw.text((766, 139), "fault bypass online", font=FONT_SMALL, fill="#b4c9dd")
    draw.line((766, 166, 900, 166), fill="#29465f", width=1)
    stats = [
        ("tick", f"{tick:02d}"),
        ("active", str(active)),
        ("completed", str(completed)),
        ("rejected", str(rejected)),
        ("faulty nodes", str(faulty)),
    ]
    y = 188
    for label, value in stats:
        draw.text((766, y), label, font=FONT_TINY, fill="#7393ad")
        draw.text((892, y), value, font=FONT_SMALL, fill="#e7f3ff", anchor="ra")
        y += 30

    draw.line((766, 345, 900, 345), fill="#29465f", width=1)
    draw.ellipse((769, 365, 781, 377), fill="#f6c453")
    draw.text((791, 361), "input", font=FONT_TINY, fill="#b4c9dd")
    draw.ellipse((769, 393, 781, 405), fill="#2ed6b0")
    draw.text((791, 389), "buffer", font=FONT_TINY, fill="#b4c9dd")
    draw.ellipse((769, 421, 781, 433), fill="#ff7591")
    draw.text((791, 417), "fault / reroute", font=FONT_TINY, fill="#b4c9dd")
    draw.text((766, 478), "No central dispatcher", font=FONT_TINY, fill="#7d9db6")
    draw.text((766, 496), "Every hop is local", font=FONT_TINY, fill="#7d9db6")
    draw.text((48, 555), "A request meets a failed node, takes the chord, and keeps moving.", font=FONT_SMALL, fill="#7197b3")
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    architecture = ChordalFunnel8Architecture(q_required=4, total_initial_requests=6)
    architecture.set_generation_ticks([1, 2, 3, 4, 5, 6])
    # Two layer-one failures force the chordal detour to become visible.
    architecture.toggle_neuron_fault(1, 1)
    architecture.toggle_neuron_fault(1, 2)
    controller = SimulationController(architecture)

    frames: list[Image.Image] = []
    for _ in range(TOTAL_TICKS):
        for phase in range(FRAMES_PER_TICK):
            frames.append(_render(architecture, phase))
        if not architecture.is_finished():
            controller.step_forward()
    # Keep the completed state on screen for the last beat.
    frames[-1] = _render(architecture, FRAMES_PER_TICK - 1)

    frames[0].save(
        OUTPUT,
        save_all=True,
        append_images=frames[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
        # Preserve the five-second timing; Pillow may coalesce duplicate state
        # frames, but it keeps their combined duration.
        optimize=False,
        disposal=2,
    )
    print(f"created {OUTPUT} ({len(frames)} frames, {len(frames) * FRAME_DURATION_MS / 1000:.1f}s)")


if __name__ == "__main__":
    main()
