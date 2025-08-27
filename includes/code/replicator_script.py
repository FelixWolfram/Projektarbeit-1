from omni.isaac.kit import SimulationApp
import argparse
import sys

OUTPUT_DIR_PATH = ""
USD_FILE_PATH = ""
USED_CAMERA_INDEX = 10  # da mehrere Kameras in der Szene implementiert sind, spezifizieren, welche Kamera verwendet werden soll

# ---------- Argumente \& Konfiguration ----------
parser = argparse.ArgumentParser("Dataset generator")
parser.add_argument("--headless", type=bool, default=False)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--num_frames", type=int, default=75)
parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_PATH)
args, _ = parser.parse_known_args()

# ---------- Konfiguration der Bildqualität und Anzahl ----------
CONFIG = {
    "renderer": "RayTracedLighting",
    "headless": args.headless,
    "width": args.width,
    "height": args.height,
    "num_frames": args.num_frames,
    "samples_per_pixel_per_frame": 256,      
    "max_bounces": 8,                    
    "max_specular_transmission_bounces": 12, 
    "max_volume_bounces": 4, 
    "subdiv_refinement_level": 2,     
    "anti_aliasing": 5  
}

# ---------- Simulation starten ----------
simulation_app = SimulationApp(launch_config=CONFIG)

import omni.replicator.core as rep
import omni.usd

# ---------- Replicator-Einstellungen ----------
import carb.settings
carb.settings.get_settings().set(
    "/exts/omni.replicator.core/maxAssetLoadingTime",
    600.0
)
# ---------- Stage laden ----------
usd_path = USD_FILE_PATH
omni.usd.get_context().open_stage(usd_path, None)

# ---------- Replizierbare Objekte erstellen ----------
k09_machine = rep.create.from_usd(usd_path, semantics=[("class", "k09")])
rep.create.group([k09_machine])

with k09_machine:
    rep.modify.pose(
        position=(0, 0, 0),
        rotation=(0, 0, 0),
        scale=(1.0, 1.0, 1.0)
    )

# ---------- Kamera und Materialien finden ----------
stage = omni.usd.get_context().get_stage()
camera_paths = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetTypeName() == "Camera"]

all_material_paths = [
    str(prim.GetPath()) 
    for prim in stage.Traverse() 
    if prim.GetTypeName() == "Material" 
       and str(prim.GetPath()).startswith("/World/Looks/")
]

camera_index = USED_CAMERA_INDEX  

print(camera_paths[camera_index])

cam = rep.get.prims(path_pattern=camera_paths[camera_index])

render_product = rep.create.render_product(cam, (args.width, args.height))

# ---------- Writer ----------
writer = rep.WriterRegistry.get("KittiWriter")
writer.initialize(output_dir=args.output_dir,
                omit_semantic_type=True)
writer.attach(render_product)

with rep.get.prims(
    path_pattern="/World/Sheets/States_Sheets/JT45__2__unload_84_sheet/JT45__2__unload_84_sheet_temp_stl",
    prim_types=["Xform"]
):
    rep.modify.semantics(
        semantics=[("class", "sheet")]
    )

with rep.get.prims(
    path_pattern="/World/Sheets/States_Parts/*/*",
    prim_types=["Xform"]
):
    rep.modify.semantics(
        semantics=[("class", "part")]
    )


import random

def run_orchestrator():
    print("Starting Orchestrator")
    rep.orchestrator.run()

    # Warten bis App gestartet ist
    while not rep.orchestrator.get_is_started():
        simulation_app.update()

    # Frame-Schleife: Für jeden Frame Elemente randomisieren
    for frame in range(args.num_frames):
        random_material_path1 = random.choice(all_material_paths)
        random_material_path2 = random.choice(all_material_paths)
        random_material = rep.get.prims(path_pattern=random_material_path1, prim_types=["Material"])
        random_material_distractor = rep.get.prims(path_pattern=random_material_path2, prim_types=["Material"])

        # Randomisierung Lichter
        with rep.get.prims(path_pattern="RectLight"):
            rep.modify.attribute("color", rep.distribution.uniform((0.3, 0.3, 0.3), (1, 1, 1)))
            rep.modify.attribute("intensity", rep.distribution.normal(100000.0, 400000.0))
            rep.modify.visibility(rep.distribution.choice([True, False, False, False, False]))

        # Randomisierung Parts
        with rep.get.prims(
            path_pattern="/World/Sheets/States_Parts/*/*",
            prim_types=["Xform"]
        ):
            rep.modify.pose(rotation=(random.uniform(-4, 4), random.uniform(-2, 2), 0))
            rep.modify.visibility(rep.distribution.choice([True, True, False]))
            rep.modify.material(random_material)

        # Randomisierung Material des Blechs
        with rep.get.prims(
            path_pattern="/World/Sheets/States_Sheets/*/*",  # Sheet
            prim_types=["Xform"]
        ):
            rep.modify.material(random_material)

        # Randomisierung Störobjekte
        with rep.get.prims(
            path_pattern="/World/Sheets/Distractors/",
            prim_types=["Mesh"]
        ):
            rep.modify.pose(position=(random.uniform(-150, 50), random.uniform(-680, -570), random.uniform(140, 210)))
            rep.modify.material(random_material_distractor)

        # Randomisierung Kamera
        with rep.get.prims(path_pattern=camera_paths[camera_index]):
            rep.modify.pose(position=(random.uniform(265, 440), random.uniform(130, 245), random.uniform(190, 280)))

        # Frame rendern
        rep.orchestrator.step(rt_subframes=8, pause_timeline=True, delta_time=0.0)

    rep.BackendDispatch.wait_until_done()
    rep.orchestrator.stop()

run_orchestrator()
simulation_app.update()
