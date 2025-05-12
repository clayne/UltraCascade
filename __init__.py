import importlib

modules_to_import = [
    "custom_nodes.UltraCascade.nodes",
    "custom_nodes.UltraCascade.nodes_sag_rag",
    "custom_nodes.UltraCascade.nodes_stage_b_patcher",
]

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for module_name in modules_to_import:
    try:
        module = importlib.import_module(module_name)
        print(f"[UltraCascade] Imported {module_name}")
    except Exception as e:
        print(f"[UltraCascade] Failed to import {module_name}: {e}")
        continue

    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
        NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
