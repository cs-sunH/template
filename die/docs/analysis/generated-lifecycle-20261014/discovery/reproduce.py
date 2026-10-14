"""Reproduce the saved generated return component, not a full acceptance run."""
from pathlib import Path
import json
from discovery import Discovery, Geometry

HERE = Path(__file__).resolve().parent
world = Discovery(Geometry(M=15985, N=63937, K=4097), cap=256)
try:
    report = world.run()
except ValueError as error:
    report = {"status": "failed_closed", "reason": str(error),
              "events": world.events, "skipped_wave_count": world.skipped_k}
report["generated_returns"] = world.outer_returns
report["middle_descriptors"] = len(world.descriptors)
(HERE / "generated-K4097.json").write_text(json.dumps(report, indent=2) + "\n")
if not world.outer_returns:
    raise SystemExit("No generated return proof: component reproduction failed")
certificate = {"status": "component_proof_only_full_simulation_failed_closed",
               "global_event_cap": 256,
               "geometry": {"M":15985, "N":63937, "K":4097, "tm":16, "tn":64, "tk":64},
               "offset": "1/20", "record": world.outer_returns[0]}
(HERE / "first-return-certificate.json").write_text(json.dumps(certificate, indent=2) + "\n")
print("Generated component proof; full discovery remains", report["status"])
