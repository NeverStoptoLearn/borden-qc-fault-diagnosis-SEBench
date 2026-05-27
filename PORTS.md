# Port-aware local launch

Run:

```bash
python scripts/choose_ports.py
python scripts/start_asset_server.py --directory borden_qc_fault_diagnosis_package
```

The task JSON setup commands automatically try asset ports:
`8000 8001 8002 8010 18000 18001`.

If the SE-Bench judge API default `8080` is occupied, use the `judge_port`
printed by `choose_ports.py` when starting the SE-Bench judge service.
