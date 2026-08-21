# Crazy Robotaxi — Hosted Demo Runbook

Live-edit flagship build (branch `robotaxi-463-live-edit`): fused per-state
drift corrector, photoreal base corrector, v6 skins, coins (custom sprite via $COIN_SPRITE, else procedural),
weather events, obstacle events, MJPEG browser HUD with a shared-token gate.

## Start

```bash
cd <worktree>/demo_kit
DEMO_HOST=<internal-hostname> DEMO_PORT=8630 ./start_demo.sh
```

- `DEMO_HOST` defaults to `127.0.0.1` (localhost-only). Set it to the
  machine's internal hostname / IP to serve players over the VPN.
- `DEMO_PORT` defaults to `8630`.
- A random token is generated unless you pass one (`./start_demo.sh mytoken`
  or `DEMO_TOKEN=mytoken`).
- The script prints the full player URL, then launches the app inside tmux
  session `robotaxi-demo` under an auto-restart loop (10 s relaunch delay,
  max 5 restarts per rolling hour, everything appended to
  `demo_kit/logs/robotaxi-demo.log`).
- **Warmup takes ~4 minutes** (world-model warmup + corrector weight sets).
  The page serves immediately; frames start after warmup finishes. Watch for
  `[chunk-pipeline] warmup done` in the log.

## Stop

```bash
./stop_demo.sh
```

Kills the tmux session, terminates any leftover `crazy-robotaxi` process
(SIGTERM, then SIGKILL after 30 s), and prints the GPU memory to confirm the
card is free.

## Restart

```bash
./stop_demo.sh && ./start_demo.sh          # new random token
./stop_demo.sh && ./start_demo.sh <token>  # keep the old token/URL
```

If the app crashes on its own, the loop restarts it automatically after 10 s.
After 5 restarts within one hour the loop gives up (see the log tail for the
`giving up` marker) — investigate before starting again.

## Player URL

```
http://<DEMO_HOST>:<DEMO_PORT>/?token=<TOKEN>
```

The page reads the token from its own URL and attaches it to every request.
Requests without the token (or with a wrong one) get **403**. For curl
checks, either append `?token=...` or send header `X-Stream-Token: <TOKEN>`.

## Keys

| Key | Action |
| --- | --- |
| W / A / S / D (or arrows) | Drive (throttle / steer / brake-reverse) |
| Space | Handbrake |
| K | Cycle world skin (arcade → comic → cyberpunk → pixel → base) |
| C | Toggle coins |
| O | Spawn obstacle event |
| V | Cycle weather (rain → snow → storm → clear) |
| 1 / 2 / 3 | View: world-model RGB / HDMap / PhysX debug |
| R | Reset rollout / new game |

## Known limits

- **Single player.** One shared world state; everyone with the token drives
  the same car. Keep the token to one player at a time.
- **native_dit is off** (the native extension fails on this GB300, and skins /
  weather fundamentally require the non-native executor anyway). Steady-state
  is ~200–240 ms per chunk (~33 fps end-to-end).
- **Weather costs ~2x while active**: the two-prompt guidance window
  (20 chunks, reopened by the 8-chunk re-swap refresh) doubles per-chunk
  cost, so expect chunk times of ~310–630 ms during weather.
- **Storm + obstacle smears**: spawning an obstacle during storm weather is a
  known visual-smear combination; avoid demoing them together.
- **Skin-swap boundary chunks are slow** (~450–930 ms once, for the text
  re-encode) — a brief hitch right after pressing K is expected.
- **One GPU**: the demo owns the GB300 while running; stop it before training
  or eval jobs.
- Occasional dropped `/control` key presses have been observed even at
  ~0.8 s spacing; if a key seems ignored, press it again.

## Troubleshooting

- **No frames / black page**: warmup probably not finished; check
  `tail -f demo_kit/logs/robotaxi-demo.log` for `warmup done`. First frames
  arrive right after.
- **403 in the browser**: token missing or wrong — re-copy the exact URL
  printed by `start_demo.sh` (the token is also in the tmux launch command:
  `tmux list-sessions` / `tmux attach -t robotaxi-demo`).
- **Port already in use**: another instance is up. `./stop_demo.sh`, or pick
  another `DEMO_PORT`. The app log prints the blocking PID (via `ss`).
- **Loop gave up (5 restarts/hour)**: read the log around the exit markers;
  common causes are GPU OOM (another job took the card) or a missing
  checkpoint path. Fix, then `./start_demo.sh` again.
- **App alive but controls dead**: check that requests carry the token
  (`curl -s -o /dev/null -w '%{http_code}' 'http://HOST:PORT/state?token=TOKEN'`
  should be 200), then check the log for Python tracebacks.
- **GPU not freed after stop**: `nvidia-smi` — if memory is still held,
  `./stop_demo.sh` again (it escalates to SIGKILL) and re-check.

## Security note (honest)

This is a **shared token in the URL**, not real authentication:

- The token appears in browser history, proxy/access logs, and anything the
  player copies or screenshots. Treat the URL itself as the secret.
- Anyone with the token has **full control of the session** (driving, skins,
  weather, resets) — there is no per-user identity or rate limiting.
- Transport is plain HTTP (no TLS): the token and frames are visible to
  anyone who can sniff the path. **Internal VPN only** — never expose the
  port to the public internet.
- Rotating the token requires a restart (`stop_demo.sh && start_demo.sh`).
