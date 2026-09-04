# LingBot Scene Presets

## Overview
Scene presets allow users to quickly load predefined scene configurations (prompt + events) into the LingBot initial scene panel.

## Features
- **Built-in presets**: 5 default scene presets (Dragon, Jet Ski Cruise, Noir Alley Combat, Water Blaster, Circuit Racer)
- **Default on load**: "Dragon" auto-applies when the page opens (no manual selection needed)
- **Save custom presets**: Save current scene configuration as a new preset
- **Persistent storage**: Presets saved in browser localStorage
- **Quick access**: Dropdown selector in "Quick Start" section
- **Images**: every preset's start image is committed under `apps/cam2v/web/assets/` and referenced by relative path, so the picker renders with no network access

## Usage

### Loading a Preset
1. Open LingBot WebRTC interface (the "Dragon" preset is pre-selected by default)
2. Find "Quick Start" dropdown in Initial Scene panel
3. Select a different preset from the dropdown
4. Preset prompt + events + start image auto-populate the form; the dropdown stays on the selected preset name

### In-Game Event Hotkeys
Once connected, each preset's events are playable via keyboard —
**Player Controls events use digit-key shortcuts (1-9)**, **Director
Controls events use letter-key shortcuts** from a fixed pool (`b, f, g,
h, m, n, o, p, r, t, u, v, x, y, z`) that deliberately excludes the
movement keys (w/a/s/d/q/e/i/j/k/l) so hotkeys never collide with driving
input. **Jump** always gets **Space** and **Crouch** always gets **Ctrl**
instead of a digit, matching common game convention — both also render as
their own row right next to the movement key grid instead of the general
event button list, since they're movement actions, not narrative
triggers. Every other event button shows its assigned digit/letter hotkey
(e.g. "Portal (1)", "Storm Rolls In (B)"). Press **c** to Clear the active
event — present and wired the same way on every game, since Clear isn't
part of any preset's catalog. Ignored while typing in a text field. Player
events beyond the digit pool or director events beyond the 15-letter pool
still work by click, just without a shown hotkey.

### Director Controls
Each preset's events are split into two categories, matching the original
REACTOR case files' `actor: "character"` (player-triggered) vs.
`actor: "environment"` (narrative/pacing) events. Both live in the *same*
panel, switched with a single **toggle switch** at the top — not two
separate panels or tab buttons. The panel's own heading also swaps between
"Player Controls" / "Director Controls" to match:
- **Off (Player)** — the regular action buttons + a "Custom Prompt" box +
  the movement key grid (w/a/s/d/q/e/i/j/k/l).
- **On (Director)** — the environment/pacing events (weather, hazards,
  wildlife, etc.) + its own separate "Director Prompt" box, so a director
  can send free-form direction text independently of the player's prompt.
  The movement key grid hides — a director doesn't drive movement.

The toggle only appears once director mode is on, reached either by adding
`?director` to the page URL (e.g. `?manual&director`) — which starts the
toggle already on — or by clicking **"Enable Director Mode"** in Player
Controls for any preset that has director events (no URL edit needed).
Player events get digit hotkeys, director events get letter hotkeys, so
the two never collide even though only one side's buttons are visible at
once. The health bar stays visible in both Player and Director Controls.

Both tabs' events are always uploaded to the server together regardless
of `?director` — the shared WebRTC protocol has no player/director
distinction, so this split is purely a client-side UI/visibility choice.

Hand-added events (via "Add" in the Initial Scene panel's Text Events
list, not from a built-in preset) default to Player and can be flipped to
Director with the **Director** checkbox on that event's own row — the live
Player/Director Controls buttons update immediately when toggled.

### Sharing a Preset via URL
Add `?preset=<slug>` to the page URL to land directly on a specific built-in
preset instead of the "Dragon" default — the slug is the preset name,
lowercased with spaces replaced by hyphens (e.g. `Water Blaster` →
`water-blaster`). Example: `http://<host>:8089/request_session?preset=circuit-racer`.
Unknown/missing slugs fall back to the default preset. This only shares
*which game loads*, not a live/running session — WebRTC still allows only
one active session per server process.

### Saving a Preset
1. Edit prompt and text events in the Initial Scene panel
2. Click "Save" button next to Quick Start dropdown
3. Enter a name for the preset (e.g., "My Custom Scene")
4. Preset is saved to browser localStorage
5. New preset appears in dropdown for future use

## Data Format

Presets are stored as JSON array in `localStorage["lingbot-presets"]`:

```json
[
  {
    "name": "Preset Name",
    "prompt": "Scene prompt text",
    "events": [
      {
        "event_id": "unique-id",
        "label": "Event Label",
        "prompt": "Event prompt text",
        "health": -10
      }
    ],
    "directorEvents": [],
    "hud": { "maxHealth": 100 }
  }
]
```

`health` is optional (omit for no health effect); `directorEvents` uses the
same shape as `events`.

## Adding a New Built-in Preset (Game)

Built-in presets live in
`integrations_v2/lingbot/apps/cam2v/web/scene_presets.json` — a plain JSON
array, fetched at page load by `loadScenePresets()` in `adapter.js` (not
inlined in the JS, so it can be hand-edited without touching code). To add
one:

1. **Write the base prompt** (1-3 sentences): subject + environment + style,
   third person (or first person for cockpit/POV scenes like Circuit
   Racer/Water Blaster). No camera-motion or input language — this app
   drives movement via the `w/a/s/d/q/e/i/j/k/l` keys and text events, not
   prose.
2. **Write events**: each a short one-sentence imperative/descriptive
   clause (`{ event_id, label, prompt, health? }`). `event_id` is a short
   lowercase token, unique within that preset's own `events`/`directorEvents`
   arrays combined (ids can repeat *across* presets — each preset's list is
   independent). Mix character-triggered actions (tricks, attacks) — put
   these in `events`, **at least 5 of them** — with environment beats
   (weather, other characters/vehicles appearing) — put these in
   `directorEvents` (see "Director Controls" above). Add an optional numeric
   `health` field where it makes narrative sense — negative for a hit/risk
   (e.g. a hazard or a combat move), positive for a recovery/reward beat,
   omit it entirely for a purely cosmetic trick with no stakes. **`events.length
   + directorEvents.length` must not exceed 20** — that's a hard server-side
   cap (`MAX_TEXT_EVENTS` in `apps/cam2v/scene.py`) applied to the *combined* total,
   not separately per category, since the server has no player/director
   concept at all (only a generic `category` string tag) — both arrays
   upload together as one flat list; going over it makes every connect
   attempt fail with "At most 20 text events are supported." `applyPreset()`
   also logs a client-side warning at selection time if a preset is over
   budget, so this should be caught before it reaches a failed connect.
   - If you have access to `REACTOR_js-sdk`'s `lib/lingbot-cases/*.json`
     (a richer, layered `base`/`camera`/`movement`/`events` scene format
     used by a different app), you can mine its `scene.base.default` and
     `scene.events[].detail` text for content and compress it down to this
     flatter prompt+events shape — drop the camera/movement layers and any
     `EXACTLY ONE ...` frame-count guard clauses, since this app doesn't use
     that layering. Reference copies of a few are kept in
     `apps/cam2v/web/assets/sources/` for exactly this purpose.
3. **Pick a start image**: commit it under `apps/cam2v/web/assets/` and
   reference it by relative path (`assets/<name>.jpg`), the way the built-in
   presets do — it ships with the package and the page needs no network
   access. A public `https://` URL still works if you would rather not commit
   a binary. Note the v1 server's remote-URL fetch, with its SSRF guard
   against private hosts, has no v2 equivalent yet: a URL is handed to the
   browser to load, not fetched server-side.
4. **Add the object** to the array in `scene_presets.json` (strict JSON —
   double-quoted keys, no trailing commas, no comments):
   ```json
   {
     "name": "My Game",
     "image": "assets/my-game.jpg",
     "prompt": "...",
     "events": [
       { "event_id": "thing1", "label": "Thing One", "prompt": "..." }
     ],
     "directorEvents": [],
     "hud": { "maxHealth": 100 }
   }
   ```
5. **(Optional) Make it the default**: the preset at index `0` is
   auto-applied on page load via `presetSelect.value = "0"; applyPreset(0)`
   in `mount()` — reorder the array (or edit those two lines) to change
   which preset that is.
6. **Reload the page** — `adapter.js` is served straight off disk on every
   request (the v2 server serves the application's `web_root()`), so a
   browser refresh picks up the change with no server restart needed, as
   long as you're running an editable (`pip install -e`) install.
7. Update the **Built-in Presets** list below and the preset count in
   **Features** to keep this doc in sync.

## Storage Location
- **Browser**: localStorage under key `lingbot-presets`
- **Persistence**: Survives page refresh, tab close
- **Scope**: Browser/domain specific
- **Cleared when**: Browser cache is cleared, incognito mode

## Built-in Presets

Event counts below are `player + director = total` (against the 20-event
combined cap — see step 2 under "Adding a New Built-in Preset"). Every
built-in preset has at least 5 player events.

### 1. Dragon (default)
- **Prompt**: "A soaring journey through a fantasy jungle on the back of a flying creature. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate, as the aerial voyage carries them toward an ancient gothic castle, its stonework growing clearer as it nears. Floating landmasses and cascading waterfalls fill the fantastical landscape below."
- **Player events (4)**: Jump, Portal, Storm, Fireworks.
- **Director events (6)**: Rival Dragon Appears, Wind Gust Rocks Mount, Castle Guards Fire Arrows, Meteor Shower, Aurora Lights the Sky, Griffin Gives Chase.
- 4 + 6 = 10 total. Portal/Storm/Fireworks are `DEFAULT_TEXT_EVENTS` from `apps/cam2v/scene.py` (also on `main`), not invented for this preset — Jump and all director events are additions specific to this preset (no source JSON exists for Dragon; the app's own example-00 default). Auto-selects on load.

### 2. Jet Ski Cruise
- **Prompt**: "Turquoise water near a sandy beach lined with palm trees. A man in a red life vest riding a white and red jet ski, keeping it on top of the water at all times."
- **Player events (4)**: Jump, One-Hand Wave, Jump Off, Jump Back On.
- **Director events (9)**: Dolphins Leap, Storm Rolls In, Rogue Wave, Shark Lunges, Waterspout Forms, Fuel Runs Low, Thrown from the Jet Ski, Fuel Cache Spotted, Calm Water Break.
- 4 + 9 = 13 total. Uses "Fuel" as its HUD label (`hud.healthLabel`) instead of "Health". Jump Off/Jump Back On are a matched risk/recovery pair (-5/+8); One-Hand Wave awards a small `+3` trick bonus; Fuel Cache Spotted (+10) and Calm Water Break (+5) are dedicated recovery beats so a full playthrough always has a way to regain fuel, not just drain it.

### 3. Noir Alley Combat
- **Prompt**: "A narrow urban alley at night, dark brick walls and heavy rain, shiny puddles on wet asphalt, yellow police tape, blue and red ambient light. A lone uniformed police officer in dark blue tactical gear holding a flashlight."
- **Player events (7)**: Jump, Draw Pistol (fires exactly one shot, health -8), Crouch, Punch Combo, Roundhouse Kick, Baton Strike, Dodge Roll.
- **Director events (5)**: Player Falls to Ground, Enemy Appears, Enemy Attacks, Chicken Walks In, Drone Appears.
- 7 + 5 = 12 total. Enemy Appears/Attacks were changed from plural to singular ("a single figure... no second figure, no group") to keep the model from generating a crowd when only one enemy is intended.

### 4. Water Blaster
- **Prompt**: "First-person point of view aiming out across a colourful floating inflatable aqua park on a calm green quarry lake under bright summer sun. A bare hand grips a blue and red toy water blaster at the lower right of the frame."
- **Player events (6)**: Jump, Crouch, Splash Blast, Raise Float Shield, Green Slime Blast, Dive.
- **Director events (4)**: Rival Blaster Ambush, Bathers Get Super Soakers, Crocodile Lunges, Giant Balloon Drops.
- 6 + 5 = 11 total.

### 5. Circuit Racer
- **Prompt**: "First-person cockpit view from inside a Formula 1 race car, gloved hands on the wheel and the glowing dash ahead, speeding down a sunlit asphalt racing circuit lined with red-and-white kerbs."
- **Player events (5)**: Jump, Kick Up Sparks, Lock-Up Smoke, Drift, DRS Boost.
- **Director events (7)**: Rain Sweeps In, Sun Glare, Tunnel Section, Road Fire, Checkered Flag, Puddle on the Track, Oil Slick Ahead.
- 5 + 7 = 12 total.

Noir Alley Combat, Water Blaster, Jet Ski Cruise, and Circuit Racer prompts/events are adapted from the richer layered scene definitions in `REACTOR_js-sdk`'s `lib/lingbot-cases/*.json` (kept as reference copies under `apps/cam2v/web/assets/sources/` in this repo) down to this app's flatter prompt+events format. Their start images are committed under `apps/cam2v/web/assets/` and referenced by relative path.

## Browser Console Access

View all saved presets:
```javascript
JSON.parse(localStorage.getItem("lingbot-presets"))
```

Clear all presets:
```javascript
localStorage.removeItem("lingbot-presets")
```

## Implementation Files
- **UI**: `integrations_v2/lingbot/apps/cam2v/web/adapter.js`
  - Top of file: Preset data definitions (`scenePresets`)
  - `makeSceneCard()`: Scene card HTML with dropdown + Save button
  - `saveCurrentPreset()` / `updatePresetDropdown()` / `loadSavedPresets()` / `applyPreset()`: Load/Save preset functions
  - `mount()`: selects + applies preset index 0 ("Dragon") on load, before `loadInitialScene()` runs
  - `presetSelect.addEventListener` / `savePresetButton.addEventListener`: Event listeners

## Future Enhancements
1. **Server-side storage**: Save presets to backend database
2. **Export/Import**: Download presets as JSON file
3. **Sharing**: Share presets via link or code
4. **Categories**: Organize presets by category
5. **Versioning**: Track preset changes over time
