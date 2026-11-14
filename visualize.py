"""Watch what the bot actually does: render a battle as a self-contained HTML
animation — no installs, just open the file in a browser.

    python visualize.py --run runs/hog26-self/bot --opponent defend --out battle.html
    python visualize.py --run runs/hog26-self/bot --opponent random
    python visualize.py --trace my_battle.msgpack --out replay.html

Blue (bottom) is the bot; red (top) is the opponent. Play/pause, speed,
scrub through the timeline, elixir bars and crowns at the sides.
"""

import argparse
import json
import math

from royalegym import load_trace, play_battle
from royalegym.replay import ReplayRecorder

from envs import build_env


def frames_from_trace(trace, every_nth_tick=5):
    h = trace.header
    subtile = h.subtile
    cards = {c.card_id: c.name for c in h.cards}
    frames = []
    for f in trace.frames[::every_nth_tick]:
        ents = []
        for e in f.entities:
            kind, card_id, hp, max_hp = e.kind, e.card_id, e.hp, e.max_hp
            if kind == 0 and card_id < 0:
                continue
            ents.append([
                round(e.x / subtile, 2),          # x in tiles
                round(e.y / subtile, 2),          # y in tiles
                int(e.team),
                int(kind),                        # 0 troop, 1 building, 2 king, 3 princess
                round(e.radius / subtile, 2),     # radius in tiles
                int(e.flying),
                round(hp / max_hp, 2) if max_hp > 0 else 1.0,
                cards.get(card_id, "?"),
            ])
        frames.append({
            "t": f.tick * h.tick_ms // 1000,      # game seconds
            "e": ents,
            "x": [round(m / 1000, 1) for m in f.elixir_milli],
            "c": list(f.crowns),
        })
    return frames


def arena_from_header(h):
    return {
        "tiles_x": h.tiles_x,
        "tiles_y": h.tiles_y,
        "water_rows": list(h.water_half_rows),     # half-tile rows of the river
        "bridges": [list(b) for b in h.bridges_half_cols],
        "grid": h.grid,
    }


HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Clash battle</title>
<script>window.onerror = function(m, s, l){
  document.body.innerHTML = '<pre style="color:#fff;background:#8b1a1a;padding:24px;font-size:14px">'
    + 'The viewer hit an error and could not draw:\n' + m + '\n(line ' + l + ')'
    + '</pre>';
};</script>
<style>
 body{background:#101418;color:#dfe6ee;font-family:system-ui,sans-serif;margin:0;
      display:flex;gap:18px;padding:18px;align-items:flex-start}
 #stage{position:relative}
 canvas{border-radius:8px;background:#20344c}
 .panel{min-width:230px}
 h1{font-size:16px;margin:0 0 10px}
 .bar{height:14px;border-radius:7px;background:#2a323b;overflow:hidden;margin:4px 0 10px}
 .bar>div{height:100%}
 #eb{background:#c247cf;width:0%} #er{background:#c247cf;width:0%}
 .crowns{font-size:22px;margin:6px 0}
 button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;padding:6px 14px;
        font-size:14px;cursor:pointer;margin-right:6px}
 select,input[type=range]{margin:6px 0}
 #tl{width:100%}
 .legend{font-size:12px;color:#9fb0c0;margin-top:14px;line-height:1.5}
 .chip{display:inline-block;padding:1px 7px;border-radius:8px;margin:2px;font-size:12px}
 .b{background:#1d4f9c}.r{background:#9c1d31}
</style></head><body>
<div id="stage"><canvas id="cv" width="432" height="768"></canvas></div>
<div class="panel">
 <h1 id="title">battle</h1>
 <div class="crowns">👑 <span id="cb">0</span> — <span id="cr">0</span></div>
 <div>Blue elixir <span id="ebt"></span></div><div class="bar"><div id="eb"></div></div>
 <div>Red elixir <span id="ert"></span></div><div class="bar"><div id="er"></div></div>
 <div>
  <button id="pp">⏸</button>
  <select id="sp"><option value="0.5">0.5×</option><option value="1" selected>1×</option>
   <option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select>
 </div>
 <input type="range" id="tl" min="0" max="100" value="0">
 <div id="clock">0:00</div>
 <div id="hands"></div>
 <div class="legend">Blue at the bottom is the bot. Circles are units (size =
  footprint, hollow = flying, bar = hp). Squares are towers. The purple flashes
  mark where a card was just played.</div>
</div>
<script>
const D = __DATA__;
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const TX = D.arena.tiles_x, TY = D.arena.tiles_y;
const S = cv.width / TX;
const wr = D.arena.water_rows, br = D.arena.bridges;
let fi = 0, playing = true, last = 0, speed = 1;

function inWater(yHalf){ return yHalf >= wr[0] && yHalf <= wr[1]; }
function onBridge(xHalf){
  return br.some(b => xHalf >= b[0] && xHalf <= b[1]);
}

function draw(){
  const f = D.frames[fi];
  ctx.clearRect(0,0,cv.width,cv.height);
  for(let ry=0; ry<TY*2; ry++){
    for(let rx=0; rx<TX*2; rx++){
      const water = inWater(ry) && !onBridge(rx);
      ctx.fillStyle = water ? '#2b5d8f' : (Math.floor(ry/2)+Math.floor(rx/2))%2 ? '#3a7d44' : '#43904e';
      ctx.fillRect(rx*S/2, ry*S/2, S/2+0.5, S/2+0.5);
    }
  }
  for(const d of f.d || []){
    ctx.strokeStyle = d[2] ? '#e05575' : '#7db2ff'; ctx.lineWidth = 2;
    ctx.strokeRect(d[0]*S-9, d[1]*S-9, 18, 18);
  }
  for(const e of f.e){
    const [x, y, team, kind, rad, fly, hpf, name] = e;
    const col = team ? '#d84a5f' : '#4a8fe0';
    const edge = team ? '#ff8296' : '#8fc1ff';
    if(kind >= 2){
      ctx.fillStyle = col;
      const w = rad*2*S;
      ctx.fillRect(x*S-w/2, y*S-w/2, w, w);
      ctx.fillStyle = '#fff';
      ctx.fillRect(x*S-w/2, y*S-w/2 + w*(1-hpf), w, w*hpf*0.16);
    } else {
      ctx.beginPath();
      ctx.arc(x*S, y*S, Math.max(rad*S,4), 0, 7);
      ctx.fillStyle = col; ctx.fill();
      if(fly){ ctx.strokeStyle = '#ffd54a'; ctx.lineWidth = 2; ctx.stroke(); }
      if(rad*S > 7 && hpf < 0.999){
        ctx.fillStyle = '#0b0e12';
        ctx.fillRect(x*S-9, y*S-Math.max(rad*S,4)-8, 18, 3);
        ctx.fillStyle = '#7CFC00';
        ctx.fillRect(x*S-9, y*S-Math.max(rad*S,4)-8, 18*hpf, 3);
      }
      if(rad*S > 9){
        ctx.fillStyle = '#fff'; ctx.font = '9px sans-serif'; ctx.textAlign='center';
        ctx.fillText(name.slice(0,7), x*S, y*S - Math.max(rad*S,4) - 11);
      }
    }
  }
  document.getElementById('cb').textContent = f.c[0];
  document.getElementById('cr').textContent = f.c[1];
  document.getElementById('eb').style.width = (f.x[0]*10)+'%';
  document.getElementById('er').style.width = (f.x[1]*10)+'%';
  document.getElementById('ebt').textContent = f.x[0].toFixed(1);
  document.getElementById('ert').textContent = f.x[1].toFixed(1);
  const m = Math.floor(f.t/60), s = Math.floor(f.t%60);
  document.getElementById('clock').textContent = m+':'+String(s).padStart(2,'0');
  document.getElementById('tl').value = 100*fi/(D.frames.length-1);
}

function loop(ts){
  if(playing && D.frames.length){
    if(ts - last > 500/speed){
      last = ts;
      fi = Math.min(fi+1, D.frames.length-1);
      if(fi === D.frames.length-1) playing = false;
      draw();
    }
  }
  requestAnimationFrame(loop);
}
document.getElementById('pp').onclick = () => {
  playing = !playing;
  if(playing && fi === D.frames.length-1) fi = 0;
  document.getElementById('pp').textContent = playing ? '⏸' : '▶';
};
document.getElementById('sp').onchange = e => speed = parseFloat(e.target.value);
document.getElementById('tl').oninput = e => {
  fi = Math.round(e.target.value/100*(D.frames.length-1)); draw();
};
if(D.title) document.getElementById('title').textContent = D.title;
draw(); requestAnimationFrame(loop);
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=None, help="trained bot dir (runs/.../bot)")
    p.add_argument("--opponent", default="random",
                   choices=["random", "noop", "defend", "push", "patient", "first-affordable"])
    p.add_argument("--deck", default="hog26")
    p.add_argument("--trace", default=None, help="render a saved .msgpack battle instead of playing one")
    p.add_argument("--out", default="battle.html")
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.trace:
        trace = load_trace(args.trace)
        title = args.trace
    else:
        from royalelearn import Learner
        from evaluate import PolicyOpponent, make_foes

        deck = args.deck if args.deck in ("hog26", "random") else args.deck.split(",")
        if args.run:
            blue = PolicyOpponent(Learner.load_policy(args.run, greedy=args.greedy))
            who = "bot"
        else:
            blue = "random"
            who = "random bot"
        foe_name, foe = make_foes()[args.opponent]
        seed = args.seed if args.seed is not None else int(__import__("random").randrange(1_000_000))
        recorder = ReplayRecorder()
        env = build_env(deck)
        env.recorder = recorder
        play_battle(env, blue=blue, red=foe, seed=seed)
        trace = recorder.trace
        title = f"{who} (blue) vs {foe_name} — seed {seed}"

    frames = frames_from_trace(trace)
    steps = trace.steps
    di = 0
    deploys = []
    for fr in frames:
        marks = []
        while di < len(steps) and steps[di].tick <= fr["t"] * 1000 / trace.header.tick_ms:
            for cmd, st in zip(steps[di].commands, steps[di].statuses):
                if st == 0:
                    marks.append([round(cmd.x / trace.header.subtile, 2),
                                  round(cmd.y / trace.header.subtile, 2),
                                  1 if cmd.team else 0])
            di += 1
        if marks:
            deploys.append(marks)
        else:
            deploys.append(None)
    for fr, d in zip(frames, deploys):
        if d:
            fr["d"] = d

    data = {"arena": arena_from_header(trace.header), "frames": frames, "title": title}
    if not frames:
        raise SystemExit(
            f"the battle recorded {len(trace.frames)} engine ticks but no drawable "
            "frames -- try another --seed"
        )
    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(args.out, "w") as fh:
        fh.write(html)
    import os
    print(f"wrote {os.path.abspath(args.out)}: {len(frames)} frames, {title} -- open that exact file in a browser")


if __name__ == "__main__":
    main()
