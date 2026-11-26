"""Zombie outbreak with a social (likeability) layer, built with Mesa 3.5.3 from ontology.json (v0.3).

Every parameter, range and rule id below comes from ontology.json. Run:
  python mesa_social.py --run social --seed 1        one run
  python mesa_social.py --compare --seeds 3          baseline then social, one run at a time
  python mesa_social.py --serve --arm social         live view on http://localhost:8123
"""
import argparse, csv, json, math, os, random, sys, threading, time
import numpy as np
from scipy.spatial import cKDTree
import mesa
from mesa.space import ContinuousSpace

HERE = os.path.dirname(os.path.abspath(__file__))
ONT = json.load(open(os.path.join(HERE, "ontology.json")))
PAR = {k: v["value"] for k, v in ONT["parameters"].items() if isinstance(v, dict) and "value" in v}
ENV = ONT["environment"]
W_A, W_I = ONT["parameters"]["compat_weights"]["aggression"], ONT["parameters"]["compat_weights"]["intelligence"]
N_HUMANS, N_ZOMBIES, N_TOTAL = ENV["population"]["humans"], ENV["population"]["zombies"], ENV["population"]["total"]
SIDE = round(1000 * math.sqrt(N_TOTAL / 475))                      # ontology environment.map
MAXPOS = math.nextafter(SIDE, 0)                                   # legacy ContinuousSpace rejects x == side
PERC, COLL, BOND = ENV["radii"]["perception"], ENV["radii"]["collision"], ENV["radii"]["bond"]
DT, STEPS_PER_TICK, DAY = ENV["time"]["dt_per_step"], ENV["time"]["steps_per_tick"], ENV["time"]["ticks_per_day"]
NIGHT_START = ENV["time"]["night_starts_at_tick_of_day"]
HUMAN_SPEED = ONT["classes"]["Human"]["attributes"]["movement_speed"]["init"]
ZOMBIE_SPEED = ONT["classes"]["Zombie"]["attributes"]["movement_speed"]["init"]
CAP = PAR["capacity"] if "capacity" in PAR else ONT["relations"]["Likes"]["capacity"]
LIVING_HUMAN = ("healthy", "exposed", "infected")
SOCIAL_OK = ("healthy", "exposed")                                  # R8/R9: infected friends are not partners
CLUSTER_TICKS = (30, 60, 100)


def compat(a, b):                                                   # R7
    d = W_A * abs(a.aggression - b.aggression) / 100 + W_I * abs(a.intelligence - b.intelligence) / 9
    return max(-1.0, min(1.0, 1 - PAR["compat_slope"] * d))


class Person(mesa.Agent):
    """One class, mutable kind (implementation_notes): turning keeps agent_id and every attribute."""

    def __init__(self, model, kind, rng):
        super().__init__(model)
        self.kind = kind
        self.state = "healthy" if kind == "human" else "turned"
        self.life = 100.0
        self.strength = rng.uniform(5, 15)
        self.intelligence = rng.uniform(1, 10)
        self.susceptibility = rng.uniform(80, 99) if kind == "human" else 0.0
        self.aggression = rng.uniform(0, 100)
        self.speed = HUMAN_SPEED if kind == "human" else ZOMBIE_SPEED
        self.heading = rng.uniform(0, 2 * math.pi)
        self.fighting = False
        self.timer = 0
        self.bites_survived = 0
        self.held = {}      # partner id -> L (relations.Likes, at most 5)
        self.pending = {}   # transient, never shown

    @property
    def alive(self):
        return self.state != "dead"


class ZombieModel(mesa.Model):
    def __init__(self, seed=1, social=True):
        super().__init__(rng=seed)
        self.social, self.seed_value = social, seed
        arm = "social" if social else "baseline"
        rng_init = random.Random(f"init-{seed}")                    # same population in both arms
        self.rng = random.Random(f"dyn-{seed}-{arm}")               # dynamics stream
        self.space = ContinuousSpace(SIDE, SIDE, False)
        self.people = []
        for kind, n in (("human", N_HUMANS), ("zombie", N_ZOMBIES)):   # ids humans 1..750 then zombies
            for _ in range(n):
                p = Person(self, kind, rng_init)
                self.space.place_agent(p, (rng_init.uniform(0, SIDE), rng_init.uniform(0, SIDE)))
                self.people.append(p)
        self.by_id = {p.unique_id: p for p in self.people}
        self.tick, self.sub = 0, 0
        self.datacollector = mesa.DataCollector(model_reporters={
            "tick": lambda m: m.tick, "humans_alive": lambda m: m.count(LIVING_HUMAN),
            "humans_infected": lambda m: m.count(("infected",)),
            "humans_not_infected": lambda m: m.count(("healthy", "exposed")),
            "zombies_alive": lambda m: sum(1 for p in m.people if p.kind == "zombie" and p.alive)})
        self.datacollector.collect(self)
        self.clusters, self.checkpoints, self.snap, self.outcome = {}, {}, {}, None
        self.extinction_tick = None

    # ---- helpers
    def count(self, states):
        return sum(1 for p in self.people if p.kind == "human" and p.state in states)

    def is_night(self):                                             # uses the tick before the increment
        return (self.tick % DAY) >= NIGHT_START

    # ---- one sub-step: decide for all (R1,R2,R3,R9,R11), move all, then contact damage (R4)
    def step(self):
        living = [p for p in self.people if p.alive]
        H = [p for p in living if p.kind == "human"]
        Z = [p for p in living if p.kind == "zombie"]
        hp = np.array([p.pos for p in H]) if H else np.empty((0, 2))
        zp = np.array([p.pos for p in Z]) if Z else np.empty((0, 2))
        tH = cKDTree(hp) if H else None
        tZ = cKDTree(zp) if Z else None                              # one spatial index per step
        night, plans = self.is_night(), []
        near_z = tZ.query(hp, k=1, distance_upper_bound=PERC) if (H and Z) else None
        near_h = tH.query(zp, k=1, distance_upper_bound=PERC) if (H and Z) else None
        for i, p in enumerate(H):
            if night:                                                # R1
                continue
            if near_z is not None and near_z[1][i] < len(Z):         # R3: zombie in sight
                t = Z[near_z[1][i]].pos
                plans.append((p, "toward" if p.fighting else "away", t[0], t[1]))
                continue
            tgt = self._social_target(p) if self.social else None    # R9
            if tgt is not None:
                plans.append((p, "friend", tgt.pos[0], tgt.pos[1]))
            else:
                plans.append((p, "wander", 0, 0))
        for j, p in enumerate(Z):                                    # R2
            if near_h is not None and near_h[1][j] < len(H):
                t = H[near_h[1][j]].pos
                plans.append((p, "toward", t[0], t[1]))
            else:
                plans.append((p, "wander", 0, 0))
        plans.sort(key=lambda q: q[0].unique_id)
        for p, mode, tx, ty in plans:                                # move pass (positions are step-start targets)
            sp, x, y = p.speed * DT, p.pos[0], p.pos[1]
            if mode == "wander":                                     # R11
                p.heading += self.rng.uniform(-PAR["turn_rate"], PAR["turn_rate"]) * DT
                x += math.cos(p.heading) * sp
                y += math.sin(p.heading) * sp
            else:
                dx, dy = (tx - x, ty - y) if mode != "away" else (x - tx, y - ty)
                dist = math.hypot(dx, dy)
                if mode == "friend" and dist <= PAR["stop_distance"]:
                    continue
                if dist > 0:
                    p.heading = math.atan2(dy, dx)
                    x += dx / dist * sp
                    y += dy / dist * sp
            self.space.move_agent(p, (min(max(x, 0.0), MAXPOS), min(max(y, 0.0), MAXPOS)))
        self._contact_damage(H, Z)
        self.sub += 1
        if self.sub >= STEPS_PER_TICK:
            self.sub = 0
            self._end_of_tick()

    def _social_target(self, p):
        for q in H_near(self, p):
            return q
        return None

    def _contact_damage(self, H, Z):                                 # R4
        H = [p for p in H if p.alive]
        Z = [p for p in Z if p.alive]
        if not H or not Z:
            return
        tH = cKDTree(np.array([p.pos for p in H]))
        tZ = cKDTree(np.array([p.pos for p in Z]))
        dmg = {}
        for i, zs in enumerate(tH.query_ball_tree(tZ, COLL)):
            for j in zs:
                dmg[H[i].unique_id] = dmg.get(H[i].unique_id, 0.0) + Z[j].strength * PAR["attack_scale"] * DT
                dmg[Z[j].unique_id] = dmg.get(Z[j].unique_id, 0.0) + H[i].strength * PAR["attack_scale"] * DT
        for uid, d in dmg.items():                                   # simultaneous
            p = self.by_id[uid]
            p.life = max(0.0, p.life - d)
            if p.life <= 0:
                p.state = "dead"

    # ---- tick end: R5, R6, R10, R8, R3 stance, metrics
    def _end_of_tick(self):
        self.tick += 1
        living = [p for p in self.people if p.alive]
        H = [p for p in living if p.kind == "human"]
        Z = [p for p in living if p.kind == "zombie"]
        if H and Z:                                                  # R5 bites
            tZ = cKDTree(np.array([p.pos for p in Z]))
            for p in H:
                if p.state in SOCIAL_OK and tZ.query(p.pos, k=1, distance_upper_bound=COLL)[1] < len(Z):
                    if self.rng.uniform(0, 100) < p.susceptibility:
                        p.state, p.timer = "infected", 0
                    else:
                        p.state, p.timer = "exposed", 0
                        p.bites_survived += 1
                        p.susceptibility = max(0.0, p.susceptibility - PAR["susceptibility_decay"])
        for p in H:                                                  # R6
            if p.state == "infected":
                p.timer += 1
                if p.timer >= PAR["turn_ticks"]:
                    p.kind, p.state, p.speed, p.timer = "zombie", "turned", ZOMBIE_SPEED, 0
        for p in H:
            if p.state == "exposed":
                p.timer += 1
                if p.timer >= PAR["exposed_ticks"]:
                    p.state, p.timer = "healthy", 0
        self._purge()                                                # R10
        self._likeability()                                          # R8
        self._stances()                                              # R3
        self.datacollector.collect(self)
        if self.tick in PAR["checkpoint_ticks"]:
            self.checkpoints[self.tick] = (self.count(LIVING_HUMAN), self.count(SOCIAL_OK))
        if self.tick % 10 == 0 and self.tick in CLUSTER_TICKS:
            self._cluster_metrics()
        if self.tick == PAR["cluster_sample_tick"]:
            self._snapshot_clusters()
        if self.tick == PAR["cluster_sample_tick"] + PAR["outcome_window"]:
            self._outcome()
        if self.extinction_tick is None and self.count(LIVING_HUMAN) == 0:
            self.extinction_tick = self.tick

    def _purge(self):                                                # R10
        for p in self.people:
            if p.kind != "human" or p.state not in SOCIAL_OK:
                p.held.clear(); p.pending.clear()
        for p in self.people:
            if p.kind == "human" and p.alive:
                for d in (p.held, p.pending):
                    for q in [q for q in d if self.by_id[q].state not in SOCIAL_OK or self.by_id[q].kind != "human"]:
                        del d[q]

    def _likeability(self):                                          # R8
        H = [p for p in self.people if p.kind == "human" and p.state in SOCIAL_OK]
        if not H:
            return
        tree = cKDTree(np.array([p.pos for p in H]))
        new = {}
        for i, p in enumerate(H):
            near = {H[j].unique_id: H[j] for j in tree.query_ball_point(p.pos, BOND) if j != i}
            cand = {}
            for q, pq in near.items():
                old = p.held.get(q, p.pending.get(q, 0.0))
                cand[q] = max(-1.0, min(1.0, old + PAR["like_rate"] * compat(p, pq)))
            for d in (p.held, p.pending):
                for q, old in d.items():
                    if q not in near:
                        cand[q] = old * (1 - PAR["like_decay"])
            new[p.unique_id] = [(q, v) for q, v in cand.items() if abs(v) >= PAR["prune_below"]]
        for p in H:
            ranked = sorted(new[p.unique_id], key=lambda qv: (-abs(qv[1]), 0 if qv[0] in p.held else 1, qv[0]))
            p.held = dict(ranked[:CAP])
            p.pending = dict(ranked[CAP:])

    def _stances(self):                                              # R3, rolled for all living humans incl. night
        Z = [p for p in self.people if p.kind == "zombie" and p.alive]
        tZ = cKDTree(np.array([p.pos for p in Z])) if Z else None
        for p in self.people:
            if p.kind == "human" and p.alive:
                sees = tZ is not None and tZ.query(p.pos, k=1, distance_upper_bound=PERC)[1] < len(Z)
                p.fighting = bool(sees and self.rng.uniform(0, 100) < p.aggression)

    # ---- metrics
    def _components(self, ids, edges):
        parent = {i: i for i in ids}
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for a, b in edges:
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        groups = {}
        for i in ids:
            groups.setdefault(find(i), []).append(i)
        return [g for g in groups.values() if len(g) >= 2]            # singletons excluded

    def _homophily(self, groups):
        H = [p for p in self.people if p.kind == "human" and p.alive]
        if len(H) < 3 or not groups:
            return None
        allv = np.var([p.aggression for p in H], ddof=1)
        ssw, n, k = 0.0, 0, len(groups)
        for g in groups:
            a = np.array([self.by_id[i].aggression for i in g])
            ssw += float(((a - a.mean()) ** 2).sum()); n += len(g)
        if n - k <= 0 or allv == 0:
            return None
        return (ssw / (n - k)) / allv

    def spatial_groups(self):
        H = [p for p in self.people if p.kind == "human" and p.alive]
        if len(H) < 2:
            return []
        tree = cKDTree(np.array([p.pos for p in H]))
        edges = [(H[i].unique_id, H[j].unique_id) for i, j in tree.query_pairs(BOND)]
        return self._components([p.unique_id for p in H], edges)

    def like_groups(self):
        ids = [p.unique_id for p in self.people if p.kind == "human" and p.state in SOCIAL_OK]
        edges = [(p.unique_id, q) for p in self.people if p.unique_id in set(ids) for q, v in p.held.items() if v >= ONT["parameters"]["cluster_threshold"]["value"]]
        return self._components(ids, edges)

    def _cluster_metrics(self):
        out = {}
        for name, groups in (("spatial", self.spatial_groups()), ("likeability", self.like_groups())):
            out[name] = {"clusters": len(groups), "mean_size": float(np.mean([len(g) for g in groups])) if groups else 0.0,
                         "homophily": self._homophily(groups)}
        self.clusters[self.tick] = out

    def _snapshot_clusters(self):
        self.snap = {name: [(g, float(np.mean([self.by_id[i].aggression for i in g]))) for g in groups]
                     for name, groups in (("spatial", self.spatial_groups()), ("likeability", self.like_groups()))}

    def _outcome(self):
        res = {}
        for name, groups in self.snap.items():
            if len(groups) < 3:
                res[name] = None
                continue
            agg = [a for g, a in groups]
            lost = [sum(1 for i in g if self.by_id[i].state == "dead" or self.by_id[i].kind == "zombie") / len(g) for g, a in groups]
            res[name] = float(np.corrcoef(agg, lost)[0, 1]) if np.std(agg) > 0 and np.std(lost) > 0 else None
        self.outcome = res


def H_near(model, p):
    """R9 candidates for human p (called only when no zombie is in sight), best first."""
    best = None
    for q, v in p.held.items():
        if v < PAR["pull_threshold"]:
            continue
        o = model.by_id[q]
        if o.kind != "human" or o.state not in SOCIAL_OK:
            continue
        if math.hypot(o.pos[0] - p.pos[0], o.pos[1] - p.pos[1]) > PERC:
            continue
        key = (-v, q)
        if best is None or key < best[0]:
            best = (key, o)
    return [best[1]] if best else []


def validate(model):
    """Checks the constraints list and attribute ranges in ontology.json; returns a list of problems."""
    errs, ids, hattr = [], set(), ONT["classes"]["Human"]["attributes"]
    hstates, zstates = ONT["classes"]["Human"]["states"], ONT["classes"]["Zombie"]["states"]
    for p in model.people:
        if p.unique_id in ids: errs.append(f"duplicate id {p.unique_id}")
        ids.add(p.unique_id)
        if (p.life <= 0) != (p.state == "dead"): errs.append(f"{p.unique_id}: life/dead mismatch")
        if p.state not in (hstates if p.kind == "human" else zstates) and p.state != "dead": errs.append(f"{p.unique_id}: bad state {p.state}")
        lo, hi = hattr["life_value"]["valid"]
        if not lo <= p.life <= hi: errs.append(f"{p.unique_id}: life out of range")
        for a in ("strength", "intelligence", "aggression", "susceptibility"):
            lo, hi = hattr[a if a != "intelligence" else "intelligence"]["valid"]
            if not lo <= getattr(p, a) <= hi: errs.append(f"{p.unique_id}: {a} out of range")
        if not (0 <= p.pos[0] <= SIDE and 0 <= p.pos[1] <= SIDE and math.isfinite(p.heading)): errs.append(f"{p.unique_id}: outside map")
        if len(p.held) > CAP: errs.append(f"{p.unique_id}: more than {CAP} relationships")
        if p.kind == "zombie" and (p.held or p.pending): errs.append(f"{p.unique_id}: zombie holds relationships")
        for q, v in p.held.items():
            o = model.by_id[q]
            if q == p.unique_id: errs.append(f"{p.unique_id}: self relationship")
            if o.kind != "human" or o.state not in SOCIAL_OK: errs.append(f"{p.unique_id}: partner {q} is {o.kind}/{o.state}")
            if not -1 <= v <= 1: errs.append(f"{p.unique_id}: L out of range")
    if sorted(p.unique_id for p in model.people) != list(range(1, N_TOTAL + 1)): errs.append("ids not 1..N")
    return errs


def run(seed, social, cap=None, check_every=0, quiet=False):
    cap = cap or PAR["safety_cap_ticks"]
    m = ZombieModel(seed=seed, social=social)
    t0, end = time.time(), None
    while True:
        m.step()
        if m.sub == 0:
            if check_every and m.tick % check_every == 0:
                e = validate(m)
                if e: raise AssertionError(f"tick {m.tick}: {e[:5]}")
            if not quiet and m.tick % 50 == 0:
                print(f"  tick {m.tick}: humans {m.count(LIVING_HUMAN)} (infected {m.count(('infected',))}), zombies {sum(1 for p in m.people if p.kind=='zombie' and p.alive)}", flush=True)
            if m.count(LIVING_HUMAN) == 0: end = "extinct"; break
            if sum(1 for p in m.people if p.kind == "zombie" and p.alive) == 0 and m.count(("infected",)) == 0: end = "no_threat"; break
            if m.tick >= cap: end = "capped"; break
    df = m.datacollector.get_model_vars_dataframe()
    return {"seed": seed, "arm": "social" if social else "baseline", "ticks": m.tick, "end": end,
            "extinction_tick": m.extinction_tick, "checkpoints": m.checkpoints, "clusters": m.clusters,
            "outcome_corr": m.outcome, "humans_final": m.count(LIVING_HUMAN), "seconds": round(time.time() - t0, 1)}, df


def compare(seeds, outdir, check_every):
    os.makedirs(outdir, exist_ok=True)
    for seed in seeds:
        for social in (False, True):                                  # one run at a time
            print(f"seed {seed} {'social' if social else 'baseline'}", flush=True)
            res, df = run(seed, social, check_every=check_every)
            tag = f"{res['arm']}_seed{seed}"
            df.to_csv(os.path.join(outdir, tag + ".csv"), index_label="row")
            json.dump(res, open(os.path.join(outdir, tag + ".json"), "w"), indent=1, default=str)
            print("  ->", res["end"], "ticks", res["ticks"], "checkpoints", res["checkpoints"], flush=True)


# ---------------- live view (serves index.html, sprites, /agents /status /pause) ----------------
def serve(arm, seed, port=8123):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    model, lock, state = ZombieModel(seed=seed, social=(arm == "social")), threading.Lock(), {"paused": True}

    def loop():
        while True:
            if state["paused"]:
                time.sleep(0.1); continue
            with lock:
                model.step()
            time.sleep(0.012)   # about 7 ticks per second, watchable
    threading.Thread(target=loop, daemon=True).start()

    def agents_json():
        with lock:
            out = []
            for p in model.people:
                rel = [{"agent_id": q, "likeability": round(v, 3), "aggression": round(model.by_id[q].aggression, 1),
                        "intelligence": round(model.by_id[q].intelligence, 2)}
                       for q, v in sorted(p.held.items(), key=lambda kv: (-abs(kv[1]), kv[0]))]
                out.append({"agent_id": p.unique_id, "agent_type": p.kind, "health_status": p.state, "x": p.pos[0], "y": p.pos[1],
                            "heading": p.heading, "movement_speed": p.speed, "life_value": p.life, "strength": p.strength,
                            "intelligence": p.intelligence, "aggression": p.aggression, "susceptibility": p.susceptibility,
                            "fighting": p.fighting, "bites_survived": p.bites_survived, "status_timer": p.timer, "relationships": rel})
            return out

    def status_json():
        with lock:
            return {"tick": model.tick, "paused": state["paused"], "day": model.tick // DAY + 1, "night": model.is_night(),
                    "human_count": model.count(LIVING_HUMAN), "zombie_count": sum(1 for p in model.people if p.kind == "zombie" and p.alive),
                    "arm": arm, "width": SIDE, "height": SIDE, "day_length": DAY,
                    "history": [[t, h, z] for t, h, z in zip(model.datacollector.model_vars["tick"][1:],
                                                             model.datacollector.model_vars["humans_alive"][1:],
                                                             model.datacollector.model_vars["zombies_alive"][1:])]}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _send(self, body, ctype):
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            if self.path.split("?")[0] == "/pause":
                state["paused"] = not state["paused"]; return self._send(json.dumps({"paused": state["paused"]}).encode(), "application/json")
            self.send_error(404)
        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"): return self._send(open(os.path.join(HERE, "index.html"), "rb").read(), "text/html")
            if path == "/agents": return self._send(json.dumps(agents_json()).encode(), "application/json")
            if path == "/status": return self._send(json.dumps(status_json()).encode(), "application/json")
            if path == "/pause":
                state["paused"] = not state["paused"]; return self._send(json.dumps({"paused": state["paused"]}).encode(), "application/json")
            if path.startswith("/sprites/") and path.endswith(".png") and "/" not in path[len("/sprites/"):] and ".." not in path:
                f = os.path.join(HERE, path[1:])
                if os.path.isfile(f): return self._send(open(f, "rb").read(), "image/png")
            self.send_error(404)
    print(f"serving {arm} arm, seed {seed} on http://localhost:{port}  (paused; press the page's pause button)")
    ThreadingHTTPServer(("", port), Handler).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["baseline", "social"]); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--compare", action="store_true"); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--serve", action="store_true"); ap.add_argument("--arm", default="social"); ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--check", type=int, default=0, help="validate every N ticks"); ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    if a.serve: serve(a.arm, a.seed, a.port)
    elif a.compare: compare(range(1, a.seeds + 1), a.out, a.check)
    elif a.run:
        r, _ = run(a.seed, a.run == "social", check_every=a.check); print(json.dumps(r, indent=1, default=str))
    else: ap.print_help()
