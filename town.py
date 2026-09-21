#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Jev小镇 (Jevons Town) — a small town where every resident's decision is a
Jev call, and you can watch the machine think.

Engine + HTTP server. Run:  uv run town.py  then open http://127.0.0.1:8787

Design notes
------------
* Every resident gets their OWN state — only what they can see from where they
  stand. The alley is visible only from the alley, so anything that happens
  there is invisible to the rest of the town. This is the "视野决定判断" demo.
* One Jev call per resident per tick, fired in parallel. Each call carries that
  resident's limited view, so answers cannot leak across residents.
* Two scripted moments:
    - 黑箱包裹 (tick 8): an object with no stated contents appears. Residents near
      it are asked whether it is dangerous — a question their state cannot
      answer. Watch what a closed-world classifier does with that.
    - 闭眼测试 (tick 25): one resident's view goes blank. They keep deciding.
"""
import json
import os
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
JEV_ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
JEV_MODEL = "~typesafe/jev-latest"
M3_ENDPOINT = "https://api.minimaxi.com/v1/responses"
M3_MODEL = "MiniMax-M3"
OBSERVE_EVERY = 8          # 每多少刻让街区观察者看一眼
PORT = 8787
TICK_SECONDS = 2.6
MAX_TICKS = 200


def env_file():
    """密钥从哪来：优先 $JEV_TOWN_ENV 指定的文件，其次脚本同目录的 .env。"""
    for cand in (os.environ.get("JEV_TOWN_ENV"), HERE / ".env"):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def get_key(name: str) -> str:
    v = (os.environ.get(name) or "").strip().strip('"').strip("'")
    if v:
        return v
    f = env_file()
    if f:
        for line in f.read_text().splitlines():
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        f"缺少 {name}：设成环境变量，或写进 {HERE}/.env（格式 {name}=...）")


# ----------------------------------------------------------------- world
LOCATIONS = {
    "plaza":  {"label": "广场", "sees": ["plaza", "market", "docks", "tavern"]},
    "market": {"label": "集市", "sees": ["market", "plaza"]},
    "alley":  {"label": "后巷", "sees": ["alley"]},          # 只有身在其中才看得见
    "tavern": {"label": "酒馆", "sees": ["tavern", "plaza"]},
    "docks":  {"label": "码头", "sees": ["docks", "plaza"]},
}
DAY_PARTS = ["清晨", "上午", "正午", "午后", "黄昏", "入夜", "深夜"]

RESIDENTS = [
    {"name": "阿岚", "slug": "alan",     "role": "布商",     "traits": "谨慎，对陌生人和无主的东西保持距离", "loc": "market"},
    {"name": "老周", "slug": "laozhou",  "role": "船夫",     "traits": "见多识广，喜欢凑近看热闹",         "loc": "docks"},
    {"name": "青禾", "slug": "qinghe",   "role": "药铺学徒", "traits": "好奇心重，容易跟着别人走",         "loc": "market"},
    {"name": "铁头", "slug": "tietou",   "role": "搬运工",   "traits": "直率，看到异常会先喊人",           "loc": "docks"},
    {"name": "素娘", "slug": "suniang",  "role": "酒馆掌柜", "traits": "消息灵通，讨厌麻烦事进店",         "loc": "tavern"},
    {"name": "小满", "slug": "xiaoman",  "role": "跑腿少年", "traits": "爱冒险，什么都想碰一下",           "loc": "plaza"},
    {"name": "陈九", "slug": "chenjiu",  "role": "更夫",     "traits": "守规矩，习惯先观察再动",           "loc": "plaza"},
    {"name": "阿绣", "slug": "axiu",     "role": "绣娘",     "traits": "胆小，遇事第一反应是回屋",         "loc": "alley"},
    {"name": "老麻", "slug": "laoma",    "role": "赌徒",     "traits": "什么都敢赌，包括不该赌的",         "loc": "tavern"},
    {"name": "程砚", "slug": "chengyan", "role": "账房",     "traits": "凡事要凭据，没有凭据就不认",       "loc": "market"},
    {"name": "阿吉", "slug": "aji",      "role": "货郎",     "traits": "来往各处，见的人多",               "loc": "plaza"},
    {"name": "哑姑", "slug": "yagu",     "role": "扫地人",   "traits": "不说话，但什么都看在眼里",         "loc": "alley"},
]

QUESTIONS = [
    ("noticed", "noul", "这个人此刻是否察觉到任何异常或值得留意的事？"),
    ("caution", "score", "这个人此刻的戒备程度如何？"),
    ("action", "choice", "这个人在接下来的一小段时间里最可能做什么？"),
]
ACTION_CRITERIA = {
    "留在原处": "维持当前正在做的事，不改变位置",
    "靠近查看": "主动走近那个引起注意的东西或人",
    "避开远离": "绕开、退后或离开当前区域",
    "招呼他人": "出声提醒或叫上别人一起",
}


class Town:
    def __init__(self):
        self.lock = threading.Lock()
        self.tick = 0
        self.residents = [dict(r, blindfold=False, last=None, stuck=0) for r in RESIDENTS]
        self.objects = []          # {"label","loc","tick"}
        self.log = []              # every decision, for the UI
        self.events = []           # narrative events
        self.started = time.time()
        self.total_cost = 0.0
        self.total_decisions = 0
        self.subscribers: list[queue.Queue] = []
        self.key = get_key("OPENROUTER_API_KEY")
        try:
            self.m3key = get_key("MINIMAX_API_KEY")
        except SystemExit:
            self.m3key = ""
        self.observations = []     # 街区观察者历次简报
        self._observing = False
        self.pool = ThreadPoolExecutor(max_workers=12)
        # 默认暂停：打开页面不会自动开始烧钱，由用户按「开始」
        self.running = False
        self.cost_cap = float(os.environ.get("JEV_TOWN_COST_CAP", "0.05"))
        self.worlds = []           # 每刻一条世界快照，供回放与平行世界分叉

    def world_snapshot(self):
        return {
            "tick": self.tick,
            "decisions": self.total_decisions,
            "cost": round(self.total_cost, 6),
            "residents": [{"name": r["name"], "slug": r["slug"], "role": r["role"], "loc": r["loc"],
                           "blindfold": r["blindfold"], "last": r["last"],
                           "answers": r.get("last_answers") or {}} for r in self.residents],
            "objects": [{"label": o["label"], "loc": o["loc"]} for o in self.objects],
        }

    def reset(self):
        with self.lock:
            self.tick = 0
            self.residents = [dict(r, blindfold=False, last=None, stuck=0) for r in RESIDENTS]
            self.objects = []
            self.log = []
            self.events = []
            self.worlds = []
            self.observations = []
            self.started = time.time()
            self.total_cost = 0.0
            self.total_decisions = 0
        self.broadcast({"type": "reset"})
        self.broadcast(self.snapshot())

    def world_at(self, tick):
        with self.lock:
            if not self.worlds:
                return None
            i = max(0, min(int(tick) - 1, len(self.worlds) - 1))
            w = dict(self.worlds[i], total=len(self.worlds), worlds=len(self.worlds))
            w["events"] = [e for e in self.events if e["tick"] <= w["tick"]][-8:]
            w["recent"] = list(reversed(
                [r for r in self.log if r.get("tick", 0) <= w["tick"]][-12:]))
            w["observations"] = list(reversed(
                [o for o in self.observations if o["tick"] <= w["tick"]][-6:]))
            return w

    def diverge(self, first_tick):
        """平行世界：回到第 first_tick 刻的分叉点，**原样**重跑一遍。

        什么都不改——不改 prompt、不改 state、不改模型。Jev 本身是非确定性的
        （同样输入重复调用，概率会在 ±0.04 上下抖动，实测 4 次得 4 个不同分数），
        所以分歧会自己长出来。看的是：哪些决定本来就接近抛硬币，
        以及一次抖动进去之后，这个世界会走多远。
        """
        with self.lock:
            if first_tick < 1 or first_tick > len(self.worlds) + 1:
                return None, "分叉点超出范围"
            base = self.worlds[first_tick - 2] if first_tick > 1 else None
            residents = [dict(r,
                              blindfold=(base["residents"][i]["blindfold"] if base else False),
                              last=(base["residents"][i]["last"] if base else None), stuck=0)
                         for i, r in enumerate(RESIDENTS)]
            if base:
                by = {r["name"]: r for r in base["residents"]}
                for r in residents:
                    r["loc"] = by[r["name"]]["loc"]
            self.tick = first_tick - 1
            self.residents = residents
            self.objects = [dict(o, tick=first_tick - 1, opaque=o["label"] == "一个没有来历的包裹")
                            for o in (base["objects"] if base else [])]
            self.worlds = self.worlds[:first_tick - 1]
            self.running = True
        self.add_event("note", f"回到第 {first_tick} 刻，原样重跑一遍。")
        self.broadcast({"type": "diverge", "tick": first_tick})
        return first_tick, None

    def control(self, action):
        if action == "start":
            self.running = True
            self.add_event("note", "镇子醒了过来。")
        elif action == "pause":
            self.running = False
            self.add_event("note", "镇子停住了。")
        elif action == "reset":
            self.running = False
            self.reset()

    def set_cap(self, value):
        v = max(0.0, min(float(value), 100.0))
        self.cost_cap = round(v, 4)
        self.add_event("note", f"花费上限改成 ${self.cost_cap:.4f}。")
        self.broadcast(self.snapshot())

    # ---------------------------------------------------------- narration
    def add_event(self, kind, text):
        with self.lock:
            self.events.append({"tick": self.tick, "kind": kind, "text": text})
        self.broadcast({"type": "event", "kind": kind, "text": text, "tick": self.tick})

    def broadcast(self, msg):
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except Exception:
                pass

    def snapshot(self):
        with self.lock:
            return {
                "type": "snapshot",
                "tick": self.tick,
                "residents": [
                    {"name": r["name"], "slug": r["slug"], "role": r["role"], "loc": r["loc"],
                     "blindfold": r["blindfold"], "last": r["last"],
                     "answers": r.get("last_answers") or {}}
                    for r in self.residents
                ],
                "objects": self.objects,
                "events": self.events[-8:],
                "cost": round(self.total_cost, 6),
                "decisions": self.total_decisions,
                "elapsed": round(time.time() - self.started, 1),
                "running": self.running,
                "cost_cap": self.cost_cap,
                "worlds": len(self.worlds),
                "recent": list(reversed(self.log[-12:])),
                "observations": list(reversed(self.observations[-6:])),
                "locations": {k: v["label"] for k, v in LOCATIONS.items()},
            }

    # ---------------------------------------------------------- jev call
    def ask(self, resident, questions):
        """One Jev call carrying this resident's own limited view."""
        view = self.view_for(resident, questions)
        body = {"model": JEV_MODEL, "state": view["state"], "questions": view["questions"]}
        req = urllib.request.Request(
            JEV_ENDPOINT, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
        except Exception as exc:  # keep the town running even if one call fails
            return {"resident": resident["name"], "error": str(exc)[:200], "ms": 0}
        ms = (time.perf_counter() - t0) * 1000
        answers = data.get("answers", {})
        for a in answers.values():                     # 头两名差多少：这个决定到底有多"接近"
            probs = sorted((a.get("probabilities") or {}).values(), reverse=True)
            if len(probs) >= 2:
                a["margin"] = round(probs[0] - probs[1], 4)
        return {
            "resident": resident["name"],
            "slug": resident["slug"],
            "role": resident["role"],
            "loc": resident["loc"],
            "blindfold": resident["blindfold"],
            "ms": round(ms),
            "answers": answers,
            "cost": data.get("usage", {}).get("cost", 0.0),
            "model": data.get("model", ""),
            "view_excerpt": view["excerpt"],
            "underdetermined": view["underdetermined"],
        }

    def view_for(self, resident, questions):
        """Build the resident's view: only what they can see, plus their memory."""
        here = resident["loc"]
        visible = LOCATIONS[here]["sees"]
        lines = [f"【时间】{DAY_PARTS[min(self.tick // 12, len(DAY_PARTS) - 1)]}",
                 f"【你】{resident['name']}，{resident['role']}。{resident['traits']}",
                 f"【你所在】{LOCATIONS[here]['label']}"]

        if resident["blindfold"]:
            lines.append("【你能看到的】你的视线被完全遮挡，看不到任何东西，也听不到任何声音。")
        else:
            for loc in visible:
                who = [r["name"] for r in self.residents if r["loc"] == loc and r["name"] != resident["name"]]
                things = [o["label"] for o in self.objects if o["loc"] == loc]
                parts = []
                if who:
                    parts.append("、".join(who))
                if things:
                    parts.append("、".join(things))
                if parts:
                    lines.append(f"【{LOCATIONS[loc]['label']}】{('、'.join(parts))}")
            lines.append(f"【你能看到的】只有上面列出的这些。你看不到的地方，你不知道那里发生了什么。")

        if resident.get("last"):
            lines.append(f"【你上一刻做的事】{resident['last']}")

        underdetermined = False
        objects_here = [o for o in self.objects if o["loc"] in visible]
        for o in objects_here:
            if o.get("opaque"):
                lines.append(f"【关于「{o['label']}」】你只看到它在那里。没有人告诉过你它是什么、里面有什么。")
                underdetermined = True

        state = "\n".join(lines)
        qs = {}
        for key, kind, text in questions:
            if kind == "noul":
                qs[key] = {"type": "noul", "instructions": text}
            elif kind == "score":
                qs[key] = {"type": "score", "instructions": text,
                           "criteria": ["放松", "寻常", "警觉", "高度戒备"]}
            else:
                qs[key] = {"type": "choice", "instructions": text, "criteria": ACTION_CRITERIA}
        # the closed-world probes: only asked of someone who can see an opaque
        # object. Deliberately mixes question types — factual attributes the
        # state never mentions (where Jev answers "no" from absence) against a
        # speculative judgement (where it admits uncertainty).
        if underdetermined:
            obj = objects_here[0]["label"]
            qs["has_name_tag"] = {"type": "noul",
                                  "instructions": f"「{obj}」上写着收件人的姓名吗？"}
            qs["is_for_me"] = {"type": "noul",
                               "instructions": f"「{obj}」是给你的吗？"}
            qs["dangerous"] = {"type": "noul",
                               "instructions": f"「{obj}」是危险的吗？"}
        return {"state": state, "questions": qs, "excerpt": lines[-2] if len(lines) > 1 else state,
                "underdetermined": underdetermined}

    # ------------------------------------------------- 街区观察者（M3）
    def digest(self):
        """给观察者的结构化摘要：谁在哪、有什么、最近发生了什么。"""
        by = {}
        for r in self.residents:
            by.setdefault(r["loc"], []).append(r)
        lines = [f"【时刻】第 {self.tick} 刻（{DAY_PARTS[min(self.tick // 12, len(DAY_PARTS) - 1)]}）",
                 "【人在哪里】"]
        for k, v in LOCATIONS.items():
            who = by.get(k, [])
            if who:
                lines.append(f"- {v['label']}：" + "、".join(f"{r['name']}（{r['role']}）" for r in who))
        objs = [f"{o['label']}，在{LOCATIONS[o['loc']]['label']}，第 {o.get('tick', '?')} 刻出现"
                for o in self.objects]
        lines.append("【镇上的物件】" + ("；".join(objs) if objs else "无"))
        lines.append("【居民上一轮自己的判断】")
        for r in self.residents[:8]:
            a = (r.get("last_answers") or {}).get("action")
            if not a:
                continue
            m = a.get("m")
            close = "（这一票很接近，随时可能改主意）" if isinstance(m, (int, float)) and m < 0.25 else ""
            lines.append(f"- {r['name']}：{a.get('v')}{close}")
        ev = [f"第{e['tick']}刻 {e['text']}" for e in self.events[-4:]]
        lines.append("【最近发生的事】" + (" ｜ ".join(ev) if ev else "无"))
        return "\n".join(lines)

    def observe(self):
        """M3 从高处看一眼镇子，写一段简报。

        它拿到的是结构化摘要，不是画面；和居民层的 Jev 正好互为对照——
        一个只看得见自己眼前那一小块、快而便宜，一个看得见全局、慢而啰嗦。
        """
        with self.lock:
            tick = self.tick
            digest = self.digest()
        prompt = ("你是这个小镇的街区观察者。下面是你此刻掌握的镇子情况：\n\n" + digest +
                  "\n\n请用中文写 2-3 句写给镇长看的简报：镇子上正在发生什么、有没有值得注意的地方、"
                  "如果觉得哪里不对劲你建议先看哪里。只写简报正文，不要标题、不要客套话、不要复述清单。")
        body = {"model": M3_MODEL, "input": prompt, "max_output_tokens": 3000,
                "reasoning": {"effort": "minimal"}}
        req = urllib.request.Request(
            M3_ENDPOINT, data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self.m3key}", "Content-Type": "application/json"},
            method="POST")
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=150) as resp:
                data = json.loads(resp.read())
            text = (data.get("output_text") or "").strip()
        except Exception as exc:
            text = f"（这次没看成：{str(exc)[:120]}）"
        ms = round((time.perf_counter() - t0) * 1000)
        with self.lock:
            self.observations.append({"tick": tick, "text": text, "ms": ms})
            del self.observations[:-12]
        self.broadcast({"type": "observation", "tick": tick, "text": text, "ms": ms})
        return text

    def move_for(self, resident, act):
        """把选中的行动变成真的位移——人要在镇上走动，地图才有意义。"""
        here = resident["loc"]
        target = None
        for o in self.objects:                      # 自己能看见的那个显眼东西
            if o["loc"] == here:
                target = here
                break
            if o["loc"] in LOCATIONS[here]["sees"]:
                target = o["loc"]
        if act == "靠近查看" and target and target != here:
            return target
        if act == "避开远离":
            if target == here:
                away = [l for l in LOCATIONS[here]["sees"]
                        if l != here and not any(o["loc"] == l for o in self.objects)]
            else:
                away = [l for l in LOCATIONS[here]["sees"] if l != here]
            if away:                                 # 后巷只看得见自己，所以那里的人走不掉
                return away[self.tick % len(away)]
        return here

    # ---------------------------------------------------------- one tick
    def run_tick(self):
        with self.lock:
            self.tick += 1
            tick = self.tick
            residents = [dict(r) for r in self.residents]

        # --- scripted moments
        if tick == 8:
            with self.lock:
                self.objects.append({"label": "一个没有来历的包裹", "loc": "plaza", "opaque": True, "tick": tick})
            self.add_event("box", "广场正中多了一个没有来历的包裹。没有人看到它是谁放的，也没有人知道里面是什么。")
        if tick == 25:
            with self.lock:
                target = next(r for r in self.residents if r["name"] == "阿绣")
                target["blindfold"] = True
            self.add_event("blindfold", "阿绣的眼睛被蒙住了——从这一刻起，她什么都看不见、听不见。")

        futures = [(r, self.pool.submit(self.ask, r, QUESTIONS)) for r in residents]
        results = []
        for r, fut in futures:
            try:
                results.append(fut.result(timeout=90))
            except Exception as exc:
                results.append({"resident": r["name"], "error": str(exc)[:200]})

        # --- apply effects
        with self.lock:
            by_name = {r["name"]: r for r in self.residents}
            for res in results:
                if "answers" not in res:
                    continue
                me = by_name.get(res["resident"])
                if not me:
                    continue
                act = (res["answers"].get("action") or {}).get("choice")
                if act:
                    me["last"] = act
                    dest = self.move_for(me, act)
                    if dest != me["loc"]:
                        res["moved_from"], res["moved_to"] = me["loc"], dest
                        me["loc"] = dest
                me["last_answers"] = {
                    k: {"v": (a.get("choice") or a.get("score") or a.get("noul")),
                        "m": a.get("margin"),
                        "p": a.get("probabilities"),
                        "t": a.get("type")}
                    for k, a in res["answers"].items()}
                self.total_cost += res.get("cost") or 0.0
                self.total_decisions += len(res.get("answers", {}))
                res["tick"] = tick
                self.log.append(res)
            self.worlds.append(self.world_snapshot())      # 供回放与分叉
        self.broadcast({"type": "tick", "tick": tick, "results": results,
                        "cost": round(self.total_cost, 6),
                        "decisions": self.total_decisions})

    def loop(self):
        while self.tick < MAX_TICKS:
            if not self.running:
                time.sleep(0.4)          # 暂停时不产生任何 API 调用
                continue
            if self.total_cost >= self.cost_cap:
                if self.running:
                    self.running = False
                    self.add_event("note", f"累计花费到 ${self.cost_cap:.2f}，自动停下了。")
                continue
            t0 = time.time()
            try:
                self.run_tick()
            except Exception as exc:
                self.add_event("error", f"tick 出错：{exc}")
            # 街区观察者：每 OBSERVE_EVERY 刻看一眼，放在单独线程里不拖慢镇子
            if self.m3key and self.tick and self.tick % OBSERVE_EVERY == 0 and not self._observing:
                self._observing = True

                def _job():
                    try:
                        self.observe()
                    except Exception:
                        pass
                    finally:
                        self._observing = False

                threading.Thread(target=_job, daemon=True).start()
            time.sleep(max(0.1, TICK_SECONDS - (time.time() - t0)))


# ----------------------------------------------------------------- server
TOWN = Town()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        raw = self.path
        path, _, qs = raw.partition("?")
        q = {}
        for kv in qs.split("&"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                q[k] = v
        if path in ("/", "/index.html"):
            self._send(200, (HERE / "ui.html").read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/assets/"):
            name = Path(path[len("/assets/"):]).name  # no traversal
            f = HERE / "assets" / name
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".svg"):
                ctype = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                         ".webp": "image/webp", ".svg": "image/svg+xml"}[f.suffix.lower()]
                self._send(200, f.read_bytes(), ctype)
            else:
                self._send(404, json.dumps({"error": "no such asset"}))
        elif path == "/replay":
            try:
                tick = int(q.get("tick", "0"))
            except ValueError:
                self._send(400, json.dumps({"error": "tick must be an int"}))
                return
            w = TOWN.world_at(tick)
            if w is None:
                self._send(404, json.dumps({"error": "还没有跑过任何一刻"}))
                return
            self._send(200, json.dumps(w, ensure_ascii=False))
        elif path == "/state":
            self._send(200, json.dumps(TOWN.snapshot(), ensure_ascii=False))
        elif path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            TOWN.subscribers.append(q)
            try:
                self.wfile.write(b"retry: 2000\n\n")
                self.wfile.write(("data: " + json.dumps(TOWN.snapshot(), ensure_ascii=False) + "\n\n").encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=20)
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        continue
                    self.wfile.write(("data: " + json.dumps(msg, ensure_ascii=False) + "\n\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                if q in TOWN.subscribers:
                    TOWN.subscribers.remove(q)
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        if self.path != "/control":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = json.loads(self.rfile.read(n) or b"{}") or {}
            action = raw.get("action")
        except Exception:
            self._send(400, json.dumps({"error": "bad body"}))
            return
        if action == "diverge":
            tick, err = TOWN.diverge(int(raw.get("tick") or 1))
            if err:
                self._send(400, json.dumps({"error": err}, ensure_ascii=False))
                return
        elif action == "set_cap":
            try:
                TOWN.set_cap(raw.get("value"))
            except (TypeError, ValueError):
                self._send(400, json.dumps({"error": "value must be a number"}))
                return
        elif action in ("start", "pause", "reset"):
            TOWN.control(action)
        else:
            self._send(400, json.dumps({"error": "action must be start|pause|reset|diverge|set_cap"}))
            return
        self._send(200, json.dumps(TOWN.snapshot(), ensure_ascii=False))


def main():
    threading.Thread(target=TOWN.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Jev小镇 已启动 → http://127.0.0.1:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
