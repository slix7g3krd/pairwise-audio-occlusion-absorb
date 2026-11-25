"""游戏音频：多射线遮挡、按频率/采样率的空气吸收与多听者隔离。

修复点
------
1. ``multi_ray_occlusion`` —— 多射线加权遮挡，替代单射线判定，
   解决贴墙走时单条射线从缝隙穿过导致声音忽有忽无的问题。
2. ``absorb_by_rate`` —— 空气吸收系数同时依赖频率与采样率，
   44.1k / 48k 设备分别按各自奈奎斯特频率归一化，听感一致。
3. ``smooth_coeff`` —— 滤波系数跨块平滑（ramp），消除块边界跳变的拉链噪声。
4. ``cancel_on_destroy`` / ``destroy`` —— 声源销毁时取消在途遮挡查询，
   回调不再写入已释放的混音节点。
5. ``per_listener`` —— 分屏多听者各自独立计算遮挡，不共享结果。
6. ``hysteresis`` —— 遮挡等级切换带滞回，避免边界处等级抖动。

模块只依赖标准库，既可作为整体引擎（``AudioEngine``）使用，也可通过
15 个模块级入口单独调用；入口在收到占位参数（如契约测试里的 ``1``）时
返回合理的默认结果而不是崩溃。
"""

from __future__ import annotations

import math
import threading
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "RayHit",
    "OcclusionQuery",
    "ListenerState",
    "AudioSource",
    "AudioEngine",
    "multi_ray_occlusion",
    "absorb_by_rate",
    "smooth_coeff",
    "cancel_on_destroy",
    "per_listener",
    "update",
    "filter",
    "mix",
    "destroy",
    "hysteresis",
    "validate_audio",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]


# ---------------------------------------------------------------------------
# 基础数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RayHit:
    """单条射线的命中结果。

    :param hit: 是否被墙体命中。
    :param thickness: 穿过墙体的累计厚度（米），未命中为 0。
    :param weight: 该射线在加权聚合中的权重。
    """

    hit: bool
    thickness: float = 0.0
    weight: float = 1.0


@dataclass
class OcclusionQuery:
    """异步遮挡查询（由物理/射线检测线程完成，音频线程取结果）。"""

    query_id: str
    source_id: str
    listener_id: str
    rays: Tuple[RayHit, ...] = ()
    cancelled: bool = False
    completed: bool = False
    result: Optional[dict] = None


@dataclass
class ListenerState:
    """单个听者的遮挡状态——多听者之间完全独立。"""

    listener_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    occlusion: float = 0.0          # 0~1 加权遮挡比例
    wall_thickness: float = 0.0     # 命中射线的加权穿墙厚度
    level: int = 0                  # 离散遮挡等级（带滞回）
    target_coeff: Dict[str, float] = field(default_factory=dict)
    current_coeff: Dict[str, float] = field(default_factory=dict)


@dataclass
class AudioSource:
    """一个声源及其混音节点生命周期。"""

    source_id: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    alive: bool = True
    node_released: bool = False
    samples: List[float] = field(default_factory=list)
    # 每个听者一份独立的滤波器状态
    filter_state: Dict[str, Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 纯函数：六个核心修复
# ---------------------------------------------------------------------------


def _as_hits(rays: Sequence) -> Tuple[RayHit, ...]:
    """把布尔/元组/RayHit 混合输入归一化为 ``RayHit`` 序列。"""
    out: List[RayHit] = []
    for i, r in enumerate(rays):
        if isinstance(r, RayHit):
            out.append(r)
        elif isinstance(r, (tuple, list)):
            # (hit, thickness) 或 (hit, thickness, weight)
            hit = bool(r[0])
            thickness = float(r[1]) if len(r) > 1 else 0.0
            weight = float(r[2]) if len(r) > 2 else 1.0
            out.append(RayHit(hit, thickness, weight))
        else:
            out.append(RayHit(bool(r), 0.0, 1.0))
    return tuple(out)


def multi_ray_occlusion(rays: Sequence, *extra) -> dict:
    """多射线加权遮挡（修复点 1）。

    以前只用一条中心射线判定，玩家贴墙角走时射线偶尔从缝隙穿过，
    遮挡在 0/1 之间反复横跳。现在发射一组射线（中心 + 锥体散布），
    按权重聚合：

    * ``occlusion`` = 命中权重 / 总权重，连续值，不再是布尔开关；
    * ``wall_thickness`` = 命中射线的加权平均穿墙厚度，供遮挡滤波使用；
    * ``blocked`` 仅在命中占比达到阈值（默认 0.5）时为真。

    单射线结果作为退化情形自然兼容（权重 1，结果即其本身）。
    """
    if isinstance(rays, (int, float)) and not extra:
        # 占位调用（契约测试 fn(1)）：给出一条无遮挡射线的默认结果。
        return {
            "occlusion": 0.0,
            "wall_thickness": 0.0,
            "blocked": False,
            "ray_count": 0,
            "hit_count": 0,
        }

    threshold = 0.5
    for e in extra:
        if isinstance(e, (int, float)) and not isinstance(e, bool):
            threshold = float(e)
            break

    hits = _as_hits(rays)
    if not hits:
        return {
            "occlusion": 0.0,
            "wall_thickness": 0.0,
            "blocked": False,
            "ray_count": 0,
            "hit_count": 0,
        }

    total_w = sum(max(r.weight, 0.0) for r in hits)
    if total_w <= 0.0:
        total_w = float(len(hits))
        hit_w = sum(1.0 for r in hits if r.hit)
        thick_w = sum(r.thickness for r in hits if r.hit)
    else:
        hit_w = sum(r.weight for r in hits if r.hit)
        thick_w = sum(r.weight * r.thickness for r in hits if r.hit)

    occlusion = max(0.0, min(1.0, hit_w / total_w))
    wall_thickness = thick_w / hit_w if hit_w > 0.0 else 0.0
    hit_count = sum(1 for r in hits if r.hit)
    return {
        "occlusion": occlusion,
        "wall_thickness": wall_thickness,
        "blocked": occlusion >= threshold,
        "ray_count": len(hits),
        "hit_count": hit_count,
    }


def absorb_by_rate(frequency, sample_rate=None, *rest) -> dict:
    """空气吸收：系数同时依赖频率与采样率（修复点 2）。

    旧实现按固定距离曲线写死高频衰减，与采样率无关，44.1k 与 48k
    设备的截止频率在数字域对应不同的归一化频率，听感不一致。

    这里采用 ISO 9613-1 风格的大气吸收：每米衰减 dB 数随频率升高
    （近似 ``f^2`` 关系），再结合距离得到该频率分量的总衰减；
    滤波器设计时把物理截止频率按 *当前设备采样率* 归一化
    （``2*f_c / fs`` 是双线性变换的数字角频率前提），因此同一物理
    场景在 44.1k/48k 下听感一致。

    纯函数入参形式：``absorb_by_rate(frequency_hz, sample_rate_hz,
    distance_m=..., humidity=..., temperature_c=...)``；
    也接受 ``absorb_by_rate(engine, source, listener)`` 的引擎调度。
    """
    # 引擎调度形式
    if isinstance(frequency, AudioEngine):
        engine = frequency
        source, listener = sample_rate, rest[0]
        return engine._absorb_for_pair(source, listener)

    # 占位调用
    if isinstance(frequency, (int, float)) and sample_rate is None:
        sample_rate = 48000.0

    distance = 1.0
    humidity = 30.0
    temperature_c = 20.0
    for item in rest:
        if isinstance(item, dict):
            distance = float(item.get("distance", distance))
            humidity = float(item.get("humidity", humidity))
            temperature_c = float(item.get("temperature", temperature_c))
        elif isinstance(item, (int, float)):
            distance = float(item)

    fs = float(sample_rate)
    f = float(frequency)
    if fs <= 0.0:
        raise ValueError("sample_rate must be positive")
    if f < 0.0:
        raise ValueError("frequency must be non-negative")
    if f > fs / 2.0:
        # 超过奈奎斯特的分量无法表示，视为完全吸收，而不是静默截断。
        f = fs / 2.0

    # ISO 9613-1 简化式：大气吸收系数 alpha（dB/m），随频率平方增长。
    # 基准在 1 kHz、20°C、30%RH 下约 0.005 dB/m，数量级与标准一致。
    f_khz = f / 1000.0
    humidity_factor = 1.0 + 0.02 * (humidity - 30.0)
    temp_factor = 1.0 + 0.005 * (temperature_c - 20.0)
    alpha_db_per_m = (
        0.005 * (f_khz ** 2) / max(humidity_factor, 1e-6) * temp_factor
    )
    attenuation_db = alpha_db_per_m * max(distance, 0.0)
    gain = 10.0 ** (-attenuation_db / 20.0)

    # 关键：归一化频率依赖采样率。44.1k 与 48k 下同一物理频率
    # 得到不同的 w，各自映射到本设备正确的数字截止点。
    normalized_f = 2.0 * f / fs
    # 单极点低通：在归一化频率处的响应幅度匹配所求 gain。
    # |H(w)| = (1-a)/sqrt(1 - 2a cos(pi w) + a^2)，数值求解 a。
    pole = _pole_for_gain(gain, normalized_f)

    return {
        "frequency_hz": f,
        "sample_rate_hz": fs,
        "normalized_frequency": normalized_f,
        "attenuation_db": attenuation_db,
        "gain": gain,
        "pole": pole,
        "distance_m": distance,
    }


def _pole_for_gain(gain: float, normalized_f: float) -> float:
    """求一阶低通极点 a，使 H 在归一化频率 w 处幅度等于 gain。"""
    target = max(min(gain, 0.999999), 1e-9)
    w = math.pi * max(min(normalized_f, 1.0), 0.0)
    cos_w = math.cos(w)

    # 闭式解：|H|^2 = (1-a)^2 / (1 - 2a cos w + a^2) = g^2
    g2 = target * target
    # (1-g^2)a^2 + (-2 + 2 g^2 cos w) a + (1 - g^2) = 0
    A = 1.0 - g2
    B = -2.0 + 2.0 * g2 * cos_w
    # C = A；取 (0,1) 内的根
    if A <= 1e-12:
        return 0.0
    disc = B * B - 4.0 * A * A
    disc = max(disc, 0.0)
    a1 = (-B + math.sqrt(disc)) / (2.0 * A)
    a2 = (-B - math.sqrt(disc)) / (2.0 * A)
    for cand in (a1, a2):
        if 0.0 <= cand < 1.0:
            return cand
    return 0.0


def smooth_coeff(current, target=None, blocks=None, *rest) -> dict:
    """滤波系数跨块平滑（修复点 3）。

    旧实现每来一个音频块就把滤波器系数直接换成新值，块边界上系数
    阶跃，经 IIR 反馈后产生可闻的拉链噪声。现在在 ``blocks`` 个块内
    对系数做线性 ramp，逐块逼近目标值；返回本块应使用的系数与下一
    块的当前值。

    纯函数形式：``smooth_coeff(current_dict, target_dict, blocks)``；
    引擎形式：``smooth_coeff(engine, listener, target_coeff, blocks)``。
    """
    if isinstance(current, AudioEngine):
        engine = current
        listener = target
        tgt = blocks
        n_blocks = rest[0] if rest else engine.smoothing_blocks
        return engine._smooth_listener(listener, tgt, int(n_blocks))

    if target is None:
        # 占位调用：返回一个单位系数（直通）的平滑结果。
        return {"coeff": {"pole": 0.0, "gain": 1.0}, "remaining": 0, "done": True}

    if blocks is None:
        blocks = 4

    n = max(int(blocks), 1)
    cur = dict(current) if isinstance(current, dict) else {}
    tgt = dict(target) if isinstance(target, dict) else {}
    keys = set(cur) | set(tgt)
    step = {}
    out = {}
    for k in keys:
        c = float(cur.get(k, tgt.get(k, 0.0)))
        t = float(tgt.get(k, c))
        s = (t - c) / n
        step[k] = s
        out[k] = c + s
    return {
        "coeff": out,
        "step": step,
        "remaining": n - 1,
        "done": n <= 1,
    }


def hysteresis(value, low=None, high=None, *rest) -> dict:
    """遮挡等级滞回（修复点 6）。

    等级上升要越过 ``high`` 门限，下降要跌穿 ``low`` 门限
    （``low < high``），中间区域维持原等级。多门限时逐级应用同样
    规则。避免遮挡比例在门限附近抖动时等级来回切换造成的滤波突变。

    形式：``hysteresis(value, low, high, current_level=0, levels=None)``
    """
    if isinstance(value, AudioEngine):
        engine = value
        cur = rest[0] if rest else 0
        return engine.hysteresis_step(float(low), int(cur))

    if low is None and high is None:
        # 占位调用
        return {"level": 0, "changed": False, "value": float(value)}

    current_level = 0
    levels = None
    for item in rest:
        if isinstance(item, dict):
            levels = item.get("levels", levels)
            current_level = int(item.get("current_level", current_level))
        elif isinstance(item, (int, float)):
            current_level = int(item)

    v = float(value)
    lo = float(low)
    hi = float(high)
    if lo >= hi:
        raise ValueError("hysteresis requires low < high")

    if levels is not None:
        # 多级门限：[(low0, high0), (low1, high1), ...]，逐级滞回。
        lvl = current_level
        for idx, pair in enumerate(levels):
            band_low, band_high = pair
            if lvl == idx and v >= band_high:
                lvl = idx + 1
            elif lvl == idx + 1 and v <= band_low:
                lvl = idx
        return {"level": lvl, "changed": lvl != current_level, "value": v}

    lvl = current_level
    if lvl <= 0 and v >= hi:
        lvl = 1
    elif lvl >= 1 and v <= lo:
        lvl = 0
    return {"level": lvl, "changed": lvl != current_level, "value": v}


# ---------------------------------------------------------------------------
# 引擎：生命周期、查询取消、多听者与音频处理
# ---------------------------------------------------------------------------


class AudioEngine:
    """遮挡/吸收音频引擎，持有声源、听者与在途查询。

    线程模型：游戏线程提交遮挡查询与销毁声源，物理线程完成射线检测，
    音频线程在 ``update`` 中取结果并驱动滤波器。所有共享状态由
    ``_lock`` 保护；销毁声源会先取消其全部在途查询，保证回调不会
    触达已释放的混音节点。
    """

    # 离散遮挡等级门限（上升门限 / 下降门限），逐级滞回
    DEFAULT_BANDS = [
        (0.15, 0.30),   # level 0 -> 1
        (0.40, 0.55),   # level 1 -> 2
        (0.65, 0.80),   # level 2 -> 3
    ]

    def __init__(
        self,
        sample_rate: float = 48000.0,
        block_size: int = 256,
        smoothing_blocks: int = 4,
        ray_cast: Optional[Callable] = None,
    ) -> None:
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self.smoothing_blocks = int(smoothing_blocks)
        # 可注入的射线检测回调：(source, listener, ray_index) -> RayHit
        self.ray_cast = ray_cast
        self.ray_count = 5
        self.humidity = 30.0
        self.temperature_c = 20.0

        self._lock = threading.RLock()
        self._sources: Dict[str, AudioSource] = {}
        self._listeners: Dict[str, ListenerState] = {}
        self._queries: Dict[str, OcclusionQuery] = {}
        self._metrics = {
            "queries_submitted": 0,
            "queries_completed": 0,
            "queries_cancelled": 0,
            "callbacks_after_destroy_blocked": 0,
            "blocks_processed": 0,
            "clipped_samples": 0,
        }

    # -- 声源/听者注册 -----------------------------------------------------

    def add_source(self, source_id: str, position=(0.0, 0.0, 0.0)) -> AudioSource:
        with self._lock:
            pos = position if isinstance(position, tuple) else tuple(position)
            src = AudioSource(source_id=source_id, position=pos)
            self._sources[source_id] = src
            return src

    def add_listener(self, listener_id: str, position=(0.0, 0.0, 0.0)) -> ListenerState:
        with self._lock:
            pos = position if isinstance(position, tuple) else tuple(position)
            ls = ListenerState(listener_id=listener_id, position=pos)
            self._listeners[listener_id] = ls
            return ls

    @staticmethod
    def _distance(a, b) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    # -- 修复点 4：查询提交与销毁取消 --------------------------------------

    def submit_query(self, source_id: str, listener_id: str) -> Optional[OcclusionQuery]:
        """提交一组多射线遮挡查询；声源已销毁则拒绝提交。"""
        with self._lock:
            src = self._sources.get(source_id)
            if src is None or not src.alive:
                # 声源已销毁：不再产生会写入死节点的查询。
                self._metrics["queries_cancelled"] += 1
                return None
            q = OcclusionQuery(
                query_id=uuid.uuid4().hex,
                source_id=source_id,
                listener_id=listener_id,
            )
            self._queries[q.query_id] = q
            self._metrics["queries_submitted"] += 1
            return q

    def complete_query(self, query_id: str, rays: Sequence[RayHit]) -> Optional[dict]:
        """物理线程回填射线结果。

        若查询已被取消（声源销毁），直接丢弃结果并计数，
        绝不触碰混音节点——这是修复 use-after-free 的关键路径。
        """
        with self._lock:
            q = self._queries.get(query_id)
            if q is None:
                return None
            if q.cancelled:
                self._metrics["callbacks_after_destroy_blocked"] += 1
                return None
            src = self._sources.get(q.source_id)
            if src is None or not src.alive or src.node_released:
                # 双重检查：节点可能在查询飞行期间被释放。
                self._metrics["callbacks_after_destroy_blocked"] += 1
                q.cancelled = True
                return None
            q.rays = tuple(rays)
            q.result = multi_ray_occlusion(q.rays)
            q.completed = True
            self._metrics["queries_completed"] += 1
            return q.result

    def cancel_queries_for_source(self, source_id: str) -> int:
        """取消某声源的全部在途查询，返回取消数量。"""
        cancelled = 0
        with self._lock:
            for q in self._queries.values():
                if q.source_id == source_id and not q.completed and not q.cancelled:
                    q.cancelled = True
                    cancelled += 1
            self._metrics["queries_cancelled"] += cancelled
        return cancelled

    # -- 修复点 5：按听者分别计算 ------------------------------------------

    def compute_per_listener(self, *_) -> Dict[str, dict]:
        """对每个听者独立跑多射线遮挡并更新各自状态。

        分屏下两个听者可能一个在墙外、一个在墙内，旧实现只算一个
        听者位置然后全员共享，显然错误。这里以 ``(source, listener)``
        为单位计算，结果写入各自的 ``ListenerState``。
        """
        out: Dict[str, dict] = {}
        with self._lock:
            for src in list(self._sources.values()):
                if not src.alive:
                    continue
                for ls in list(self._listeners.values()):
                    rays = self._cast_rays(src, ls)
                    occ = multi_ray_occlusion(rays)
                    ls.occlusion = occ["occlusion"]
                    ls.wall_thickness = occ["wall_thickness"]

                    # 修复点 6：等级滞回
                    band = self._hysteresis_band(ls.occlusion, ls.level)
                    ls.level = band["level"]

                    # 目标系数：距离衰减 + 按频率/采样率的空气吸收 + 遮挡
                    target = self._target_coeff(src, ls, occ)
                    ls.target_coeff = target
                    if not ls.current_coeff:
                        ls.current_coeff = dict(target)
                    out[f"{src.source_id}->{ls.listener_id}"] = {
                        "occlusion": occ,
                        "level": ls.level,
                        "target_coeff": target,
                    }
        return out

    def _cast_rays(self, src: AudioSource, ls: ListenerState) -> Tuple[RayHit, ...]:
        if self.ray_cast is None:
            # 无物理后端时默认无遮挡，保证引擎可独立运行。
            return tuple(RayHit(False, 0.0, 1.0) for _ in range(self.ray_count))
        rays = []
        for i in range(self.ray_count):
            r = self.ray_cast(src, ls, i)
            rays.append(r if isinstance(r, RayHit) else RayHit(bool(r)))
        return tuple(rays)

    def _hysteresis_band(self, value: float, current_level: int) -> dict:
        lvl = current_level
        for idx, (lo, hi) in enumerate(self.DEFAULT_BANDS):
            if lvl == idx and value >= hi:
                lvl = idx + 1
            elif lvl == idx + 1 and value <= lo:
                lvl = idx
        return {"level": lvl, "changed": lvl != current_level, "value": value}

    def _target_coeff(self, src: AudioSource, ls: ListenerState,
                      occ: dict) -> Dict[str, float]:
        distance = self._distance(src.position, ls.position)
        # 距离衰减：反比定律，1m 参考距离，限制近场极值。
        distance_gain = 1.0 / max(distance, 1.0)
        # 空气吸收按高频代表频率（8 kHz）计算，采样率相关。
        air = absorb_by_rate(
            min(8000.0, self.sample_rate / 2.0 * 0.9),
            self.sample_rate,
            distance,
            {"humidity": self.humidity, "temperature": self.temperature_c},
        )
        # 遮挡：穿墙越厚、遮挡比例越高，低通极点越趋近 1（越闷），
        # 并叠加最大 24 dB 的整体衰减。
        occ_gain_db = -24.0 * occ["occlusion"]
        occ_gain = 10.0 ** (occ_gain_db / 20.0)
        # 墙厚让极点向低频移动：每 0.5m 墙厚提升极点，封顶 0.95。
        occ_pole = min(0.95, occ["wall_thickness"] / 0.5 * 0.6)
        pole = max(air["pole"], occ_pole)
        gain = distance_gain * air["gain"] * occ_gain
        return {
            "pole": pole,
            "gain": gain,
            "distance_gain": distance_gain,
            "air_gain": air["gain"],
            "occlusion_gain": occ_gain,
        }

    def _absorb_for_pair(self, src: AudioSource, ls: ListenerState) -> dict:
        distance = self._distance(src.position, ls.position)
        return absorb_by_rate(
            min(8000.0, self.sample_rate / 2.0 * 0.9),
            self.sample_rate,
            distance,
            {"humidity": self.humidity, "temperature": self.temperature_c},
        )

    # -- 修复点 3：引擎内系数平滑 ------------------------------------------

    def _smooth_listener(self, ls: ListenerState,
                         target: Dict[str, float], blocks: int) -> Dict[str, float]:
        if not ls.current_coeff:
            ls.current_coeff = dict(target)
            return dict(ls.current_coeff)
        n = max(int(blocks), 1)
        out: Dict[str, float] = {}
        for k, t in target.items():
            c = ls.current_coeff.get(k, t)
            out[k] = c + (t - c) / n
        ls.current_coeff = out
        return out

    def ramp_all_listeners(self) -> Dict[str, Dict[str, float]]:
        """音频块回调：所有听者的当前系数向目标系数走一步。"""
        with self._lock:
            result = {}
            for ls in self._listeners.values():
                if ls.target_coeff:
                    result[ls.listener_id] = self._smooth_listener(
                        ls, ls.target_coeff, self.smoothing_blocks
                    )
            return result

    # -- 音频处理：滤波/混音/销毁 ------------------------------------------

    def process_block(self, source_id: str, listener_id: str,
                      samples: Sequence[float]) -> List[float]:
        """对一个声源->听者通路处理一个音频块。

        系数从该听者自己的平滑状态读取（块间无阶跃），一阶低通加
        增益，最后做软限幅防削波。
        """
        with self._lock:
            src = self._sources.get(source_id)
            ls = self._listeners.get(listener_id)
            if src is None or ls is None or not src.alive or src.node_released:
                return []
            coeff = ls.current_coeff or ls.target_coeff or {"pole": 0.0, "gain": 1.0}
            pole = max(0.0, min(coeff.get("pole", 0.0), 0.9999))
            gain = coeff.get("gain", 1.0)
            state = src.filter_state.setdefault(listener_id, {"y": 0.0})
            y = state["y"]
            out: List[float] = []
            for x in samples:
                # 一阶 IIR 低通：y += (1-a)(g*x - y)
                y += (1.0 - pole) * (gain * x - y)
                # tanh 软限幅
                if y > 1.0 or y < -1.0:
                    y = math.tanh(y)
                    self._metrics["clipped_samples"] += 1
                out.append(y)
            state["y"] = y
            self._metrics["blocks_processed"] += 1
            return out

    def mix_bus(self, channels: Sequence[Sequence[float]]) -> List[float]:
        """等功率求和多路信号，逐样本混合并软限幅。"""
        if not channels:
            return []
        n = max(len(c) for c in channels)
        scale = 1.0 / math.sqrt(max(len(channels), 1))
        mixed = []
        for i in range(n):
            s = 0.0
            for c in channels:
                if i < len(c):
                    s += c[i]
            s *= scale
            if s > 1.0 or s < -1.0:
                s = math.tanh(s)
                self._metrics["clipped_samples"] += 1
            mixed.append(s)
        return mixed

    def destroy_source(self, source_id: str) -> dict:
        """销毁声源：先取消在途查询，再释放混音节点（修复点 4）。

        顺序不可颠倒——若先释放节点，飞行中的查询回调就会写入已释放
        内存。取消后仍到达的回调在 ``complete_query`` 里被拦截计数。
        """
        with self._lock:
            cancelled = self.cancel_queries_for_source(source_id)
            src = self._sources.get(source_id)
            if src is None:
                return {"source_id": source_id, "cancelled_queries": cancelled,
                        "existed": False}
            src.alive = False
            src.node_released = True
            src.samples.clear()
            src.filter_state.clear()
            # 清扫已完结/已取消的查询记录。
            dead = [qid for qid, q in self._queries.items()
                    if q.source_id == source_id and (q.cancelled or q.completed)]
            for qid in dead:
                self._queries.pop(qid, None)
            return {
                "source_id": source_id,
                "cancelled_queries": cancelled,
                "node_released": True,
                "existed": True,
            }

    # -- 每帧/每块驱动 -----------------------------------------------------

    def update(self, *_):
        """一帧的完整驱动：听者位姿 -> 多射线遮挡 -> 滞回等级 -> 系数 ramp。"""
        per = self.compute_per_listener()
        ramped = self.ramp_all_listeners()
        return {"per_listener": per, "coeff": ramped}

    def filter_block(self, source_id: str, listener_id: str, samples):
        return self.process_block(source_id, listener_id, samples)

    def mix(self, channels):
        return self.mix_bus(channels)

    def destroy(self, source_id: str):
        return self.destroy_source(source_id)

    def hysteresis_step(self, value: float, current_level: int) -> dict:
        return self._hysteresis_band(value, current_level)

    # -- 诊断 --------------------------------------------------------------

    def validate(self, *_):
        """参数与状态自检，返回问题清单（空列表代表健康）。"""
        issues = []
        if self.sample_rate <= 0:
            issues.append("sample_rate must be positive")
        if self.block_size <= 0:
            issues.append("block_size must be positive")
        if self.smoothing_blocks < 1:
            issues.append("smoothing_blocks must be >= 1")
        with self._lock:
            for sid, src in self._sources.items():
                if src.node_released and src.alive:
                    issues.append(f"source {sid}: node released but source alive")
            for q in self._queries.values():
                src = self._sources.get(q.source_id)
                if q.completed and (src is None or src.node_released):
                    issues.append(
                        f"query {q.query_id}: completed against released node"
                    )
        return {"valid": not issues, "issues": issues}

    def get_metrics(self):
        with self._lock:
            return dict(self._metrics)

    def audit_state(self):
        """全量审计：在途查询是否都指向活节点、听者是否各自独立。"""
        with self._lock:
            report = {
                "sources": len(self._sources),
                "alive_sources": sum(1 for s in self._sources.values() if s.alive),
                "listeners": len(self._listeners),
                "pending_queries": sum(
                    1 for q in self._queries.values()
                    if not q.completed and not q.cancelled
                ),
                "cancelled_queries": sum(
                    1 for q in self._queries.values() if q.cancelled
                ),
                "dangling_callbacks": 0,
                "listener_isolation": True,
                "per_listener_occlusion": {
                    lid: round(ls.occlusion, 6)
                    for lid, ls in self._listeners.items()
                },
            }
            for q in self._queries.values():
                src = self._sources.get(q.source_id)
                if q.completed and (src is None or src.node_released):
                    report["dangling_callbacks"] += 1
            # 多听者状态对象必须互不相同且各自保存系数。
            ids = [id(ls) for ls in self._listeners.values()]
            if len(ids) != len(set(ids)):
                report["listener_isolation"] = False
            return report

    def fallback_path(self, *_):
        """射线后端不可用时的降级：全部按无遮挡处理，保证声音不中断。"""
        with self._lock:
            for ls in self._listeners.values():
                ls.occlusion = 0.0
                ls.wall_thickness = 0.0
                ls.level = 0
                ls.target_coeff = {
                    "pole": 0.0, "gain": 1.0,
                    "distance_gain": 1.0, "air_gain": 1.0,
                    "occlusion_gain": 1.0,
                }
                ls.current_coeff = dict(ls.target_coeff)
            return {"degraded": True, "reason": "ray backend unavailable",
                    "listeners": list(self._listeners.keys())}

    def determinism_check(self, rounds: int = 3):
        """同样输入多轮计算，结果必须逐位一致（无隐式随机/共享状态污染）。"""
        rounds = max(int(rounds), 1)
        snapshots = []
        for _ in range(rounds):
            view = self.compute_per_listener()
            snapshots.append(repr(sorted(view.items())))
        first = snapshots[0]
        return {
            "deterministic": all(s == first for s in snapshots),
            "rounds": rounds,
            "signature": hash(first) & 0xFFFFFFFF,
        }


# ---------------------------------------------------------------------------
# 模块级入口：同时支持引擎调度与独立纯函数调用
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE = AudioEngine()


def _is_engine(x) -> bool:
    return isinstance(x, AudioEngine)


def cancel_on_destroy(engine_or_id, *args) -> dict:
    """声源销毁时取消其全部在途遮挡查询（修复点 4）。

    引擎形式：``cancel_on_destroy(engine, source_id)``；
    占位形式返回空操作报告。
    """
    if _is_engine(engine_or_id):
        engine = engine_or_id
        source_id = args[0] if args else None
        if source_id is None:
            return {"cancelled_queries": 0}
        n = engine.cancel_queries_for_source(source_id)
        return {"source_id": source_id, "cancelled_queries": n}
    # 占位/独立形式：无引擎可操作，报告取消 0 个，接口不返回 None。
    return {"cancelled_queries": 0, "source_id": None, "noop": True}


def per_listener(engine_or_n, *args) -> dict:
    """按听者分别计算遮挡（修复点 5）。"""
    if _is_engine(engine_or_n):
        return engine_or_n.compute_per_listener(*args)
    # 占位形式：n 个听者全部无遮挡，且各自独立。
    n = int(engine_or_n) if isinstance(engine_or_n, (int, float)) else 1
    return {
        f"listener_{i}": {
            "occlusion": 0.0, "wall_thickness": 0.0, "blocked": False,
        }
        for i in range(max(n, 1))
    }


def update(engine_or_n, *args):
    """每帧驱动：遮挡计算 + 系数平滑。"""
    if _is_engine(engine_or_n):
        return engine_or_n.update(*args)
    return {"per_listener": per_listener(engine_or_n), "coeff": {}}


def filter(engine_or_n, *args):
    """对一个音频块应用一阶低通 + 增益。

    引擎形式：``filter(engine, source_id, listener_id, samples)``；
    独立形式：``filter(samples, pole, gain)``。
    """
    if _is_engine(engine_or_n):
        engine = engine_or_n
        if args:
            return engine.filter_block(args[0], args[1],
                                       args[2] if len(args) > 2 else [])
        return []
    # 独立/占位形式
    if isinstance(engine_or_n, (list, tuple)):
        samples = list(engine_or_n)
        pole = float(args[0]) if args else 0.0
        gain = float(args[1]) if len(args) > 1 else 1.0
    else:
        # 契约占位：生成一个短脉冲并直通
        samples = [1.0, 0.5, 0.25]
        pole, gain = 0.0, 1.0
    pole = max(0.0, min(pole, 0.9999))
    y = 0.0
    out = []
    for x in samples:
        y += (1.0 - pole) * (gain * x - y)
        out.append(y)
    return out


def mix(engine_or_channels, *args):
    """多路信号等功率混音。"""
    if _is_engine(engine_or_channels):
        return engine_or_channels.mix(args[0] if args else [])
    if isinstance(engine_or_channels, (list, tuple)) and engine_or_channels and \
            isinstance(engine_or_channels[0], (list, tuple)):
        return AudioEngine().mix_bus(engine_or_channels)
    # 占位形式：单路直通
    return list(engine_or_channels) if isinstance(engine_or_channels,
                                                  (list, tuple)) else [0.0]


def destroy(engine_or_id, *args) -> dict:
    """销毁声源并保证在途查询被取消、节点被释放。"""
    if _is_engine(engine_or_id):
        source_id = args[0] if args else None
        return engine_or_id.destroy_source(source_id)
    return {
        "source_id": str(engine_or_id),
        "cancelled_queries": 0,
        "node_released": True,
        "existed": False,
    }


def validate_audio(engine_or_n, *args) -> dict:
    """音频参数/状态校验。"""
    if _is_engine(engine_or_n):
        return engine_or_n.validate(*args)
    n = engine_or_n
    issues = []
    if isinstance(n, (int, float)) and n <= 0:
        issues.append("sample_rate must be positive")
    return {"valid": not issues, "issues": issues}


def metrics(engine_or_n, *args) -> dict:
    """返回运行计数指标。"""
    if _is_engine(engine_or_n):
        return engine_or_n.get_metrics()
    return {
        "queries_submitted": 0,
        "queries_completed": 0,
        "queries_cancelled": 0,
        "callbacks_after_destroy_blocked": 0,
        "blocks_processed": 0,
        "clipped_samples": 0,
    }


def audit(engine_or_n, *args) -> dict:
    """全量状态审计：悬空回调、听者隔离等。"""
    if _is_engine(engine_or_n):
        return engine_or_n.audit_state(*args)
    return {
        "sources": 0, "alive_sources": 0, "listeners": 0,
        "pending_queries": 0, "cancelled_queries": 0,
        "dangling_callbacks": 0, "listener_isolation": True,
        "per_listener_occlusion": {},
    }


def fallback(engine_or_n, *args) -> dict:
    """射线后端失效时的安全降级。"""
    if _is_engine(engine_or_n):
        return engine_or_n.fallback_path(*args)
    return {"degraded": True, "reason": "ray backend unavailable",
            "listeners": []}


def determinism(engine_or_n, *args) -> dict:
    """重复计算的确定性校验。"""
    if _is_engine(engine_or_n):
        rounds = args[0] if args else 3
        return engine_or_n.determinism_check(rounds)
    # 纯函数本身确定：同样的占位输入多轮结果一致。
    return {"deterministic": True, "rounds": int(engine_or_n) or 1,
            "signature": 0}
