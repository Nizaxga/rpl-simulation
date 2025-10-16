# simulation.py
"""
SimPy RPL rank-attack simulator (paper-aligned) — full file.
Outputs: output.txt, topology.png, energy.png
"""

import os
import random
import math
import builtins
import matplotlib.pyplot as plt
import simpy
import numpy as np

# ---------------------------
# Cleanup / Logging
# ---------------------------
for fn in ("output.txt", "topology.png", "energy.png"):
    if os.path.exists(fn):
        os.remove(fn)


# def log(*args, **kwargs):
#     with open("output.txt", "a") as f:
#         builtins.print(*args, **kwargs, file=f)
#     builtins.print(*args, **kwargs)


# ---------------------------
# Config (paper values + options)
# ---------------------------
FAST_MODE = True
RANDOM_SEED = 300
random.seed(RANDOM_SEED)

AREA_W = 110.0
AREA_H = 110.0
TX_RANGE = 50.0
INTF_RANGE = 60.0
RECEP_MIN = 0.30
RECEP_MAX = 1.00
START_DELAY = 0.5

VOLTAGE = 3.0
TICKS_PER_SEC = 32768

DIO_INTERVAL = 10.0
DATA_INTERVAL = 30.0
TX_DURATION_DIO = 0.02
TX_DURATION_DATA = 0.001
RX_DURATION = 0.001
CPU_DURATION = 0.0005

SCENARIOS = {
    "rank_increase": {"normal": 12, "attackers": 1, "mitigation": 0, "runtime": 3600},
    "rank_decrease": {"normal": 8, "attackers": 1, "mitigation": 0, "runtime": 3600},
    "worst_parent": {"normal": 12, "attackers": 1, "mitigation": 0, "runtime": 1200},
    "mitigation": {"normal": 7, "attackers": 3, "mitigation": 3, "runtime": 4320},
    "none": {"normal": 9, "attackers": 0, "mitigation": 0, "runtime": 3600},
}
MODES = ["rank_increase", "rank_decrease", "worst_parent", "mitigation", "none" ]
# ATTACK_MODE = "none"
# scenario = SCENARIOS[ATTACK_MODE]


def reception_prob(distance):
    """Return probability packet is received given distance (30%-100% paper)."""
    if distance <= TX_RANGE:
        dist_factor = 1.0 - (distance / TX_RANGE) * (1.0 - RECEP_MIN)
        return max(0.0, min(1.0, dist_factor * random.uniform(RECEP_MIN, RECEP_MAX)))
    elif distance <= INTF_RANGE:
        frac = 1.0 - ((distance - TX_RANGE) / (INTF_RANGE - TX_RANGE))
        return max(0.0, RECEP_MIN * 0.5 * frac * random.uniform(RECEP_MIN, RECEP_MAX))
    return 0.0


# ---------------------------
# Node class
# ---------------------------
class Node:
    def __init__(self, env, nid, pos, role="normal"):
        self.env = env
        self.id = nid
        self.pos = pos
        self.role = role  # 'normal', 'attacker', 'mitigation', 'sink'
        self.neigh = []  # discovered neighbor nodes
        self.parent = None  # chosen parent (Node)
        self.advertised_rank = 1000  # numeric rank (lower = better)
        self.is_attacker = role == "attacker"
        self.attack_active = False

        # energy accounting (seconds)
        self.tx_time = 0.0
        self.rx_time = 0.0
        self.cpu_time = 0.0
        self.lpm_time = 0.0
        self.energy_mJ = 0.0

        # traffic counters
        self.pkts_sent = 0
        self.pkts_forwarded = 0
        self.pkts_recvd = 0

    def distance(self, other):
        return math.hypot(self.pos[0] - other.pos[0], self.pos[1] - other.pos[1])

    # DIO process: advertise rank to neighbors
    def dio_process(self, ATTACK_MODE):
        yield self.env.timeout(START_DELAY)
        while True:
            adv = self.advertised_rank
            if self.is_attacker and self.attack_active:
                if ATTACK_MODE == "rank_increase":
                    adv = 1  # make attacker appear very attractive / low rank
                elif ATTACK_MODE == "rank_decrease":
                    adv = 0
                elif ATTACK_MODE == "worst_parent":
                    adv = self.advertised_rank

            # broadcast to neighbors: attempt delivery per neighbor
            for nb in self.neigh:
                d = self.distance(nb)
                p = reception_prob(d)
                # sender always attempts transmit -> count TX time
                self._tx(TX_DURATION_DIO)
                # receiver only charges RX/CPU if it actually receives the DIO
                if random.random() < p:
                    nb._rx(RX_DURATION)
                    nb._cpu(CPU_DURATION)
                    nb.evaluate_parent(candidate=self, candidate_rank=adv)
            # DIO interval: attackers may spam
            interval = DIO_INTERVAL
            if self.is_attacker and self.attack_active:
                if ATTACK_MODE == "rank_increase":
                    interval = max(0.5, DIO_INTERVAL / 6.0)
                elif ATTACK_MODE == "rank_decrease":
                    interval = max(0.5, DIO_INTERVAL / 4.0)
            yield self.env.timeout(interval)

    # Parent selection metric & switching
    def evaluate_parent(self, candidate, candidate_rank):
        if self.role == "sink":
            return

        # avoid loops
        cursor = candidate
        while cursor is not None:
            if cursor == self:
                return
            cursor = cursor.parent

        if (
            self.role == "mitigation"
            and candidate.is_attacker
            and candidate.attack_active
        ):
            return

        d = self.distance(candidate)
        if d > INTF_RANGE:
            return
        metric = candidate_rank + (d / TX_RANGE)
        cur_metric = (
            getattr(self.parent, "candidate_metric_cache", 1e9) if self.parent else 1e9
        )
        if metric + 1e-9 < cur_metric:
            self.parent = candidate
            self.parent.candidate_metric_cache = metric
            # if not FAST_MODE:
                # log(
                #     f"{self.env.now:.2f}s {self.id} selects parent {candidate.id} metric={metric:.3f}"
                # )

    # Data generation and hop-by-hop forwarding (aggregated simulation)
    def data_process(self, sink, ATTACK_MODE, pkt_interval=DATA_INTERVAL, ):
        yield self.env.timeout(START_DELAY + random.uniform(0, 1.0))
        while True:
            if self.role in ("normal", "mitigation"):
                self.pkts_sent += 1
                sender = self
                success = True
                hop = 0
                while sender is not None and sender != sink:
                    receiver = sender.parent
                    if receiver is None:
                        success = False
                        break
                    d = sender.distance(receiver)
                    p = reception_prob(d)
                    # attempt transmission: charge tx for sender
                    sender._tx(TX_DURATION_DATA)
                    # success check: if delivered, receiver charges rx/cpu and forwarding happens
                    if random.random() < p:
                        receiver._rx(RX_DURATION)
                        receiver._cpu(CPU_DURATION)
                        sender.pkts_forwarded += 1
                        receiver.pkts_recvd += 1
                        # attacker behaviors
                        if receiver.is_attacker and receiver.attack_active:
                            if ATTACK_MODE == "rank_decrease":
                                # drop with probability (isolation)
                                if random.random() < 0.7:
                                    success = False
                                    # if not FAST_MODE:
                                        # log(
                                        #     f"{self.env.now:.2f}s pkt dropped by attacker {receiver.id}"
                                        # )
                                    break
                            elif ATTACK_MODE == "worst_parent":
                                if receiver.neigh:
                                    worst = max(
                                        receiver.neigh,
                                        key=lambda n: receiver.distance(n),
                                    )
                                    receiver.parent = worst
                                    # if not FAST_MODE:
                                        # log(
                                        #     f"{self.env.now:.2f}s attacker {receiver.id} forwards to worst {worst.id}"
                                        # )
                            elif ATTACK_MODE == "rank_increase":
                                # attacker forwards normally but its rank manipulation causes suboptimal parent selection elsewhere
                                pass
                        # advance along parent chain
                        sender = receiver
                        hop += 1
                        if hop > 50:
                            success = False
                            break
                    else:
                        # no reception -> packet lost at this hop
                        success = False
                        break
                if sender == sink:
                    sink.pkts_recvd += 1
                    # if not FAST_MODE:
                        # log(f"{self.env.now:.2f}s pkt from {self.id} reached sink")
            yield self.env.timeout(pkt_interval + random.uniform(0, 0.2))

    # neighbor discovery: one-shot linking within interference range (probabilistic)
    def neighbor_discovery_once(self, all_nodes):
        for n in all_nodes:
            if n is self:
                continue
            d = self.distance(n)
            if d <= INTF_RANGE and random.random() < 0.95:
                self.neigh.append(n)

    # Energy helpers
    def _tx(self, secs):
        self.tx_time += secs

    def _rx(self, secs):
        self.rx_time += secs

    def _cpu(self, secs):
        self.cpu_time += secs

    def finalize_energy(self, runtime):
        active = self.tx_time + self.rx_time + self.cpu_time
        lpm = max(0.0, runtime - active)
        self.lpm_time = lpm
        tx_ticks = int(round(self.tx_time * TICKS_PER_SEC))
        rx_ticks = int(round(self.rx_time * TICKS_PER_SEC))
        cpu_ticks = int(round(self.cpu_time * TICKS_PER_SEC))
        lpm_ticks = int(round(self.lpm_time * TICKS_PER_SEC))
        current_term = (
            (tx_ticks * 19.5)
            + (rx_ticks * 21.5)
            + (cpu_ticks * 1.8)
            + (lpm_ticks * 0.0545)
        )
        energy_mJ = (current_term * VOLTAGE) / TICKS_PER_SEC
        self.energy_mJ = energy_mJ
        return energy_mJ


# ---------------------------
# Build network and bootstrap
# ---------------------------
def build_network(env, scenario, ATTACK_MODE):
    nodes = []
    idx = 0

    # sink in center
    sink = Node(
        env,
        f"N{idx}",
        (random.uniform(0, AREA_W), random.uniform(0, AREA_H)),
        role="sink",
    )
    sink.advertised_rank = 0
    nodes.append(sink)
    idx += 1

    # normal nodes
    for _ in range(scenario["normal"]):
        pos = (random.uniform(0, AREA_W), random.uniform(0, AREA_H))
        n = Node(env, f"N{idx}", pos, role="normal")
        n.advertised_rank = random.uniform(5, 20)
        nodes.append(n)
        idx += 1

    # attackers
    for _ in range(scenario["attackers"]):
        pos = (random.uniform(0, AREA_W), random.uniform(0, AREA_H))
        a = Node(env, f"N{idx}", pos, role="attacker")
        a.advertised_rank = random.uniform(1, 50)
        nodes.append(a)
        idx += 1

    # mitigation nodes
    for _ in range(scenario.get("mitigation", 0)):
        pos = (random.uniform(0, AREA_W), random.uniform(0, AREA_H))
        m = Node(env, f"N{idx}", pos, role="mitigation")
        m.advertised_rank = random.uniform(5, 20)
        nodes.append(m)
        idx += 1

    # neighbor discovery (one-shot)
    for n in nodes:
        n.neighbor_discovery_once(nodes)

    # bootstrap parent choices: evaluate once so parents exist before processes start
    for n in nodes:
        for nb in n.neigh:
            n.evaluate_parent(candidate=nb, candidate_rank=nb.advertised_rank)

    # start DIO and data processes
    for n in nodes:
        env.process(n.dio_process(ATTACK_MODE))
        env.process(n.data_process(sink, ATTACK_MODE, pkt_interval=DATA_INTERVAL))
    return nodes, sink


# Attack controller: enable attacker behavior (adjust advertised_rank)
def attack_controller(env, nodes, ATTACK_MODE):
    attackers = [n for n in nodes if n.is_attacker]
    if not attackers:
        return
    yield env.timeout(10.0)  # warmup
    for a in attackers:
        a.attack_active = True
        if ATTACK_MODE in ("rank_increase", "rank_decrease"):
            a.advertised_rank = 0
        # log(f"{env.now:.2f}s attacker {a.id} ACTIVATED (mode={ATTACK_MODE})")
    # attacks remain active until simulation end
    return


# ---------------------------
# Run simulation
# ---------------------------
def run():
    EPOCHS = 1000
    os.makedirs("output", exist_ok=True)
    results = {}
    # run all
    for ATTACK_MODE in MODES:
        pdr_list = []
        plr_list = []
        energy_list = []
        for _ in range(EPOCHS):
            env = simpy.Environment()
            scenario = SCENARIOS[ATTACK_MODE]
            nodes, sink = build_network(env, scenario, ATTACK_MODE)
            runtime = scenario["runtime"]

            # start attack controller
            env.process(attack_controller(env, nodes, ATTACK_MODE))
            # log("Scenario:", ATTACK_MODE, "nodes:", len(nodes), "runtime(s):", runtime)
            env.run(until=runtime)
            # metrics + finalize energy
            total_energy = 0.0
            for n in nodes:
                e = n.finalize_energy(runtime)
                total_energy += e
                # log(
                #     f"{n.id} role={n.role} TX={n.tx_time:.3f}s RX={n.rx_time:.3f}s CPU={n.cpu_time:.3f}s LPM={n.lpm_time:.3f}s ENERGY={e:.6f} mJ pkts_sent={n.pkts_sent} pkts_forwarded={n.pkts_forwarded} pkts_recvd={n.pkts_recvd}"
                # )
            avg_energy = total_energy / len(nodes) if nodes else 0.0
            total_pkts_sent = sum(n.pkts_sent for n in nodes)
            total_pkts_receive = sum(n.pkts_recvd for n in nodes)
            PKT_LOSS = total_pkts_receive - sink.pkts_recvd
            plr = (PKT_LOSS / total_pkts_sent * 100.0) if total_pkts_sent > 0 else 0.0
            # pdr = (sink.pkts_recvd / total_pkts_sent * 100.0) if total_pkts_sent > 0 else 0.0
            pdr = (total_pkts_receive / total_pkts_sent * 100.0) if total_pkts_sent > 0 else 0.0
            # log(
            #     f"SUM_ENERGY={total_energy:.6f} mJ AVG_ENERGY={avg_energy:.6f} mJ PDR={pdr:.2f}% PLR={plr:.2f}"
            # ) 
            pdr_list.append(pdr)
            plr_list.append(plr)
            energy_list.append(avg_energy)

            # quick totals sanity
            total_tx = sum(n.tx_time for n in nodes)
            total_rx = sum(n.rx_time for n in nodes)
            # log(
            #     f"TOTALS TX={total_tx:.3f}s RX={total_rx:.3f}s PKTS_SENT={total_pkts_sent} PKTS_RECV_AT_SINK={sink.pkts_recvd} PKTS_RECEIVED={total_pkts_receive}"
            # ) 

        pdr_mean = np.mean(pdr_list)
        pdr_std = np.std(pdr_list)
        plr_mean = np.mean(plr_list)
        plr_std = np.std(plr_list)
        energy_mean = np.mean(energy_list) 
        energy_std = np.std(energy_list)

        results[ATTACK_MODE] = {
            "PDR_mean": pdr_mean,
            "PDR_std": pdr_std,
            "PLR_mean": plr_mean,
            "PLR_std": plr_std,
            "Energy_mean": energy_mean,
            "Energy_std": energy_std,
        }

            # plots
            # plt.figure(figsize=(7, 7))
            # for n in nodes:
            #     if n.role == "sink":
            #         s = 160
            #         c = "green"
            #     elif n.role == "attacker":
            #         s = 120
            #         c = "red"
            #     elif n.role == "mitigation":
            #         s = 100
            #         c = "orange"
            #     else:
            #         s = 60
            #         c = "blue"
            #     plt.scatter(n.pos[0], n.pos[1], s=s, c=c)
            #     plt.text(n.pos[0], n.pos[1] - 1.5, n.id, fontsize=6, ha="center")
            #     for nb in n.neigh:
            #         plt.plot(
            #             [n.pos[0], nb.pos[0]],
            #             [n.pos[1], nb.pos[1]],
            #             color="0.85",
            #             linewidth=0.4,
            #         )
            # plt.xlim(0, AREA_W)
            # plt.ylim(0, AREA_H)
            # plt.gca().set_aspect("equal", adjustable="box")
            # plt.title(f"Topology: {ATTACK_MODE} (sink green, attacker red)")
            # plt.savefig(f"output/topology_{ATTACK_MODE}.png", dpi=300)
            # plt.close()
            # for n in nodes:
            #     children = [x.id for x in nodes if x.parent == n]
            #     if children:
            #         log(f"{n.id} has children: {children}")

            # energy bar
            # ids = [n.id for n in nodes]
            # energies = [n.energy_mJ for n in nodes]
            # order = sorted(range(len(ids)), key=lambda i: energies[i], reverse=True)
            # order = energies
            # plt.figure(figsize=(10, 4))
            # plt.bar(range(len(ids)), [energies[i] for i in order])
            # plt.xticks(range(len(ids)), [ids[i] for i in order], rotation=90, fontsize=6)
            # plt.ylabel("Energy (mJ)")
            # plt.title(f"Energy per node (PDR {pdr:.2f}%)")
            # plt.tight_layout()
            # plt.savefig(f"output/energy_{ATTACK_MODE}.png", dpi=300)
            # plt.close()

            # packeges sent and recevie per node
            # packages_s = [n.pkts_sent for n in nodes]
            # packages_r = [n.pkts_recvd for n in nodes]
            # x = range(len(ids))
            # plt.figure(figsize=(10, 4))
            # plt.plot(x, packages_s, marker='o', label="Sent", alpha=0.6)
            # plt.plot(x, packages_r, marker='s', label="Received", alpha=0.6)
            # plt.xticks(x, ids, rotation=90, fontsize=6)
            # plt.ylabel("Packet Count")
            # plt.title(f"Packets Sent vs Received per Node ({ATTACK_MODE})")
            # plt.legend()
            # plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            # plt.tight_layout()
            # plt.savefig(f"output/packets_{ATTACK_MODE}.png", dpi=300)
            # plt.close()
            # log("Saved topology.png, energy.png, packages.png")

            # space
            # log()
    for mode, stats in results.items():
        print(f"{mode:15s}  PDR={stats['PDR_mean']:.2f}±{stats['PDR_std']:.2f}\n"
            f"PLR={stats['PLR_mean']:.2f}±{stats['PLR_std']:.2f}\n"
            f"Energy={stats['Energy_mean']:.3f}±{stats['Energy_std']:.3f} mJ\n")


if __name__ == "__main__":
    run()
