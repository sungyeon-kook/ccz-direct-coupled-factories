"""Signed pi/8 rotations with the adaptive X/Y gadget, and an end-to-end gadget-level check of the reduced circuit.
Convention: rot(P, th) = exp(i th P). The reduced circuit needs U_v = rot(P_v, s_v pi/8), s_v = (-1)^|v|.
|A> = (|0> + e^{i pi/4}|1>)/sqrt2. Gadget: measure P (x) Z_A -> m; read A in basis X or Y -> b.
Rule tested: basis = X if m == -s else Y   (s = +1: rot(P,+pi/8); s = -1: rot(P,-pi/8)).
Byproduct: the branch implements U or P.U; P is Z-type, so it commutes with all later rotations and only
updates the Pauli frame (no later sign changes)."""
import itertools, numpy as np
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1, -1]).astype(complex); Y = 1j * X @ Z
def kron(*ops):
    out = np.array([[1]], complex)
    for o in ops: out = np.kron(out, o)
    return out
def rot(P, th): return np.cos(th) * np.eye(len(P)) + 1j * np.sin(th) * P
w = np.exp(1j * np.pi / 4); A = np.array([1, w]) / np.sqrt(2)
eig = {('X', +1): np.array([1, 1]) / np.sqrt(2), ('X', -1): np.array([1, -1]) / np.sqrt(2),
       ('Y', +1): np.array([1, 1j]) / np.sqrt(2), ('Y', -1): np.array([1, -1j]) / np.sqrt(2)}

def branch_map(P, m, basis, b):
    """data map K = <b|_A Pi_m (. (x) |A>), Pi_m projector of P (x) Z_A"""
    n = len(P); PZ = np.kron(P, Z); Pi = (np.eye(2 * n) + m * PZ) / 2
    K = np.zeros((n, n), complex)
    for i in range(n):
        e = np.zeros(n); e[i] = 1
        st = (Pi @ np.kron(e, A)).reshape(n, 2)
        K[:, i] = st @ eig[(basis, b)].conj()
    return K
def classify(K, U, P):
    """return 0 if K ~ U, 1 if K ~ P U, None otherwise (up to scale/phase)"""
    if np.linalg.norm(K) < 1e-12: return 'zero'
    Kn = K / np.linalg.norm(K) * np.sqrt(len(K))
    for f, F in ((0, np.eye(len(P))), (1, P)):
        if abs(abs(np.trace((F @ U).conj().T @ Kn)) / len(K) - 1) < 1e-9: return f
    return None

print("single gadget, P = Z (1 data qubit)")
rule_ok = True; frame_table = {}
for s in (+1, -1):
    U = rot(Z, s * np.pi / 8)
    for m in (+1, -1):
        basis = 'X' if m == -s else 'Y'
        for b in (+1, -1):
            f = classify(branch_map(Z, m, basis, b), U, Z)
            frame_table[(s, m, b)] = f
            rule_ok &= f in (0, 1)
            print(f"  s={s:+d} m={m:+d} basis={basis} b={b:+d} -> {'U' if f == 0 else 'P.U' if f == 1 else f}")
        wrong = 'Y' if basis == 'X' else 'X'
        f_w = [classify(branch_map(Z, m, wrong, b), U, Z) for b in (+1, -1)]
        print(f"     (wrong basis {wrong} would give {f_w}: not U up to frame)")
print("rule 'X iff m = -s' gives the target rotation up to Pauli frame in all 8 branches:", rule_ok)

# ---- end-to-end: 4 data qubits, 8 signed rotations, every gadget branch enumerated (4^8 branches) ----
plus = np.ones(2) / np.sqrt(2)
verts = list(itertools.product([0, 1], repeat=3))
def zstr(bits): return kron(*[Z if t else I2 for t in bits])
Ps = [zstr(v + (1,)) for v in verts]; S = [(-1) ** sum(v) for v in verts]
psi0 = kron(*[plus.reshape(2, 1)] * 4).ravel()
target = np.diag([1] * 7 + [-1]) @ kron(*[plus.reshape(2, 1)] * 3).ravel()
maps = {}
for k, (P, s) in enumerate(zip(Ps, S)):
    for m in (+1, -1):
        basis = 'X' if m == -s else 'Y'
        for b in (+1, -1):
            K = branch_map(P, m, basis, b)
            maps[(k, m, b)] = (K, frame_table[(s, m, b)])
worst_fid, tot_prob, acc_min = 1.0, 0.0, 1.0
for outcomes in itertools.product([(+1, +1), (+1, -1), (-1, +1), (-1, -1)], repeat=8):
    psi = psi0.copy(); frame = np.eye(16)
    for k, (m, b) in enumerate(outcomes):
        K, f = maps[(k, m, b)]; psi = K @ psi
        if f == 1: frame = Ps[k] @ frame
    pr = np.vdot(psi, psi).real
    if pr < 1e-14: continue
    tot_prob += pr
    psi = frame @ psi / np.sqrt(pr)                       # undo the recorded Z-type frame
    proj = kron(I2, I2, I2, (np.eye(2) + X) / 2); phi = proj @ psi
    acc = np.vdot(phi, phi).real; acc_min = min(acc_min, acc)
    red = phi.reshape(8, 2) @ np.array([1, 1]) / np.sqrt(2); red /= np.linalg.norm(red)
    worst_fid = min(worst_fid, abs(np.vdot(target, red)) ** 2)
print(f"end-to-end over all 4^8 branches: total probability {tot_prob:.6f}, min acceptance {acc_min:.6f}, min fidelity with CCZ|+++> {worst_fid:.6f}")
