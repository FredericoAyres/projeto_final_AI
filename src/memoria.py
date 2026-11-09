"""Medição de memória no Windows (diagnóstico; não afeta o treino).

No Windows, o que derruba os processos é o COMMIT: a memória privada reservada, que tem de caber em
RAM + pagefile. Ele pode estourar mesmo com RAM física livre.
"""
import ctypes
import ctypes.wintypes as wt
import threading
import time

import psutil

GB, MB = 2 ** 30, 2 ** 20


class _PerfInfo(ctypes.Structure):
    _fields_ = [("cb", wt.DWORD), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", wt.DWORD), ("ProcessCount", wt.DWORD), ("ThreadCount", wt.DWORD)]


def commit_sistema():
    """(commit em uso, limite) do Windows, em bytes."""
    p = _PerfInfo()
    p.cb = ctypes.sizeof(p)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb)
    return p.CommitTotal * p.PageSize, p.CommitLimit * p.PageSize


def processos():
    """[(papel, pid, RAM física MB, commit privado MB)] deste processo e de todos os filhos."""
    eu = psutil.Process()
    out = []
    for papel, p in [("principal", eu)] + [("worker", c) for c in eu.children(recursive=True)]:
        try:
            m = p.memory_info()
            out.append((papel, p.pid, m.rss / MB, m.private / MB))
        except psutil.Error:
            pass
    return out


class Amostrador(threading.Thread):
    """Mede o commit (este processo + filhos) a cada 0,5 s e guarda o pico. `parar = True` encerra."""

    def __init__(self):
        super().__init__(daemon=True)
        self.pico_proc, self.pico_sis, self.parar = 0.0, 0.0, False

    def run(self):
        while not self.parar:
            self.pico_proc = max(self.pico_proc, sum(c for *_, c in processos()))
            self.pico_sis = max(self.pico_sis, commit_sistema()[0] / GB)
            time.sleep(0.5)
