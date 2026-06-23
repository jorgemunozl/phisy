"""
Toy example: async vs multithreading.

Key insight: both handle I/O-bound work, but differently:
- threading: OS switches between threads (preemptive)
- async: single thread, yields control during awaits (cooperative)
"""
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


def blocking_task(name: str, delay: float) -> str:
    """Simulates I/O (e.g. network request)."""
    time.sleep(delay)
    return f"{name} done"


# --- Multithreading ---
def run_threaded():
    from threading import Thread

    results = []

    def worker(name: str, delay: float):
        results.append(blocking_task(name, delay))

    threads = [
        Thread(target=worker, args=(f"T{i}", 0.5))
        for i in range(3)
    ]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - start
    print(f"Threading: {elapsed:.2f}s, {results}")


# --- Async ---
async def async_task(name: str, delay: float) -> str:
    """Non-blocking: yields control during sleep."""
    await asyncio.sleep(delay)
    return f"{name} done"


async def run_async():
    start = time.perf_counter()
    results = await asyncio.gather(
        async_task("A0", 0.5),
        async_task("A1", 0.5),
        async_task("A2", 0.5),
    )
    elapsed = time.perf_counter() - start
    print(f"Async:     {elapsed:.2f}s, {list(results)}")


# --- Async + thread pool (for blocking code) ---
async def run_async_with_blocking():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=3) as pool:
        start = time.perf_counter()
        tasks = [
            loop.run_in_executor(pool, blocking_task, f"E{i}", 0.5)
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
    print(f"Async+TP:  {elapsed:.2f}s, {list(results)}")


if __name__ == "__main__":
    print("3 tasks × 0.5s each (parallel/concurrent):\n")
    run_threaded()
    asyncio.run(run_async())
    asyncio.run(run_async_with_blocking())
