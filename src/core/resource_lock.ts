import { existsSync, mkdirSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { randomUUID } from "node:crypto";

/**
 * Small cross-process lock for the JSON resources shared by agent workers.
 * mkdir is atomic on the filesystems supported by Bun/Node, so workers cannot
 * enter the same critical section at once.  The stale timeout only applies to
 * a lock whose owner stopped without cleanup.
 */
const heldLocks = new Set<string>();
const LOCK_WAIT_MS = 50;
const LOCK_MAX_WAIT_MS = 15_000;
const LOCK_STALE_MS = 120_000;

function sleepSync(milliseconds: number): void {
	const buffer = new SharedArrayBuffer(4);
	Atomics.wait(new Int32Array(buffer), 0, 0, milliseconds);
}

export function withFileLockSync<T>(targetPath: string, fn: () => T): T {
	const lockPath = `${targetPath}.lock`;
	if (heldLocks.has(lockPath)) return fn();

	mkdirSync(dirname(targetPath), { recursive: true });
	const started = Date.now();
	while (true) {
		try {
			mkdirSync(lockPath);
			break;
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
			try {
				if (Date.now() - statSync(lockPath).mtimeMs > LOCK_STALE_MS) {
					rmSync(lockPath, { recursive: true, force: true });
					continue;
				}
			} catch {
				// The owner may have released the lock between stat and removal.
			}
			if (Date.now() - started >= LOCK_MAX_WAIT_MS) {
				throw new Error(`Timed out waiting for resource lock: ${targetPath}`);
			}
			sleepSync(LOCK_WAIT_MS);
		}
	}

	heldLocks.add(lockPath);
	try {
		return fn();
	} finally {
		heldLocks.delete(lockPath);
		rmSync(lockPath, { recursive: true, force: true });
	}
}

/** Write a complete file and atomically replace the prior version. */
export function atomicWriteFileSync(path: string, content: string): void {
	mkdirSync(dirname(path), { recursive: true });
	const temporaryPath = `${path}.${process.pid}.${randomUUID()}.tmp`;
	try {
		writeFileSync(temporaryPath, content, "utf8");
		renameSync(temporaryPath, path);
	} finally {
		if (existsSync(temporaryPath)) rmSync(temporaryPath, { force: true });
	}
}
