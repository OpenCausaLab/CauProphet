#!/usr/bin/env bun
/**
 * server/dev.ts — CausalClaw 一键启动脚本
 *
 * 自动检测运行模式：
 *
 *   生产模式（npm 全局安装，web/dist/ 已存在）
 *     → 只启动 Bun 后端（它自带静态文件服务）
 *     → 自动打开浏览器
 *
 *   开发模式（从源码运行，web/dist/ 不存在）
 *     → 同时启动 Bun 后端 + Vite 前端开发服务器
 *
 * 用法：
 *   bun run dev           通过 package.json scripts（开发者）
 *   causalclaw → Web UI   通过 CLI 入口（用户）
 */

import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

// scripts/dev.ts was at root/scripts/, server/dev.ts is at root/server/
// Both are one level below project root.
const projectRoot = join(fileURLToPath(import.meta.url), "..", "..");

// ── 颜色辅助 ──────────────────────────────────────────────────────────────
const cyan   = (s: string) => `\x1b[36m${s}\x1b[0m`;
const yellow = (s: string) => `\x1b[33m${s}\x1b[0m`;
const dim    = (s: string) => `\x1b[2m${s}\x1b[0m`;
const bold   = (s: string) => `\x1b[1m${s}\x1b[0m`;

// ── 检测模式 ───────────────────────────────────────────────────────────────
const webDist    = join(projectRoot, "web", "dist", "index.html");
const isProd     = existsSync(webDist);

// ── 启动后端 ──────────────────────────────────────────────────────────────
const serverEntry = join(projectRoot, "server", "index.ts");
const serverProc  = Bun.spawn(
	["bun", "run", serverEntry],
	{
		cwd:    projectRoot,
		stdout: "pipe",
		stderr: "pipe",
		env:    { ...process.env, NODE_ENV: isProd ? "production" : "development" },
	},
);

let webProc: ReturnType<typeof Bun.spawn> | null = null;

if (isProd) {
	// 生产模式：后端已经内置静态文件服务，只需打开浏览器
	console.log(`\n${bold("🦞 CauProphet")} ${dim("— web interface starting")}`);
	console.log(dim("  http://localhost:3001\n"));

	// 等待服务器启动（简单延迟）后打开浏览器
	setTimeout(async () => {
		try {
			const { $ } = await import("bun");
			// 跨平台打开浏览器
			const open = process.platform === "darwin" ? "open"
			           : process.platform === "win32"  ? "start"
			           : "xdg-open";
			await $`${open} http://localhost:3001`.quiet().nothrow();
		} catch { /* non-fatal */ }
	}, 1500);

} else {
	// 开发模式：同时启动 Vite dev server
	console.log(`\n${bold("🦞 CauProphet")} ${dim("— dev mode")}`);
	console.log(dim("  Backend  → http://localhost:3001"));
	console.log(dim("  Frontend → http://localhost:5173\n"));

	const webDir = join(projectRoot, "web");
	webProc = Bun.spawn(
		["bun", "run", "dev"],
		{
			cwd:    webDir,
			stdout: "pipe",
			stderr: "pipe",
			env:    process.env,
		},
	);
}

// ── 日志流转发 ─────────────────────────────────────────────────────────────
async function pipeLines(
	stream: ReadableStream<Uint8Array>,
	prefix: string,
): Promise<void> {
	const decoder = new TextDecoder();
	const reader  = stream.getReader();
	let   buf     = "";

	while (true) {
		const { value, done } = await reader.read();
		if (done) break;
		buf += decoder.decode(value, { stream: true });
		const lines = buf.split("\n");
		buf = lines.pop() ?? "";
		for (const line of lines) {
			if (line.trim()) process.stdout.write(`${prefix} ${line}\n`);
		}
	}
	if (buf.trim()) process.stdout.write(`${prefix} ${buf}\n`);
}

pipeLines(serverProc.stdout, cyan("[server]"));
pipeLines(serverProc.stderr, cyan("[server]"));
if (webProc) {
	const wp = webProc;
	pipeLines(wp.stdout as ReadableStream<Uint8Array>, yellow("[web]   "));
	pipeLines(wp.stderr as ReadableStream<Uint8Array>, yellow("[web]   "));
}

// ── 优雅退出 ──────────────────────────────────────────────────────────────
function cleanup() {
	console.log(`\n${dim("Shutting down...")}`);
	try { serverProc.kill(); } catch { /* already dead */ }
	try { webProc?.kill(); }  catch { /* already dead */ }
	process.exit(0);
}

process.on("SIGINT",  cleanup);
process.on("SIGTERM", cleanup);

// 等待进程退出
const exits: Promise<number>[] = [serverProc.exited];
if (webProc) exits.push(webProc.exited);
await Promise.all(exits);
cleanup();
