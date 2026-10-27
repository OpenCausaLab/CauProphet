#!/usr/bin/env bun
import { copyFileSync, existsSync, mkdirSync, readdirSync, renameSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { stdin as input, stdout as output } from "node:process";
import * as readline from "node:readline/promises";
import * as p from "@clack/prompts";
import { GlobalState } from "@/core/state";

// Import view strategies
import type { InteractionStrategy } from "@/views/types";
import { ReactInteractionView } from "@/views/ReactView";
import { ClackInteractionView } from "@/views/ClackView";

async function main() {
	const projectRoot = join(fileURLToPath(import.meta.url), "..", "..", "..");
	const state = GlobalState.getInstance();

	// ── 1. 配置向导（仅在首次或配置缺失时运行）────────────────────────────
	const needsOpenAiConfig = !state.config.openaiApiKey;
	if (needsOpenAiConfig || !state.config.workspace) {
		console.log("Configuration partially missing. Let's set it up.");
		const rl = readline.createInterface({ input, output });

		if (needsOpenAiConfig) {
			const baseUrl = await rl.question(
				"OpenAI Base URL [https://api.openai.com/v1]: ",
			);
			const apiKey = await rl.question("OpenAI API Key: ");
			const model = await rl.question("Model [gpt-4o]: ");

			if (!apiKey.trim()) {
				console.log("API Key required. Exiting.");
				process.exit(1);
			}

			state.config.openaiBaseUrl = baseUrl.trim() || "https://api.openai.com/v1";
			state.config.openaiApiKey = apiKey.trim();
			state.config.model = model.trim() || "gpt-4o";

			const ctxWin = await rl.question("Context window size in tokens [128000]: ");
			state.config.contextWindow = ctxWin.trim() ? Number(ctxWin.trim()) || 128_000 : 128_000;
		}

		if (!state.config.workspace) {
			const defaultPath = resolve(projectRoot, "..", "cauprophet-workspace");
			const workspace = await rl.question(
				`CauProphet Workspace path [${defaultPath}]: `,
			);
			state.config.workspace = resolve(workspace.trim() || defaultPath);
		}

		rl.close();
		if (state.config.workspace) mkdirSync(state.config.workspace, { recursive: true });
		state.saveConfig();
		console.log("Configuration updated and saved!\n");
	}

	// ── 2. 切换到工作区目录 ────────────────────────────────────────────────
	if (state.config.workspace) {
		try {
			process.chdir(state.config.workspace);
		} catch (error) {
			console.error(`Failed to enter workspace: ${state.config.workspace}. ${String(error)}`);
			process.exit(1);
		}
	}

	// ── 3. Workspace 初始化（复制模板文件）────────────────────────────────
	try {
		const templateAgentsMd = join(projectRoot, "templates", "AGENTS.md");
		const workspaceAgentsMd = join(process.cwd(), "AGENTS.md");
		if (existsSync(templateAgentsMd) && !existsSync(workspaceAgentsMd)) {
			copyFileSync(templateAgentsMd, workspaceAgentsMd);
		}

		// Copy skills/ from templates/ into workspace (preserve user edits with -n)
		const templateSkillsDir = join(projectRoot, "templates", "skills");
		const workspaceSkillsDir = join(process.cwd(), "skills");
		if (existsSync(templateSkillsDir)) {
			mkdirSync(workspaceSkillsDir, { recursive: true });
			const { $ } = await import("bun");
			// -n: never overwrite existing files (preserves user edits)
			await $`cp -rn ${templateSkillsDir}/. ${workspaceSkillsDir}/`.quiet();
		}

		// Migrate legacy flat skill files (skills/<name>.md → skills/<name>/SKILL.md)
		// This upgrades workspaces created before the directory-based skill format.
		if (existsSync(workspaceSkillsDir)) {
			for (const entry of readdirSync(workspaceSkillsDir, { withFileTypes: true })) {
				if (entry.isFile() && entry.name.endsWith(".md")) {
					const skillName = entry.name.replace(/\.md$/, "");
					const oldPath = join(workspaceSkillsDir, entry.name);
					const newDir  = join(workspaceSkillsDir, skillName);
					const newPath = join(newDir, "SKILL.md");
					try {
						mkdirSync(newDir, { recursive: true });
						renameSync(oldPath, newPath);
						console.log(`[CausalClaw] Migrated skill: skills/${entry.name} → skills/${skillName}/SKILL.md`);
					} catch (e) {
						console.warn(`[CausalClaw] Failed to migrate skill ${entry.name}: ${String(e)}`);
					}
				}
			}
		}
	} catch {
		// Non-fatal — agent will operate without workspace AGENTS.md / skills
	}

	// ── 4. Python 环境引导 ─────────────────────────────────────────────────
	await bootstrapPythonEnv(projectRoot);

	// ── 5. 模式选择（CLI 或 Web）──────────────────────────────────────────
	p.intro("🦞 CauProphet");

	const mode = await p.select({
		message: "How would you like to interact?",
		options: [
			{
				value: "cli",
				label: "Terminal (CLI)",
				hint: "Chat directly in this terminal using the Clack interface",
			},
			{
				value: "web",
				label: "Web UI",
				hint: "Start the browser interface (backend + frontend)",
			},
		],
	});

	if (p.isCancel(mode)) {
		p.cancel("Cancelled.");
		process.exit(0);
	}

	if (mode === "web") {
		// ── Web 模式：启动一键脚本 ────────────────────────────────────────
		p.log.info("Starting the web interface…");
		p.outro("Opening browser interface. Press Ctrl+C to stop.");

		const devScript = join(projectRoot, "server", "dev.ts");
		const { $ } = await import("bun");
		// 替换当前进程（exec-style）：直接运行 dev.ts，输出不做前缀处理
		process.exit(
			(await $`bun ${devScript}`.nothrow()).exitCode ?? 0,
		);
	}

	// ── CLI 模式 ───────────────────────────────────────────────────────────
	// Environment-variable override still works for automation (VIEW_MODE=clack)
	const viewMode = process.env.VIEW_MODE?.toLowerCase() || "clack";

	let strategy: InteractionStrategy;
	if (viewMode === "clack") {
		strategy = new ClackInteractionView();
	} else if (viewMode === "react") {
		strategy = new ReactInteractionView();
	} else {
		throw new Error(`Unknown view mode: ${viewMode}`);
	}

	await strategy.start();
}

/**
 * Bootstraps the Python environment for causal analysis tools.
 *
 * 1. Checks if `uv` is installed; if not, auto-installs it.
 * 2. Runs `uv sync --locked` in the workspace to install and check dependencies.
 * 3. Rechecks existing environments so interrupted installations can recover.
 */
async function bootstrapPythonEnv(projectRoot: string): Promise<void> {
	const { $ } = await import("bun");
	// process.cwd() is the workspace root at this point (chdir was called above)
	const workspaceDir = process.cwd();

	// 1. Sync pyproject.toml / uv.lock / scripts/ from the bundled python/ into the workspace root
	// We always sync scripts so the latest version is available; pyproject.toml and uv.lock are only
	// copied if they don't exist (to avoid overwriting user modifications).
	const sourcePythonDir = join(projectRoot, "python");
	if (existsSync(sourcePythonDir)) {
		try {
			// Copy only the runtime files the tools need. A blanket `cp -rn python/. workspace/`
			// used to drag everything along — including python/tests/ and the dev .venv — into the
			// agent's workspace. Tests must stay out: they are dev-only and would pollute the
			// workspace (and collide with the repo's own tests/ dir at the workspace root).
			for (const f of ["pyproject.toml", "uv.lock"]) {
				const src = join(sourcePythonDir, f);
				const dst = join(workspaceDir, f);
				if (existsSync(src) && !existsSync(dst)) {
					await $`cp ${src} ${dst}`.quiet();
				}
			}

			// Always overwrite scripts/ to ensure tool updates
			const sourceScripts = join(sourcePythonDir, "scripts");
			if (existsSync(sourceScripts)) {
				await $`cp -r ${sourceScripts} ${workspaceDir}/`.quiet();
			}
		} catch (e) {
			console.warn(`[CauProphet] Failed to sync Python environment to workspace: ${String(e)}`);
		}
	}

	console.log("[CauProphet] Checking Python environment in workspace root...");

	// Step 1: Check if uv is available
	try {
		await $`uv --version`.quiet();
	} catch {
		// uv not found — install it
		console.log("[CauProphet] Installing uv (Python package manager)...");
		try {
			await $`curl -LsSf https://astral.sh/uv/install.sh | sh`.quiet();
			// Add to PATH for this session
			const home = process.env["HOME"] ?? "";
			process.env["PATH"] = `${home}/.local/bin:${process.env["PATH"]}`;
		} catch (e) {
			console.warn(
				`[CauProphet] Failed to auto-install uv: ${String(e)}\n` +
				"Please install uv manually: https://docs.astral.sh/uv/getting-started/installation/\n" +
				"Causal analysis tools will not be available until uv is installed.",
			);
			return;
		}
	}

	// Step 2: Run uv sync to install Python + dependencies
	try {
		console.log("[CauProphet] Installing Python dependencies (this may take a minute on first run)...");
		await $`uv sync --locked`.quiet();
		console.log("[CauProphet] Python environment ready in workspace root.");
	} catch (e) {
		console.warn(
			`[CauProphet] Failed to sync Python dependencies: ${String(e)}\n` +
			"Causal analysis tools may not work. Try running manually:\n" +
			`  cd ${workspaceDir} && uv sync`,
		);
	}
}

main().catch(console.error);
