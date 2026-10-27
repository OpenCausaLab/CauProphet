import { promises as fs } from "node:fs";
import path from "node:path";
import { $ } from "bun";
import {
	createCausalBeliefTool, createNewCausalBelief,
	createScratchpadTool, createNewScratchpad,
	listCausalBeliefsTool, listCausalBeliefs,
	listScratchpadsTool, listScratchpads,
	openCausalBeliefTool, openCausalBelief,
	openScratchpadTool, openScratchpad,
	exportBeliefAdjMatrixTool, exportBeliefAdjMatrix,
	exportScratchpadAdjMatrixTool, exportScratchpadAdjMatrix,
	writeCausalBeliefTool, writeCausalBelief,
	writeScratchpadTool, writeScratchpad,
} from "@/tools/causal_belief_tools";
import {
	causalDiagnoseTool, causalDiagnose,
	causalDiscoverTool, causalDiscover, type CausalToolRuntime,
	causalInferTool, causalInfer,
} from "@/tools/causal_tools";
import { causalScoutTool, causalScout } from "@/tools/causal_scout";
import {
	searchCausalCognitionMemory,
	searchCausalCognitionTool,
	updateCausalCognition,
	updateCausalCognitionTool,
} from "@/tools/causal_cognition_tools";

import {
	writeCognitionNoteTool, writeCognitionNote,
	instantiateCognitionNoteTool, instantiateCognitionNote,
} from "@/tools/cognition_note_tools";

// --- 1. 读取文件 (Read File) ---
export const readFileTool = {
	type: "function" as const,
	function: {
		name: "read_file",
		description: "读取指定路径的文件内容，支持行范围分页以防止因为文件过大被系统截断",
		parameters: {
			type: "object",
			properties: {
				filePath: { type: "string" },
				startLine: {
					type: "integer",
					description: "起始行（1-indexed），默认从 1 开始",
				},
				endLine: {
					type: "integer",
					description: "结束行（1-indexed），如果不提供则默认读取到文件末尾",
				},
			},
			required: ["filePath"],
		},
	},
};

export async function readFile(
	filePath: string,
	startLine?: number,
	endLine?: number,
): Promise<string> {
	try {
		const content = await fs.readFile(path.resolve(filePath), "utf-8");
		const lines = content.split(/\r?\n/);
		const totalLines = lines.length;

		const start = startLine !== undefined ? Math.max(1, startLine) : 1;
		const end =
			endLine !== undefined ? Math.min(totalLines, endLine) : totalLines;

		if (start > totalLines) {
			return `Error: startLine (${start}) is greater than the total number of lines (${totalLines}).`;
		}
		if (start > end) {
			return `Error: startLine (${start}) cannot be greater than endLine (${end}).`;
		}

		const sliced = lines.slice(start - 1, end);
		return (
			`[Showing lines ${start}-${end} of ${totalLines} total lines]\n` +
			sliced.join("\n")
		);
	} catch (_e) {
		return `Error reading file: ${String(_e)}`;
	}
}

// --- 2. 写入/覆盖文件 (Write File) ---
export const writeFileTool = {
	type: "function" as const,
	function: {
		name: "write_file",
		description: "将内容写入文件（如果文件存在则覆盖）",
		parameters: {
			type: "object",
			properties: {
				filePath: { type: "string" },
				content: { type: "string" },
			},
			required: ["filePath", "content"],
		},
	},
};

export async function writeFile(
	filePath: string,
	content: string,
): Promise<string> {
	try {
		// 自动创建不存在的父目录
		await fs.mkdir(path.dirname(filePath), { recursive: true });
		await fs.writeFile(path.resolve(filePath), content, "utf-8");
		return `Successfully written to ${filePath}`;
	} catch (_e) {
		return `Error writing file: ${String(_e)}`;
	}
}

// --- 3. 执行 Shell 命令 (Execute Command) ---
import { spawn } from "node:child_process";

/**
 * 返回给模型的内联输出上限（字符数）。
 * 超过此长度时输出头部，剩余部分写入 .causalclaw/tmp/ 溢出文件，
 * 并通过截断标记告知 agent 用 read_file 按需分页读取。
 */
const MAX_INLINE_OUTPUT = 10_000;

/** 内存缓冲上限（字符数），防止异常巨大的输出撑爆内存。保留头部。 */
const MAX_MEMORY_BUFFER = 500_000;

/** 溢出文件存放目录（相对于工作区根目录）。 */
const SPILL_DIR = ".causalclaw/tmp";

/** 溢出文件存活时间（毫秒），超过后由下一次 execute_command 调用自动清理。 */
const SPILL_TTL_MS = 30 * 60 * 1000; // 30 分钟

/**
 * 清理过期的溢出文件（mtime 超过 SPILL_TTL_MS）。
 * 失败静默忽略——清理不应影响命令执行。
 */
async function cleanupSpillFiles(): Promise<void> {
	try {
		const files = await fs.readdir(SPILL_DIR);
		const now = Date.now();
		for (const name of files) {
			const filePath = path.join(SPILL_DIR, name);
			try {
				const s = await fs.stat(filePath);
				if (now - s.mtimeMs > SPILL_TTL_MS) {
					await fs.unlink(filePath);
				}
			} catch {
				// 单个文件 stat/unlink 失败忽略
			}
		}
	} catch {
		// 目录不存在则跳过
	}
}

/**
 * 将 stdout/stderr 全量写入溢出文件，返回文件路径（供 agent 用 read_file 读取）。
 */
async function spillOutput(stdout: string, stderr: string): Promise<string> {
	const ts = Date.now();
	const rand = Math.random().toString(36).slice(2, 6);
	const filename = `cmd_${ts}_${rand}.txt`;
	const filePath = path.join(SPILL_DIR, filename);

	await fs.mkdir(SPILL_DIR, { recursive: true });

	let content = "";
	if (stdout) content += `=== STDOUT (${stdout.length} chars) ===\n${stdout}\n`;
	if (stderr) content += `=== STDERR (${stderr.length} chars) ===\n${stderr}\n`;

	await fs.writeFile(filePath, content, "utf-8");
	return filePath;
}

/**
 * 格式化单个输出流。
 * 不超限则全量返回；超限则返回头部 + 截断标记 + 溢出文件路径。
 */
function formatOutputChunk(label: string, content: string, spillPath: string | null): string {
	if (content.length === 0) return "";
	if (content.length <= MAX_INLINE_OUTPUT) {
		return `${label}:\n${content}\n`;
	}
	const omitted = content.length - MAX_INLINE_OUTPUT;
	const hint = spillPath ? `, full output at: ${spillPath}` : "";
	return `${label}:\n${content.slice(0, MAX_INLINE_OUTPUT)}\n... [${omitted} more chars truncated${hint}]\n`;
}

export const executeCommandTool = {
	type: "function" as const,
	function: {
		name: "execute_command",
		description: "在终端中执行 Bash/Shell 命令，可设置超时时间轮询，超时返回增量输出及存活状态",
		parameters: {
			type: "object",
			properties: {
				command: { type: "string" },
				timeout: {
					type: "string",
					description: "超时时间(ms)。超过此时间后会返回增量输出与进程存活状态(PID)，据此决定是继续等待还是运行 kill -9 终止它。默认为 \"10000\"(10秒)。"
				}
			},
			required: ["command", "timeout"],
		},
	},
};

export async function executeCommand(command: string, timeoutStr: string = "10000"): Promise<string> {
	const timeout = Number.parseInt(timeoutStr, 10) || 10000;

	// 每次执行命令前清理过期溢出文件
	cleanupSpillFiles();

	return new Promise((resolve) => {
		const child = spawn(command, { shell: true });
		let stdout = "";
		let stderr = "";
		let isResolved = false;

		child.stdout.on("data", (data) => {
			stdout += data.toString();
			if (stdout.length > MAX_MEMORY_BUFFER) stdout = stdout.slice(0, MAX_MEMORY_BUFFER);
		});

		child.stderr.on("data", (data) => {
			stderr += data.toString();
			if (stderr.length > MAX_MEMORY_BUFFER) stderr = stderr.slice(0, MAX_MEMORY_BUFFER);
		});

		child.on("error", (error) => {
			if (isResolved) return;
			isResolved = true;
			resolve(`Error starting command: ${error.message}`);
		});

		child.on("close", async (code) => {
			if (isResolved) return;
			isResolved = true;

			const totalLen = stdout.length + stderr.length;
			const spillPath = totalLen > MAX_INLINE_OUTPUT ? await spillOutput(stdout, stderr) : null;

			let result = `Exit Code: ${code}\n`;
			result += formatOutputChunk("STDOUT", stdout, spillPath);
			result += formatOutputChunk("STDERR", stderr, spillPath);
			if (spillPath) result += `[Full output saved to: ${spillPath} — use read_file with startLine/endLine to read it in pages]\n`;

			resolve(result || "Command executed successfully with no output.");
		});

		setTimeout(async () => {
			if (isResolved) return;
			isResolved = true;

			const totalLen = stdout.length + stderr.length;
			const spillPath = totalLen > MAX_INLINE_OUTPUT ? await spillOutput(stdout, stderr) : null;

			let result =
				`Timeout (${timeout}ms) reached. Process is STILL RUNNING in background.\n` +
				`PID: ${child.pid} (Use 'kill -9 ${child.pid}' to terminate it if it's stuck)\n`;
			result += formatOutputChunk("STDOUT so far", stdout, spillPath);
			result += formatOutputChunk("STDERR so far", stderr, spillPath);
			if (spillPath) result += `[Full output so far saved to: ${spillPath} — use read_file with startLine/endLine to read it in pages]\n`;

			resolve(result);
		}, timeout);
	});
}

// --- 4. 网页搜索 (Web Search) ---
// 已拆分至独立文件，支持 DuckDuckGo / Brave / 博查 等多种后端
import { webSearchTool, webSearch } from "@/tools/web_search_tools";
export { webSearchTool, webSearch };


// --- 5. 列出目录结构 (List Directory) ---
export const listDirectoryTool = {
	type: "function" as const,
	function: {
		name: "list_directory",
		description: "列出指定目录下的所有文件和文件夹",
		parameters: {
			type: "object",
			properties: {
				dirPath: { type: "string", description: "如 '.', './src'" },
			},
			required: ["dirPath"],
		},
	},
};

export async function listDirectory(dirPath: string): Promise<string> {
	try {
		const files = await fs.readdir(path.resolve(dirPath), {
			withFileTypes: true,
		});
		return files
			.map((f) => `${f.isDirectory() ? "[DIR]" : "[FILE]"} ${f.name}`)
			.join("\n");
	} catch (_e) {
		return `Error listing directory: ${String(_e)}`;
	}
}

// --- 6. 全局文件搜索 (Search Files / Grep) ---
export const searchFilesTool = {
	type: "function" as const,
	function: {
		name: "search_files",
		description: "在当前项目目录下递归搜索包含特定关键词的文件内容",
		parameters: {
			type: "object",
			properties: { keyword: { type: "string" } },
			required: ["keyword"],
		},
	},
};

async function hasGrep(): Promise<boolean> {
	try {
		await $`grep --version`.quiet();
		return true;
	} catch {
		return false;
	}
}

async function walkDir(dir: string, callback: (filePath: string) => Promise<void>) {
	const files = await fs.readdir(dir, { withFileTypes: true });
	for (const file of files) {
		const res = path.resolve(dir, file.name);
		if (file.isDirectory()) {
			if (file.name === "node_modules" || file.name === ".git" || file.name === ".venv") continue;
			await walkDir(res, callback);
		} else {
			await callback(res);
		}
	}
}

async function searchFilesTypeScript(keyword: string): Promise<string> {
	try {
		const results: string[] = [];
		const rootDir = path.resolve(".");
		const lowerKeyword = keyword.toLowerCase();

		await walkDir(rootDir, async (filePath) => {
			try {
				const content = await fs.readFile(filePath, "utf-8");
				if (content.toLowerCase().includes(lowerKeyword)) {
					const lines = content.split(/\r?\n/);
					lines.forEach((line, idx) => {
						if (line.toLowerCase().includes(lowerKeyword)) {
							const relPath = path.relative(rootDir, filePath);
							results.push(`./${relPath}:${idx + 1}:${line.trim()}`);
						}
					});
				}
			} catch {
				// Skip binary or unreadable files
			}
		});

		if (results.length === 0) return "No matches found.";
		const text = results.join("\n");
		return text.length > 2000 ? text.slice(0, 2000) + "\n... [truncated]" : text;
	} catch (e: any) {
		return `Error searching: ${e.message}`;
	}
}

export async function searchFiles(keyword: string): Promise<string> {
	const grepAvailable = await hasGrep();
	if (grepAvailable) {
		try {
			const { stdout } =
				await $`grep -rin ${keyword} . --exclude-dir=node_modules --exclude-dir=.git`
					.quiet()
					.nothrow();
			const text = stdout.toString();
			return text.length > 0 ? text.slice(0, 2000) : "No matches found.";
		} catch (_e) {
			return searchFilesTypeScript(keyword);
		}
	} else {
		return searchFilesTypeScript(keyword);
	}
}

// --- 统一导出 (Unified Exports) ---
export const tools = [
	readFileTool,
	writeFileTool,
	executeCommandTool,
	webSearchTool,
	listDirectoryTool,
	searchFilesTool,
	listCausalBeliefsTool,
	listScratchpadsTool,
	openCausalBeliefTool,
	openScratchpadTool,
	createCausalBeliefTool,
	createScratchpadTool,
	writeCausalBeliefTool,
	writeScratchpadTool,
	causalDiagnoseTool,
	causalDiscoverTool,
	causalInferTool,
	causalScoutTool,
	exportBeliefAdjMatrixTool,
	exportScratchpadAdjMatrixTool,
	searchCausalCognitionTool,
	updateCausalCognitionTool,
	writeCognitionNoteTool,
	instantiateCognitionNoteTool,
];

export const toolHandlers: Record<string, (...args: any[]) => Promise<string>> = {
	read_file: readFile,
	write_file: writeFile,
	execute_command: executeCommand,
	web_search: webSearch,
	list_directory: listDirectory,
	search_files: searchFiles,
	list_causal_beliefs: listCausalBeliefs,
	list_scratchpads: listScratchpads,
	open_causal_belief: (id: string) => openCausalBelief(id),
	open_scratchpad: (id: string) => openScratchpad(id),
	create_causal_belief: (title: string, objective: string, dataPath?: string) => createNewCausalBelief(title, objective, dataPath),
	create_scratchpad: (title: string, objective: string, dataPath?: string) => createNewScratchpad(title, objective, dataPath),
	write_causal_belief: (content: string) => writeCausalBelief(content),
	write_scratchpad: (content: string) => writeScratchpad(content),
	causal_diagnose: (dataPath: string, treatment?: string, outcome?: string) => causalDiagnose(dataPath, treatment, outcome),
	causal_discover: (method: string, dataPath: string, alpha?: string, indepTest?: string, lambda1?: string, wThreshold?: string, scoreFunc?: string, depth?: string, timeout?: string, runtime?: CausalToolRuntime) => causalDiscover(method, dataPath, alpha, indepTest, lambda1, wThreshold, scoreFunc, depth, timeout, runtime),
	causal_infer: (
		dataPath: string,
		treatment: string,
		outcome: string,
		commonCauses?: string,
		effectModifiers?: string,
		method?: string,
		treatmentValue?: string | number,
		controlValue?: string | number,
		estimand?: string,
		interventionType?: string,
		shiftDelta?: string | number,
		runtime?: CausalToolRuntime,
	) => causalInfer(
		dataPath,
		treatment,
		outcome,
		commonCauses,
		effectModifiers,
		method,
		treatmentValue,
		controlValue,
		estimand,
		interventionType,
		shiftDelta,
		runtime,
	),
	causal_scout: (input: Record<string, unknown>) => causalScout(input),
	export_belief_adj_matrix: (path: string) => exportBeliefAdjMatrix(path),
	export_scratchpad_adj_matrix: (path: string) => exportScratchpadAdjMatrix(path),
	search_causal_cognition: (input: Record<string, unknown>) =>
		searchCausalCognitionMemory(input),
	update_causal_cognition: (input: Record<string, unknown>) =>
		updateCausalCognition(input),
	write_cognition_note: (input: Record<string, unknown>) => writeCognitionNote(input),
	instantiate_cognition_note: (input: Record<string, unknown>) => instantiateCognitionNote(input),
};
