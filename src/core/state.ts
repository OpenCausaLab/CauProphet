import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { type Config, ConfigSchema } from "@/schemas/config.schema";

export class GlobalState {
	private static instance: GlobalState;
	public config!: Config;

	// Store config in user's home directory (~/.cauprophet or fallback to ~/.causalclaw)
	private configDir = (() => {
		if (process.env.CAUPROPHET_CONFIG_DIR) return process.env.CAUPROPHET_CONFIG_DIR;
		const newDir = join(homedir(), ".cauprophet");
		const legacyDir = join(homedir(), ".causalclaw");
		if (existsSync(join(legacyDir, "config.json")) && !existsSync(join(newDir, "config.json"))) {
			return legacyDir;
		}
		return newDir;
	})();
	private configPath = join(this.configDir, "config.json");

	private constructor() {
		this.loadConfig();
	}

	public static getInstance(): GlobalState {
		if (!GlobalState.instance) {
			GlobalState.instance = new GlobalState();
		}
		return GlobalState.instance;
	}

	private loadConfig() {
		if (existsSync(this.configPath)) {
			try {
				const raw = readFileSync(this.configPath, "utf-8");
				const parsed = JSON.parse(raw);
				this.config = ConfigSchema.parse(parsed);
			} catch (e: unknown) {
				if (e instanceof SyntaxError) {
					console.error(
						`Config parse error: invalid JSON in ${this.configPath}
` +
						`  ${e.message}
` +
						`  Check for trailing commas, missing quotes, or unescaped characters.`,
					);
				} else {
					// Zod validation error — show which fields are wrong
					console.error(
						`Config validation error in ${this.configPath}:`,
						String(e),
					);
				}
				console.error("Using defaults — run setup wizard to reconfigure.");
				this.config = ConfigSchema.parse({});
			}
		} else {
			this.config = ConfigSchema.parse({});
		}
	}

	public saveConfig() {
		if (!existsSync(this.configDir)) {
			mkdirSync(this.configDir, { recursive: true });
		}
		writeFileSync(
			this.configPath,
			JSON.stringify(this.config, null, 2),
			"utf-8",
		);
	}
}
