import * as p from "@clack/prompts";
import pc from "picocolors";
import { runAgentLoop } from "@/core/orchestrator";
import type { InteractionStrategy } from "./types";

import { Marked } from "marked";
import TerminalRenderer from "marked-terminal";

export class ClackInteractionView implements InteractionStrategy {
	async start(): Promise<void> {
		// console.clear();
		p.intro(pc.bgGreen(pc.black(" CauProphet (Clack View) ")));

		const marked = new Marked();
		marked.setOptions({
			renderer: new TerminalRenderer({
				width: (process.stdout.columns || 80) - 10,
				reflowText: true,
			}) as any,
		});
		const s = p.spinner();
		let lastTool = "";

		while (true) {
			const userInput = await p.text({
				message: pc.cyan("─".repeat((process.stdout.columns || 80) - 5)),
				placeholder: "Ask CauProphet",
			});

			if (p.isCancel(userInput)) {
				p.outro(pc.gray("Goodbye!"));
				process.exit(0);
			}

			const text = (userInput as string).trim();
			if (!text) continue;

			if (text.startsWith("/")) {
				const cmd = text.slice(1).toLowerCase();
				if (cmd === "exit" || cmd === "quit") {
					p.outro(pc.gray("Goodbye!"));
					process.exit(0);
				} else if (cmd === "help") {
					p.note("/help - Show help\n/exit - Exit", "Available commands");
					continue;
				}
			}

			try {
				await runAgentLoop(text, (status) => {
					if (lastTool) {
						lastTool = lastTool.replace("Running tool", "Completed");
						s.stop(pc.gray(lastTool));
						lastTool = "";
					}
					if (status.startsWith("THINK:")) {
						const think = status.slice(6).trim();
						if (think) {
							p.log.message(pc.gray(think));
						}
					} else if (status.startsWith("CONTENT:")) {
						const content = status.slice(8).trim();
						if (content) {
							p.log.success((marked.parse(content) as string).trim());
						}
					} else if (status.startsWith("EVENT:START:")) {
						const tool = status.slice(12);
						lastTool = `Running tool: ${pc.bold(tool)}`;
						s.start(pc.gray(lastTool));
					} else if (status.startsWith("WARN:")) {
						p.log.warn(status.slice(5));
					}
				});
			} catch (error) {
				p.log.error(`Interrupted by error: ${String(error)}`);
			}
		}
	}
}
