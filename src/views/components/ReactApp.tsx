import { Box, Static, Text, useApp } from "ink";
import Spinner from "ink-spinner";
import TextInput from "ink-text-input";
import { marked } from "marked";
import TerminalRenderer from "marked-terminal";
import React, { useState } from "react";

marked.setOptions({
	renderer: new TerminalRenderer({
		width: process.stdout.columns ? process.stdout.columns - 5 : 80,
	}),
});

import { runAgentLoop } from "@/core/orchestrator";

type Message = {
	id: string;
	type: "user" | "think" | "agent" | "tool" | "warn" | "error";
	content: string;
};

// ========================
// Components
// ========================

const UserMessage = ({ content }: { content: string }) => (
    <Box width="100%" backgroundColor="#3e454a" paddingX={1}>
        <Text color="#3b82f6"> ❯ {content}</Text>
    </Box>
);

const ThinkMessage = ({ content }: { content: string }) => (
    <Box width="100%" flexDirection="column" backgroundColor="#262626" paddingX={1} marginTop={1}>
        {/* <Text italic color="#949494">Thinking</Text> */}
        <Text italic color="#B0B0B0">{content}</Text>
    </Box>
);

const AgentMessage = ({ content }: { content: string }) => {
	const renderedText = marked.parse(content) as string;
	return (
		<Box width="100%" flexDirection="column" paddingX={1} paddingY={0} marginTop={1}>
			<Text>{renderedText.trim()}</Text>
			<Box marginTop={1}>
				<Text color="#3a3d3b" dimColor>
					{"─".repeat((process.stdout.columns || 80) - 2)}
				</Text>
			</Box>
		</Box>
	);
};

// ========================
// Main App
// ========================

export default function App() {
	const { exit } = useApp();
	const [messages, setMessages] = useState<Message[]>([]);
	const [input, setInput] = useState("");
	const [isBusy, setIsBusy] = useState(false);
	const [status, setStatus] = useState("");

	const handleSubmit = async (query: string) => {
		const text = query.trim();
		if (!text) return;

		if (text.startsWith("/")) {
			const cmd = text.slice(1).toLowerCase();
			if (cmd === "exit" || cmd === "quit") {
				setMessages((prev) => [
					...prev,
					{ id: `${Date.now()}-user`, type: "user", content: text },
				]);
				setInput("");
				setIsBusy(true);
				setStatus("Goodbye!");
				setTimeout(() => exit(), 100);
				return;
			} else if (cmd === "help") {
				setMessages((prev) => [
					...prev,
					{ id: `${Date.now()}-user`, type: "user", content: text },
					{
						id: `${Date.now()}-agent`,
						type: "agent",
						content: "Available commands: \n/help Show help \n/exit or /quit Exit",
					},
				]);
				setInput("");
				return;
			}
		}

		// Add user message
		const userMsgId = Date.now().toString();
		setMessages((prev) => [
			...prev,
			{ id: userMsgId, type: "user", content: text },
		]);
		setInput("");
		setIsBusy(true);
		setStatus("Thinking...");

		try {
			await runAgentLoop(text, (statusMsg) => {
				const id = `${Date.now()}${Math.random()}`;
				if (statusMsg.startsWith("THINK:")) {
					setMessages((prev) => [
						...prev,
						{ id, type: "think", content: statusMsg.slice(6) },
					]);
					setStatus("Composing...");
				} else if (statusMsg.startsWith("CONTENT:")) {
					setMessages((prev) => [
						...prev,
						{ id, type: "agent", content: statusMsg.slice(8) },
					]);
				} else if (statusMsg.startsWith("EVENT:START:")) {
					const toolName = statusMsg.slice(12);
					setMessages((prev) => [
						...prev,
						{ id, type: "tool", content: toolName },
					]);
					setStatus(`Running tool: ${toolName}...`);
				} else if (statusMsg.startsWith("WARN:")) {
					setMessages((prev) => [
						...prev,
						{ id, type: "warn", content: statusMsg.slice(5) },
					]);
				}
			});
		} catch (e) {
			setMessages((prev) => [
				...prev,
				{
					id: `${Date.now()}err`,
					type: "error",
					content: String(e),
				},
			]);
		} finally {
			setIsBusy(false);
			setStatus("");
		}
	};

	return (
		<React.Fragment>
			<Static items={messages}>
				{(msg) => {
					if (msg.type === "user")
						return <UserMessage key={msg.id} content={msg.content} />;
					if (msg.type === "think")
						return <ThinkMessage key={msg.id} content={msg.content} />;
					if (msg.type === "tool")
						return (
							<Box key={msg.id} paddingX={1}>
								<Text color="#3a3d3b">⚙ Running tool: {msg.content}</Text>
							</Box>
						);
					if (msg.type === "warn")
						return (
							<Box key={msg.id} paddingX={2}>
								<Text color="yellow">⚠ {msg.content}</Text>
							</Box>
						);
					if (msg.type === "error")
						return (
							<Box key={msg.id} paddingX={2}>
								<Text color="red">Error: {msg.content}</Text>
							</Box>
						);
					return <AgentMessage key={msg.id} content={msg.content} />;
				}}
			</Static>

			{isBusy ? (
				<Box paddingX={2} paddingY={1}>
					<Text color="#7DD3A5">
						<Spinner type="dots" /> {status}
					</Text>
				</Box>
			) : (
				<Box paddingX={1} paddingY={0} marginTop={1} marginBottom={1} borderStyle="round" borderColor="#3b82f6" width="100%">
					<Text bold color="#3b82f6"> ❯ </Text>
					<Box paddingLeft={1} flexGrow={1}>
						<TextInput
							value={input}
							onChange={setInput}
							onSubmit={handleSubmit}
							placeholder="Ask CauProphet"
						/>
					</Box>
				</Box>
			)}
		</React.Fragment>
	);
}
