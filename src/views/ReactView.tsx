// Abandoned
import { render } from "ink";
import App from "./components/ReactApp";
import type { InteractionStrategy } from "./types";

export class ReactInteractionView implements InteractionStrategy {
	async start(): Promise<void> {
		render(<App />);
	}
}
