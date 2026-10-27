export interface InteractionStrategy {
	start(): Promise<void>;
}
