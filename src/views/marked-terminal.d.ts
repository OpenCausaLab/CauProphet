declare module "marked-terminal" {
	import type { Renderer } from "marked";

	const TerminalRenderer: new (options?: {
		width?: number;
		reflowText?: boolean;
	}) => Renderer;
	export default TerminalRenderer;
}
